"""End-to-end mission benchmark for the full stack on the LIBERO-Goal kitchen (frozen G0.5 as the low-level policy), with
one module removed per arm, so each module's contribution is measured on the same missions and start states.

A mission has three steps and three events:
  step 1  spoken task; the command stream is lost --drop_at steps in        -> MEMORY re-issues the ongoing task
  step 2  spoken task; --change_at steps in the user changes their mind     -> PLANNER parses the new request,
          ("actually, don't X: do W instead")                                  STEERING nudges toward W for 2 s
  step 3  shown only as a picture of the goal                               -> WORLD MODEL picks the command whose
                                                                               imagined 1.6 s best matches the goal video
Arms (--arm):
  full         planner (RoboBrain 2.5) + memory (MEM-1 task decoder) + steering (pointing nudge) + world model,
               replanning every 10 steps
  no_planner   G0.5 receives the user's raw sentences (whole mission text; the change-of-mind sentence) as instruction
  no_memory    after the command loss G0.5 runs with a blank instruction
  no_steering  at the change of mind the new command is issued without the nudge
  no_wm        the picture step is mapped to a skill by the planner VLM looking at the goal picture, not by the world model
  long_chunks  as full, but the 32-step chunk is executed before replanning (long low-level horizon)
Step completion = the simulator's goal check for the current step (held 5 steps); mission success = all three steps.
Steering comparison (--steer, --target; same planner command and same --nudge_steps window after the change of mind):
  --steer nudge  straight-line reference toward 4 cm above a target point, for --nudge_steps steps
          --target table  the task's demo-averaged contact point (results/frs/targets.json): privileged, simulation only
          --target click  a simulated human click: that point projected into the agent view, --click_px pixel noise,
                          lifted back to 3D with the depth image
          --target vlm    the planner (RoboBrain 2.5) points at the contact spot in the agent view, lifted to 3D with depth
          --target random a random point over the table (control: does the target information matter at all?)
          --target_noise  std (m) of a fixed per-(mission, start state) 3D error added to the target
  --steer dsbc   the noise policy in DSBC_CKPT (server env) steers G0.5's sampler, no target (DSBC v2, SuperMimic-style)
  --steer home   the arm retracts to its rest pose, then G0.5 gets the new command (no target, no learned module)
          --retract home|lift   full retraction to the rest pose, or only straight up 8 cm (shortest, least collision risk)
          --grip open|keep|place gripper during the retraction: open (drops a held object), keep the current state, or
                                 if the fingers hold something (own finger sensor) lower it until it rests and release first
          --after_retract dsbc   the noise policy in DSBC_CKPT steers G0.5 for --nudge_steps after the retraction
  --steer none   the new command only
Every step records arm-link contacts with the scene (collision metric) and whether the fingers held something at the
change of mind. World model as a verifier (--wm_verify K): G0.5 proposes K chunks for the current command, the world
model imagines 1.6 s ahead for each, and the one closest to that skill's own demonstrations at the same time is executed;
--picture planner lets the planner VLM map the goal picture to a skill (the world model then never sees the target task).
--picture wm_pic   realistic world-model goal reading: every replan, G0.5 proposes a chunk for each of the 10 commands, the
                   world model imagines 1.6 s of each, and the chunk whose imagined scene is closest to the goal PICTURE
                   (the same single demo end frame from another episode that the planner sees) is executed; --pic_commit N
                   then keeps the most-chosen command (--picture wm compares against the target task's demo video: privileged)
--picture wm_end   command-level world model (143_endstate_wm.py): imagine the scene at the end of each of the 10 commands
                   from the current view, commit to the command whose imagined end state is closest to the goal picture
Built on 73_showcase.py (models, memory decoder, world-model goal matching, scripted rest pose before the picture step).
Output: results/e2e/<tag or arm>/episodes.jsonl (+ summary.json)
"""
import argparse, asyncio, glob, json, os, re, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

G = os.environ.get("G05_ROOT", "/export/home/kaziz/g05"); JOBS = f"{G}/jobs"
sys.path.insert(0, JOBS); sys.path.insert(0, f"{G}/GalaxeaVLA"); os.chdir(f"{G}/GalaxeaVLA")
import t4_fp16  # noqa: E402,F401
import lawam_hybrid as H  # noqa: E402
from mem_decoder import Decoder  # noqa: E402
from experiments.libero.libero_eval_utils import (  # noqa: E402
    LIBERO_DUMMY_ACTION, LIBERO_GRIPPER_OPEN_COMMAND, LiberoGripperCommandState, build_libero_raw_obs, ensure_libero_config,
    extract_libero_images, libero_gripper_state_to_command, single_action_to_libero_action, _is_gripper_hold_sentinel)
from experiments.libero.libero_vector_env import LiberoAsyncVectorEnv  # noqa: E402
from scripts.utils.policy_ws_client import MultiPolicyWSClient  # noqa: E402

ARMS = ("full", "no_planner", "no_memory", "no_steering", "no_wm", "long_chunks")
# (step 1, step 2 as first asked, step 2 after the change of mind, picture step) as LIBERO-Goal task indices
MISSIONS = [(2, 7, 0, 1), (0, 5, 9, 8), (9, 0, 7, 4), (7, 8, 2, 1), (5, 9, 1, 0), (1, 2, 0, 9), (6, 7, 9, 2), (0, 1, 8, 7)]

p = argparse.ArgumentParser()
p.add_argument("--server_uri", required=True)
p.add_argument("--arm", default="full", choices=ARMS)
p.add_argument("--inits", type=int, default=3, help="start states per mission")
p.add_argument("--missions", default="", help="comma-separated mission indices (default: all)")
p.add_argument("--num_parallel", type=int, default=3)
p.add_argument("--vlm", default=f"{G}/models/RoboBrain2.5-4B")
p.add_argument("--vlm_device", default="cuda:1")
p.add_argument("--drop_at", type=int, default=40)
p.add_argument("--change_at", type=int, default=40)
p.add_argument("--nudge_steps", type=int, default=40)
p.add_argument("--budget", type=int, default=320, help="policy steps per mission step")
p.add_argument("--output_dir", default=f"{G}/results/e2e")
p.add_argument("--tag", default="", help="output subdirectory (default: the arm)")
p.add_argument("--steer", default="nudge", choices=["nudge", "dsbc", "home", "none"])
p.add_argument("--target", default="table", choices=["table", "click", "vlm", "random"])
p.add_argument("--target_noise", type=float, default=0.0)
p.add_argument("--click_px", type=float, default=4.0, help="std (pixels, 256x256 view) of the simulated click")
p.add_argument("--retract", default="home", choices=["home", "lift", "frac", "learned"])
p.add_argument("--retract_frac", type=float, default=1.0, help="--retract frac: stop after this fraction of the retraction path")
p.add_argument("--retract_model", default="", help="--retract learned: student checkpoint (144_retract_student.py)")
p.add_argument("--retract_margin", type=float, default=0.05, help="--retract learned: shortest depth within this of the best")
p.add_argument("--init_offset", type=int, default=0, help="first start-state index per mission (held-out sets)")
p.add_argument("--max_stage", type=int, default=3, help="end a mission after this many steps (2: stop after the change of mind)")
p.add_argument("--save_change_obs", action="store_true", help="save cameras + state at the change of mind (change_obs/)")
p.add_argument("--grip", default="open", choices=["open", "keep", "place"])
p.add_argument("--after_retract", default="none", choices=["none", "dsbc"])
p.add_argument("--wm_verify", type=int, default=1, help="K > 1: world model picks among K chunks for the current command")
p.add_argument("--picture", default="wm", choices=["wm", "planner", "wm_pic", "wm_end"])
p.add_argument("--pic_commit", type=int, default=0, help="wm_pic: after N replans commit to the most-chosen command (0 = never)")
p.add_argument("--verify_temp", type=float, default=1.0, help="proposals 2..K start from N(0, T^2) noise (diverse candidates)")
p.add_argument("--verify_select", default="wm", choices=["wm", "random"], help="random: control for the world model's choice")
a = p.parse_args()
OUT = f"{a.output_dir}/{a.tag or a.arm}"; os.makedirs(OUT, exist_ok=True)
STEER = "none" if a.arm in ("no_steering", "no_planner") else a.steer
DEPTH = STEER == "nudge" and a.target in ("click", "vlm")
dev = torch.device("cuda:1" if torch.cuda.device_count() > 1 else "cuda:0")
AMP = (lambda: torch.autocast("cuda", dtype=torch.float16)) if t4_fp16.T4_MODE else (lambda: torch.autocast("cuda", dtype=torch.bfloat16))
EXEC = 32 if a.arm == "long_chunks" else 10
VRNG = np.random.default_rng(1234)  # diverse verifier proposals / random-selection control
nrm = lambda s: " ".join(str(s).lower().replace("_", " ").split())  # noqa: E731


class WM(nn.Module):  # 23_train_wm.py architecture (as 73_showcase.py)
    def __init__(self, width, depth, k):
        super().__init__()
        self.inp = nn.Linear(768, width); self.pos = nn.Parameter(torch.randn(64, width) * 0.02)
        self.act = nn.Sequential(nn.Linear(7 * k, width * 4), nn.GELU(), nn.Linear(width * 4, width * 4))
        self.act_pos = nn.Parameter(torch.randn(4, width) * 0.02)
        layer = nn.TransformerEncoderLayer(width, max(4, width // 64), width * 4, dropout=0.0, batch_first=True, norm_first=True)
        self.tf = nn.TransformerEncoder(layer, depth); self.norm = nn.LayerNorm(width); self.out = nn.Linear(width, 768)

    def forward(self, u, chunk):
        x = self.inp(u) + self.pos
        at = self.act(chunk).view(u.shape[0], 4, -1) + self.act_pos
        return u + self.out(self.norm(self.tf(torch.cat([x, at], 1))[:, :64]))


def make_env_fn(bddl, goal_bddls, seed, depth=False):
    DEPTH = depth  # noqa: N806 (shadows the module flag inside the worker process)

    def _fn():
        from libero.libero.envs import OffScreenRenderEnv
        import libero.libero.envs.bddl_utils as BDDLUtils
        env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256, camera_depths=DEPTH, horizon=6000)
        env.seed(seed)  # horizon: a mission (3 steps + rest pose) can exceed the default 1000 steps
        P_ = env.env; goals = [BDDLUtils.robosuite_parse_problem(g)["goal_state"] for g in goal_bddls]; _orig = env.step

        def _step(act):
            obs, r, done, info = _orig(act); info = dict(info or {})
            info["goal_sat"] = [bool(all(P_._eval_predicate(s) for s in gs)) for gs in goals]
            sim = P_.sim; n_arm = 0  # collision metric: arm links (not the gripper) touching anything outside the robot
            for ci in range(sim.data.ncon):
                c = sim.data.contact[ci]; n1, n2 = sim.model.geom_id2name(c.geom1) or "", sim.model.geom_id2name(c.geom2) or ""
                r1, r2 = n1.startswith("robot0_"), n2.startswith("robot0_")
                if r1 != r2 and not (n1 if r2 else n2).startswith(("robot0_", "gripper0_", "mount0_")):
                    n_arm += 1
            info["arm_contacts"] = n_arm
            if DEPTH:  # agent-view world->pixel matrix and depth-buffer range, to lift a clicked/pointed pixel to 3D
                from robosuite.utils.camera_utils import get_camera_transform_matrix
                sim = P_.sim; ext = float(sim.model.stat.extent)
                info["cam"] = (get_camera_transform_matrix(sim, "agentview", 256, 256).tolist(),
                               float(sim.model.vis.map.znear) * ext, float(sim.model.vis.map.zfar) * ext)
            return obs, r, False, info
        env.step = _step
        return env
    return _fn


async def main():
    ensure_libero_config()
    from libero.libero import benchmark, get_libero_path
    import pandas as pd
    from robosuite.utils import transform_utils as TU
    from robobrain import RoboBrain
    suite = benchmark.get_benchmark_dict()["libero_goal"]()
    langs = [suite.get_task(i).language for i in range(suite.n_tasks)]
    bddls = [str(Path(get_libero_path("bddl_files")) / suite.get_task(i).problem_folder / suite.get_task(i).bddl_file) for i in range(suite.n_tasks)]
    ck = torch.load(f"{G}/results/wm_large8k/wm.pt", map_location="cpu", weights_only=False); KW = ck["args"]["k"]
    wm = WM(ck["args"]["width"], ck["args"]["depth"], KW).to(dev).eval(); wm.load_state_dict(ck["model"])
    fm, fs = ck["f_mean"].to(dev), ck["f_std"].to(dev); am, as_ = torch.tensor(ck["a_mean"], device=dev), torch.tensor(ck["a_std"], device=dev)
    lam = H.load_lawm(device="cpu"); lam.vision_encoder.to(dev)
    mck = torch.load(f"{G}/results/mem_intent/offsets_decoder.pt", map_location="cpu", weights_only=False)
    mdec = Decoder(mck["n_frames"], mck["n_tasks"]).to(dev).eval(); mdec.load_state_dict(mck["model"])
    goal_suite = mck["suites"].index("libero_goal"); allowed = torch.tensor([s == goal_suite for s in mck["suite_id"]], device=dev)
    names_mem = mck["task_names"]
    rb = RoboBrain(a.vlm, device=a.vlm_device)
    targets = json.load(open(f"{G}/results/frs/targets.json"))
    if a.retract == "learned":  # retraction student (144_retract_student.py), on the DINOv3 encoder already loaded
        import importlib.util
        _sp = importlib.util.spec_from_file_location("retract_student", f"{G}/jobs/144_retract_student.py")
        RS = importlib.util.module_from_spec(_sp); _sp.loader.exec_module(RS)
        rs_model = RS.load(a.retract_model, dev, lam, H)
    if a.picture == "wm_end":  # command-level world model (143_endstate_wm.py)
        import importlib.util
        _sp = importlib.util.spec_from_file_location("endstate", f"{G}/jobs/143_endstate_wm.py")
        ES = importlib.util.module_from_spec(_sp); _sp.loader.exec_module(ES)
        es_ck = torch.load(f"{G}/results/endstate_wm/model.pt", map_location="cpu", weights_only=False)
        es_model = ES.EndState().to(dev).eval(); es_model.load_state_dict(es_ck["model"])

    def feats(ims):
        x = torch.from_numpy(np.stack(ims)).permute(0, 3, 1, 2).to(dev)
        with torch.no_grad():
            f = H.dino_from_uint8(lam, x)
        b = f.shape[0]
        return F.avg_pool2d(f.view(b, 16, 16, 768).permute(0, 3, 1, 2), 2).permute(0, 2, 3, 1).reshape(b, 64, 768).float()

    # goal videos (time-indexed mean demo features) and one demo end frame (the "picture") per LIBERO-Goal task
    C = f"{G}/results/feat_cache"; DS = f"{G}/data/libero_merged_g05grip"; m = np.load(f"{C}/meta.npz")
    dd = pd.concat([pd.read_parquet(f, columns=["index", "task_index"]) for f in sorted(glob.glob(f"{DS}/data/*/*.parquet"))]).sort_values("index")
    tidx = dd["task_index"].values; tp = pd.read_parquet(f"{DS}/meta/tasks.parquet").reset_index(); tcol = [c for c in tp.columns if tp[c].dtype == object][0]
    name2idx = {nrm(r[tcol]): int(r["task_index"]) for _, r in tp.iterrows()}
    fc = np.load(f"{C}/feats.npy", mmap_mode="r"); L = m["ep_length"]; fr0 = m["frame_index"]
    ref, pic = {}, {}

    def goal_video(task):
        if task not in ref:
            rows_t = np.nonzero(tidx == name2idx[nrm(langs[task])])[0]; starts = rows_t[fr0[rows_t] == 0]
            ref[task] = torch.from_numpy(np.stack([np.asarray(fc[np.sort(starts + np.minimum(f_, L[starts] - 1))], np.float32).mean(0)
                                                   for f_ in range(0, int(L[starts].max()), 8)])).to(dev)
        return ref[task]

    pic_f = {}

    def goal_pic_feats(task):  # DINOv3 features of the goal picture (--picture wm_pic)
        if task not in pic_f:
            pic_f[task] = feats([goal_picture(task)])[0].to(dev)
        return pic_f[task]

    def goal_picture(task):  # last frame of one demonstration of the task (LeRobot v3: episodes packed in AV1 files)
        cached = f"{C}/goal_pic_ds{name2idx[nrm(langs[task])]}.npy"  # 142_goal_pics.py (no video decoding at run time)
        if task not in pic and os.path.exists(cached):
            pic[task] = np.load(cached)
        if task not in pic:
            from torchcodec.decoders import VideoDecoder
            rows_t = np.nonzero(tidx == name2idx[nrm(langs[task])])[0]; st0 = int(rows_t[fr0[rows_t] == 0][0])
            ep = int(pd.concat([pd.read_parquet(f, columns=["index", "episode_index"]) for f in sorted(glob.glob(f"{DS}/data/*/*.parquet"))])
                     .set_index("index").loc[st0, "episode_index"])
            me = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"{DS}/meta/episodes/*/*.parquet"))]).set_index("episode_index")
            key = "videos/observation.images.image/"
            row = me.loc[ep]
            path = f"{DS}/videos/observation.images.image/chunk-{int(row[key + 'chunk_index']):03d}/file-{int(row[key + 'file_index']):03d}.mp4"
            t_end = float(row[key + "to_timestamp"]) - 0.1
            pic[task] = VideoDecoder(path).get_frame_played_at(t_end).data.permute(1, 2, 0).numpy()
        return pic[task]

    def mem_infer(buf):
        t = len(buf) - 1
        idx = [max(t - d, 0) for d in (160, 80, 40, 20)] + [t]
        x = torch.stack([buf[k] for k in idx]).half()[None]
        with torch.no_grad(), AMP():
            lg = mdec(x).float()[0].masked_fill(~allowed, -1e9)
        k = int(lg.argmax())
        return names_mem[k], torch.softmax(lg, -1)[k].item()

    def chunk_to_wm(chunk, obs):
        pk = next(k for k in chunk if k.endswith("ee_pose")); gk = next((k for k in chunk if k.endswith("gripper")), None)
        pose = np.asarray(chunk[pk], np.float32).reshape(-1, 6)
        cur = 1.0 if libero_gripper_state_to_command(obs) == LIBERO_GRIPPER_OPEN_COMMAND else 0.0
        grip = np.full(len(pose), cur, np.float32) if gk is None else np.asarray(chunk[gk], np.float32).reshape(len(pose), -1)[:, 0].copy()
        last = cur
        for i, g_ in enumerate(grip):
            grip[i] = last if _is_gripper_hold_sentinel(float(g_)) else float(np.clip(g_, 0, 1)); last = grip[i]
        act = np.concatenate([pose, grip[:, None]], 1)
        return np.concatenate([act, np.repeat(act[-1:], max(0, 32 - len(act)), 0)])[:32]

    def coarse_ref(obs, task, point, horizon=32):  # pointing nudge: straight line to 4 cm above the target point (95_frs_client.py)
        T = targets["tasks"][nrm(langs[task])]; k = np.asarray(targets["k"], np.float64)
        target = np.asarray(point, np.float64) + np.array([0.0, 0.0, 0.04]); pos = np.asarray(obs["robot0_eef_pos"], np.float64).copy()
        out = np.zeros((horizon, 7), np.float32); out[:, 6] = targets["gripper_open"]
        for h in range(horizon):
            units = (target - pos) / k; n = float(np.linalg.norm(units))
            step = units / n * min(T["speed"], n) if n > 1e-6 else np.zeros(3)
            out[h, :3] = step; pos = pos + step * k
        return out

    cal = {}  # pixel convention of robosuite's projection vs the raw agent-view array (calibrated once on object positions)

    def project(k, p):  # world point -> raw agent-view pixel (row, col)
        M = np.asarray(cam[k][0]); v = M @ np.append(np.asarray(p, np.float64), 1.0); pc, pr = v[0] / v[2], v[1] / v[2]
        fv, fh = cal["conv"]
        return (255 - pr if fv else pr), (255 - pc if fh else pc)

    def lift(k, r, c):  # raw agent-view pixel (row, col) -> world point on the visible surface, from the depth image
        M, near, far = cam[k]; d = np.asarray(obs[k]["agentview_depth"], np.float64).reshape(256, 256)
        z = near / (1.0 - d[int(np.clip(round(r), 0, 255)), int(np.clip(round(c), 0, 255))] * (1.0 - near / far))
        fv, fh = cal["conv"]; pr, pc = (255 - r if fv else r), (255 - c if fh else c)
        return (np.linalg.inv(np.asarray(M)) @ np.array([pc * z, pr * z, z, 1.0]))[:3]

    def calibrate(k):
        pts = [np.asarray(v, np.float64) for kk, v in obs[k].items() if kk.endswith("_pos") and "_to_" not in kk and np.asarray(v).shape == (3,)]
        errs = {}
        for conv in ((0, 0), (0, 1), (1, 0), (1, 1)):
            cal["conv"] = conv; e = []
            for p_ in pts:
                r, c = project(k, p_)
                if 0 <= r < 256 and 0 <= c < 256:
                    e.append(float(np.linalg.norm(lift(k, r, c) - p_)))
            errs[conv] = float(np.median(e)) if e else 9.9
        cal["conv"] = min(errs, key=errs.get)
        print(json.dumps({"camera_calibration_median_err_m": {str(c_): round(v, 3) for c_, v in errs.items()}, "chosen": cal["conv"],
                          "n_points": len(pts)}), flush=True)

    def steer_target(k, task, mi, j):  # nudge target for the new command -> (point or None, record)
        tbl = np.asarray(targets["tasks"][nrm(langs[task])]["point"], np.float64)
        rng = np.random.default_rng(1000 * mi + j); info = dict(source=a.target)
        if a.target == "table":
            p_ = tbl.copy()
        elif a.target == "random":  # anywhere over the table, at a typical grasp height
            p_ = np.array([rng.uniform(-0.30, 0.15), rng.uniform(-0.30, 0.30), 0.95])
        else:
            if "conv" not in cal:
                calibrate(k)
            if a.target == "click":
                r, c = project(k, tbl); r, c = r + rng.normal(0, a.click_px), c + rng.normal(0, a.click_px)
            else:
                q = (f"The robot must {langs[task]}. Point to the spot that the robot gripper should reach first: the object to pick up, "
                     "or the handle, knob or object to push or turn. Please provide its 2D coordinates. Your answer should be formatted "
                     "as a tuple, i.e. [(x, y)], where the tuple contains the x and y coordinates of a point satisfying the conditions above.")
                ans = rb.ask([([extract_libero_images(obs[k])["image"]], q)], max_new_tokens=24)[0]; info["answer"] = ans
                m_ = re.search(r"\(\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\)", ans)
                if not m_:
                    return None, info
                ux, uy = float(m_.group(1)) / 1000.0 * 256, float(m_.group(2)) / 1000.0 * 256  # RoboBrain 2.5 points are 0-1000
                r, c = 255 - uy, 255 - ux  # the planner sees the agent view rotated 180 degrees (extract_libero_images)
                info["pixel_upright"] = [round(ux, 1), round(uy, 1)]
            p_ = lift(k, r, c)
        if a.target_noise > 0:
            p_ = p_ + rng.normal(0.0, a.target_noise, 3)
        info.update(point=[round(float(x), 4) for x in p_], err_cm=round(float(np.linalg.norm(p_ - tbl)) * 100, 1))
        return p_, info

    skills = "\n".join(f"{k + 1}. {c}" for k, c in enumerate(langs))

    def plan_mission(img, text):  # -> [skill, skill]
        q = (f"You control a robot arm in a kitchen. The user says: \"{text}\"\nThe robot can run these skills:\n{skills}\n"
             "The request has exactly two steps given in words (a picture is handled separately). Which two skills carry them out, "
             "in order? Answer with exactly two skill numbers separated by a comma.")
        ans = rb.ask([([img], q)], max_new_tokens=16)[0]
        return [int(x) - 1 for x in re.findall(r"\d+", ans) if 1 <= int(x) <= len(langs)][:2], ans

    def plan_change(img, text):  # -> skill
        q = (f"You control a robot arm in a kitchen. While the robot works, the user says: \"{text}\"\nThe robot can run these skills:\n"
             f"{skills}\nWhich single skill should the robot do now? Answer with one skill number.")
        ans = rb.ask([([img], q)], max_new_tokens=8)[0]
        ks = [int(x) - 1 for x in re.findall(r"\d+", ans) if 1 <= int(x) <= len(langs)]
        return (ks[0] if ks else None), ans

    def plan_picture(img, goal_img):  # no_wm arm: the VLM maps the goal picture to a skill
        q = ("The first image is the current kitchen, the second is how the user wants it to look. The robot can run these skills:\n"
             f"{skills}\nWhich single skill turns the current kitchen into the second picture? Answer with one skill number.")
        ans = rb.ask([([img, goal_img], q)], max_new_tokens=8)[0]
        ks = [int(x) - 1 for x in re.findall(r"\d+", ans) if 1 <= int(x) <= len(langs)]
        return (ks[0] if ks else None), ans

    mids = [int(x) for x in a.missions.split(",")] if a.missions else list(range(len(MISSIONS)))
    jobs = [(mi, j) for mi in mids for j in range(a.init_offset, a.init_offset + a.inits)]
    if os.path.exists(f"{OUT}/episodes.jsonl"):  # resume: a continuation job skips the missions already recorded
        done = {(e["mission"], e["init"]) for e in map(json.loads, open(f"{OUT}/episodes.jsonl"))}
        jobs = [jb for jb in jobs if jb not in done]
        print(json.dumps({"resume": True, "already_done": len(done), "remaining": len(jobs)}), flush=True)
    P = a.num_parallel
    vec = await asyncio.to_thread(LiberoAsyncVectorEnv, env_fns=[make_env_fn(bddls[1], bddls, 7, DEPTH)] * P,
                                  dummy_env_fn=make_env_fn(bddls[1], bddls, 7, DEPTH))
    fout = open(f"{OUT}/episodes.jsonl", "a")
    async with MultiPolicyWSClient(a.server_uri, P * len(langs), timeout_s=600) as client:
        for b0 in range(0, len(jobs), P):
            batch = jobs[b0:b0 + P]; n = len(batch)
            inits = [suite.get_task_init_states(MISSIONS[mi][0])[(7 + 11 * j) % 50] for mi, j in batch]
            inits += [inits[0]] * (P - n)
            obs = list(await asyncio.to_thread(vec.set_init_state_each, inits))
            gst = [LiberoGripperCommandState.from_obs(o) for o in obs]
            act = [k < n for k in range(P)]; cam = [None] * P
            for _ in range(20):
                ob, _, _, inf = await asyncio.to_thread(vec.step_selective, act, [LIBERO_DUMMY_ACTION] * P)
                obs = [x if x is not None else o for x, o in zip(ob, obs)]
                cam = [(i_ or {}).get("cam", c_) for i_, c_ in zip(inf, cam)]
            rest = [(np.asarray(o["robot0_eef_pos"], np.float64).copy(), np.asarray(o["robot0_eef_quat"], np.float64).copy()) for o in obs]
            st, rec = [], []
            for k in range(P):
                mi, j = batch[k] if k < n else batch[0]
                s1, s2, w, gtask = MISSIONS[mi]
                text = f"{langs[s1].capitalize()}, and then {langs[s2]}. After that, make the kitchen look like this picture."
                change = f"Actually, don't {langs[s2]}: {langs[w]} instead."
                img0 = extract_libero_images(obs[k])["image"]
                if a.arm == "no_planner":
                    cmds, pans = [text, text], "(no planner: raw request)"
                else:
                    pl, pans = plan_mission(img0, text)
                    cmds = ([langs[x] for x in pl] + ["", ""])[:2]  # an unparsable answer leaves a blank command (counted as a planner error)
                st.append(dict(stage=0, step=0, held=0, total=0, cmd=cmds[0], lost=False, changed=False, nudge=0, home=False, picked=None,
                               target=None))
                rec.append(dict(mission=mi, init=j, arm=a.tag or a.arm, text=text, change=change, plan_answer=pans, plan=cmds,
                                plan_ok=(a.arm == "no_planner") or (cmds[0] == langs[s1] and cmds[1] == langs[s2]),
                                steps_done=[], mem=None, change_plan=None, picture=None, wm_choices=[],
                                steer_cfg=dict(steer=STEER, target=a.target, target_noise=a.target_noise), steer=None))
            stage_task = lambda k: [MISSIONS[batch[k][0]][0], MISSIONS[batch[k][0]][2] if st[k]["changed"] else MISSIONS[batch[k][0]][1],  # noqa: E731
                                    MISSIONS[batch[k][0]][3]][st[k]["stage"]]
            buf = [[] for _ in range(P)]

            def fingers_closed(k):  # the robot's own finger sensor: holding / grasping when well below the widest opening
                w = float(np.sum(np.abs(np.asarray(obs[k]["robot0_gripper_qpos"]).reshape(-1)[:2])))
                st[k]["wmax"] = max(st[k].get("wmax", 0.0), w)
                return w < 0.6 * st[k]["wmax"]

            def change_obs(k, old_task):  # what the robot sees and feels at the change of mind (retraction student input)
                ims = extract_libero_images(obs[k])
                return dict(image=ims["image"], wrist=ims["wrist_image"], eef_pos=np.asarray(obs[k]["robot0_eef_pos"], np.float32),
                            eef_quat=np.asarray(obs[k]["robot0_eef_quat"], np.float32), rest_pos=rest[k][0].astype(np.float32),
                            gripper_qpos=np.asarray(obs[k]["robot0_gripper_qpos"], np.float32).reshape(-1), held=fingers_closed(k),
                            old_cmd=old_task, new_cmd=langs.index(st[k]["cmd"]) if st[k]["cmd"] in langs else -1, step=st[k]["step"])

            def waypoints(k, mode, frac=1.0):
                c = np.asarray(obs[k]["robot0_eef_pos"], np.float64); h = rest[k][0]
                if mode == "lift":
                    return [c + np.array([0.0, 0.0, 0.08])]
                zs = max(c[2] + 0.05, h[2])
                pts = [np.array([c[0], c[1], zs]), np.array([h[0], h[1], zs]), h.copy()]
                if frac >= 1.0:
                    return pts
                seg = [c] + pts; out = []  # partial retraction: stop after a fraction of the path length
                lens = [float(np.linalg.norm(seg[i + 1] - seg[i])) for i in range(3)]; left = frac * sum(lens)
                for i in range(3):
                    if left <= lens[i]:
                        out.append(seg[i] + (seg[i + 1] - seg[i]) * (left / max(lens[i], 1e-9))); break
                    out.append(seg[i + 1]); left -= lens[i]
                return out

            async def go_home(js):
                """Scripted retraction: to the rest pose before the picture step (as 73_showcase.py), or after a change of
                mind with --retract (home | lift) and --grip (open: drop; keep: hold the current grip; place: if the
                fingers hold something, lower it until it rests and release before moving)."""
                wps, grip, phase, aux = {}, {}, {}, {}
                def orient_target(k, frac):  # partial retractions turn only part of the way to the rest orientation
                    q = np.asarray(obs[k]["robot0_eef_quat"], np.float64)
                    return rest[k][1] if frac >= 1.0 else TU.quat_slerp(q, rest[k][1], frac)

                for k in js:
                    mode, gmode, held = st[k].get("rmode", "home"), st[k].get("grip", "open"), fingers_closed(k)
                    fr = st[k].get("rfrac", 1.0) if mode == "frac" else 1.0
                    wps[k] = waypoints(k, mode, fr); st[k]["rcont"] = 0
                    grip[k] = (1.0 if held else -1.0) if gmode == "keep" else -1.0
                    phase[k] = "place" if (gmode == "place" and held) else "move"
                    aux[k] = dict(t=0, z0=float(obs[k]["robot0_eef_pos"][2]), zs=[], mode=mode, frac=fr, q=orient_target(k, fr))
                for _ in range(220):
                    mask = [False] * P; acts = [LIBERO_DUMMY_ACTION] * P
                    for k in js:
                        c = np.asarray(obs[k]["robot0_eef_pos"], np.float64)
                        if phase[k] == "place":  # lower the held object until it stops descending (resting), max 6 cm
                            x = aux[k]; x["t"] += 1; x["zs"].append(c[2])
                            stalled = len(x["zs"]) > 4 and x["zs"][-5] - c[2] < 0.002
                            if x["t"] <= 40 and c[2] > x["z0"] - 0.06 and not stalled:
                                acts[k] = [0.0, 0.0, -0.5, 0.0, 0.0, 0.0, 1.0]; mask[k] = True; continue
                            phase[k], x["t"] = "release", 0
                        if phase[k] == "release":  # open the fingers in place, then move away
                            aux[k]["t"] += 1
                            if aux[k]["t"] <= 12:
                                acts[k] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]; mask[k] = True; continue
                            phase[k], grip[k], wps[k] = "move", -1.0, waypoints(k, aux[k]["mode"], aux[k]["frac"])
                            aux[k]["q"] = orient_target(k, aux[k]["frac"])
                        if not wps[k]:
                            continue
                        e = wps[k][0] - c
                        while np.linalg.norm(e) < 0.015 and len(wps[k]) > 1:
                            wps[k].pop(0); e = wps[k][0] - c
                        if aux[k]["mode"] == "lift":  # straight up, orientation unchanged
                            er = np.zeros(3)
                        else:
                            er = TU.quat2axisangle(TU.mat2quat(TU.quat2mat(aux[k]["q"]) @ TU.quat2mat(np.asarray(obs[k]["robot0_eef_quat"])).T))
                        if len(wps[k]) == 1 and np.linalg.norm(e) < 0.01 and np.linalg.norm(er) < 0.05:
                            wps[k] = []; continue
                        acts[k] = np.concatenate([np.clip(e / 0.05, -1, 1), np.clip(er / 0.5, -1, 1), [grip[k]]]).astype(np.float32).tolist(); mask[k] = True
                    if not any(mask):
                        break
                    ob, _, _, inf = await asyncio.to_thread(vec.step_selective, mask, acts)
                    for k in range(P):
                        if mask[k]:
                            obs[k] = ob[k]; st[k]["total"] += 1
                            st[k]["rcont"] += int((inf[k] or {}).get("arm_contacts", 0) > 0)
                for k in js:
                    gst[k] = LiberoGripperCommandState.from_obs(obs[k]); st[k]["home"] = False
                    if st[k].pop("rmode", None) is not None:  # a change-of-mind retraction (not the picture-step one)
                        if isinstance(rec[k].get("steer"), dict):
                            rec[k]["steer"]["retract_contact_steps"] = st[k]["rcont"]
                        if st[k].pop("after", "none") == "dsbc":
                            st[k]["nudge"] = a.nudge_steps  # the noise policy steers the first chunks of the new command
                        st[k].pop("grip", None)

            for k in range(P):  # the fingers start fully open: reference width for fingers_closed()
                fingers_closed(k)
            while any(act):
                if any(st[k]["home"] for k in range(P) if act[k]):
                    await go_home([k for k in range(P) if act[k] and st[k]["home"]])
                live = [k for k in range(P) if act[k]]
                cand, extra, vt = {}, {}, {}
                for k in live:
                    mi = batch[k][0]; s1, s2, w, gtask = MISSIONS[mi]
                    if st[k]["stage"] == 0 and st[k]["step"] >= a.drop_at and not st[k]["lost"]:  # command stream lost
                        st[k]["lost"] = True
                        if a.arm == "no_memory":
                            st[k]["cmd"] = ""; rec[k]["mem"] = dict(recalled=None)
                        else:
                            name, pr = mem_infer(buf[k]); st[k]["cmd"] = name
                            rec[k]["mem"] = dict(recalled=name, conf=pr, correct=nrm(name) == nrm(langs[s1]))
                    if st[k]["stage"] == 1 and st[k]["step"] >= a.change_at and not st[k]["changed"]:  # change of mind
                        st[k]["changed"] = True; st[k]["cmd_since"] = st[k]["step"]
                        rec[k]["held_at_change"] = fingers_closed(k)
                        img = extract_libero_images(obs[k])["image"]
                        if a.arm == "no_planner":
                            st[k]["cmd"] = rec[k]["change"]; rec[k]["change_plan"] = dict(answer="(raw)", correct=None)
                        else:
                            kk, ans = plan_change(img, rec[k]["change"])
                            st[k]["cmd"] = langs[kk] if kk is not None else ""
                            rec[k]["change_plan"] = dict(answer=ans, correct=kk == w)
                        if STEER == "nudge" and nrm(st[k]["cmd"]) in targets["tasks"]:
                            pt, rec[k]["steer"] = steer_target(k, langs.index(st[k]["cmd"]), mi, batch[k][1])
                            if pt is not None:  # an unparsable pointing answer leaves the arm unsteered
                                st[k]["target"] = pt; st[k]["nudge"] = a.nudge_steps
                        elif STEER == "dsbc" and st[k]["cmd"] in langs:
                            rec[k]["steer"] = dict(source="noise_policy", ckpt=os.environ.get("DSBC_CKPT")); st[k]["nudge"] = a.nudge_steps
                        elif STEER == "home":  # abort the old motion: retract (next loop), then the new command
                            rec[k]["steer"] = dict(source="home", retract=a.retract, grip=a.grip, after=a.after_retract)
                            st[k].update(home=True, rmode=a.retract, grip=a.grip, after=a.after_retract)
                            if a.retract in ("frac", "learned"):
                                fr = a.retract_frac
                                if a.retract == "learned":  # the student picks the depth from the cameras, state and commands
                                    fr, probs = RS.choose(rs_model, change_obs(k, s2), a.retract_margin)
                                    rec[k]["steer"]["student_p"] = [round(float(v), 3) for v in probs]
                                st[k].update(rmode="frac", rfrac=fr); rec[k]["steer"]["frac"] = fr
                        rec[k]["total_at_change"] = st[k]["total"]
                        if a.save_change_obs:
                            os.makedirs(f"{OUT}/change_obs", exist_ok=True)
                            np.savez_compressed(f"{OUT}/change_obs/m{mi}_i{batch[k][1]}.npz", **change_obs(k, s2))
                    if st[k]["stage"] == 2:
                        if a.picture == "wm_pic" and a.pic_commit and st[k]["picked"] is None and len(st[k].get("votes", [])) >= a.pic_commit:
                            st[k]["picked"] = max(set(st[k]["votes"]), key=st[k]["votes"].count)  # commit to the world model's choice
                            rec[k]["picture"] = dict(answer=langs[st[k]["picked"]], correct=st[k]["picked"] == gtask, votes=st[k]["votes"])
                        if a.picture == "wm_end" and st[k]["picked"] is None:  # imagine each command's outcome once
                            ES_N = lambda x: (x.to(dev) - es_ck["mu"].to(dev)) / es_ck["sd"].to(dev)  # noqa: E731
                            x_now = ES_N(ES.pool(feats([extract_libero_images(obs[k])["image"]])[0].cpu().numpy()))
                            dist = ES.score(es_model, x_now, ES_N(ES.pool(goal_pic_feats(gtask).cpu().numpy())))
                            st[k]["picked"] = int(dist.argmin())
                            rec[k]["picture"] = dict(answer=langs[st[k]["picked"]], correct=st[k]["picked"] == gtask,
                                                     dist=[round(float(v), 4) for v in dist])
                        if a.picture in ("wm_pic", "wm_end") and st[k]["picked"] is not None:
                            cand[k] = [langs[st[k]["picked"]]]
                        elif a.arm == "no_wm" or a.picture == "planner":
                            if st[k]["picked"] is None:
                                kk, ans = plan_picture(extract_libero_images(obs[k])["image"], goal_picture(gtask))
                                st[k]["picked"] = kk if kk is not None else -1; rec[k]["picture"] = dict(answer=ans, correct=kk == gtask)
                            cand[k] = [langs[st[k]["picked"]]] if st[k]["picked"] >= 0 else [""]
                        else:
                            cand[k] = list(langs)
                    else:
                        cand[k] = [st[k]["cmd"]]
                    if a.wm_verify > 1 and len(cand[k]) == 1 and cand[k][0] in langs and st[k]["nudge"] == 0:
                        vt[k] = langs.index(cand[k][0]); cand[k] = cand[k] * a.wm_verify  # K proposals, same command
                idx_c, raws = [], []
                for k in live:
                    for c_i, text in enumerate(cand[k]):
                        raw = build_libero_raw_obs(obs=obs[k], task_description=text, embodiment_type="libero")[0]
                        if k in vt and c_i > 0 and a.verify_temp != 1.0:  # diverse proposals for the verifier
                            raw["frs_mode"] = "noise"; raw["z_init"] = (VRNG.standard_normal((32, 20)) * a.verify_temp).astype(np.float32)
                        if st[k]["nudge"] > 0 and len(cand[k]) == 1:
                            if STEER == "nudge":
                                raw["ref_action"] = coarse_ref(obs[k], langs.index(st[k]["cmd"]) if st[k]["cmd"] in langs else 0, st[k]["target"])
                                raw["frs_mode"] = "ref"
                            else:  # the noise policy alone steers G0.5's sampler (no reference, no target)
                                raw["frs_mode"] = "dsbc"
                        raws.append(raw); idx_c.append(k * len(langs) + c_i)
                await client.infer_batch(idx_c, raws)
                chosen = {}
                for k in live:
                    chunks = [client.action_chunks[k * len(langs) + c_i] for c_i in range(len(cand[k]))]
                    if len(chunks) == 1:
                        chosen[k] = chunks[0]; continue
                    if k in vt and a.verify_select == "random":  # control: same candidates, no world model
                        b = int(VRNG.integers(len(chunks))); chosen[k] = chunks[b]; rec[k]["wm_choices"].append(b); continue
                    # verification: the commanded skill's own demonstrations; picture step (--picture wm): the target task's
                    pic_goal = k not in vt and a.picture == "wm_pic"  # realistic: only the goal picture the user showed
                    gv = goal_video(vt.get(k, MISSIONS[batch[k][0]][3])) if not pic_goal else None
                    cur = feats([extract_libero_images(obs[k])["image"]])[0]
                    an = (torch.from_numpy(np.stack([chunk_to_wm(c, obs[k]) for c in chunks])).to(dev) - am) / as_
                    with torch.no_grad(), AMP():
                        u = ((cur - fm) / fs)[None].expand(len(chunks), -1, -1)
                        for s in range(4):
                            u = wm(u, an[:, s * KW:(s + 1) * KW].reshape(len(chunks), -1)).float()
                        if pic_goal:
                            g = (goal_pic_feats(MISSIONS[batch[k][0]][3]) - fm) / fs
                        else:
                            g = (gv[min((st[k]["step"] - st[k].get("cmd_since", 0) + 32) // 8, len(gv) - 1)] - fm) / fs
                        sc = ((u - g[None]) ** 2).mean((1, 2)).cpu().numpy()
                    b = int(np.argmin(sc)); chosen[k] = chunks[b]; rec[k]["wm_choices"].append(b)
                    if pic_goal:
                        st[k].setdefault("votes", []).append(b)
                n_exec = min(a.nudge_steps, EXEC) if any(st[k]["nudge"] > 0 for k in live) else EXEC
                for s in range(n_exec):
                    acts = [LIBERO_DUMMY_ACTION] * P; mask = [False] * P
                    for k in live:
                        if act[k] and not st[k]["home"]:  # an env waiting for its retraction does not move
                            single = {kk:(np.asarray(v)[min(s, np.asarray(v).shape[0] - 1)] if np.asarray(v).ndim >= 1 else v) for kk, v in chosen[k].items()}
                            acts[k] = single_action_to_libero_action(single, obs=obs[k], gripper_state=gst[k]); mask[k] = True
                    if not any(mask):
                        break
                    ob, _, _, infos = await asyncio.to_thread(vec.step_selective, mask, acts)
                    cam = [(i_ or {}).get("cam", c_) for i_, c_ in zip(infos, cam)]
                    imgs =[extract_libero_images(ob[k])["image"] if mask[k] else None for k in range(P)]
                    fe = feats([imgs[k] for k in range(P) if mask[k]]); fk = 0
                    advanced = False
                    for k in range(P):
                        if not mask[k]:
                            continue
                        obs[k] = ob[k]; buf[k].append(fe[fk].half()); fk += 1
                        st[k]["step"] += 1; st[k]["total"] += 1; st[k]["nudge"] = max(0, st[k]["nudge"] - 1)
                        st[k]["acont"] = st[k].get("acont", 0) + int((infos[k] or {}).get("arm_contacts", 0) > 0)
                        task = stage_task(k)
                        st[k]["held"] = st[k]["held"] + 1 if infos[k]["goal_sat"][task] else 0
                        if st[k]["held"] >= 5:
                            rec[k]["steps_done"].append(dict(stage=st[k]["stage"], task=langs[task], steps=st[k]["step"], total=st[k]["total"]))
                            st[k]["stage"] += 1; st[k]["step"] = 0; st[k]["held"] = 0; st[k]["cmd_since"] = 0; advanced = True; buf[k] = []
                            if st[k]["stage"] >= min(3, a.max_stage):
                                act[k] = False
                            else:
                                st[k]["cmd"] = rec[k]["plan"][1] if st[k]["stage"] == 1 else ""
                                st[k]["home"] = st[k]["stage"] == 2
                        elif st[k]["step"] >= a.budget:
                            act[k] = False
                    if advanced:
                        break
            for k in range(n):
                r = rec[k]; r.update(success=st[k]["stage"] >= 3, stage_reached=st[k]["stage"], total_steps=st[k]["total"],
                                     arm_contact_steps=st[k].get("acont", 0))
                fout.write(json.dumps(r) + "\n"); fout.flush()
                print(json.dumps({kk: r[kk] for kk in ("mission", "init", "arm", "success", "stage_reached", "plan_ok")}), flush=True)
    await asyncio.to_thread(vec.close)
    eps = [json.loads(l) for l in open(f"{OUT}/episodes.jsonl")]
    summ = {"arm": a.tag or a.arm, "episodes": len(eps), "mission_success": float(np.mean([e["success"] for e in eps])),
            "step_reached_mean": float(np.mean([e["stage_reached"] for e in eps])),
            "step1_done": float(np.mean([e["stage_reached"] >= 1 for e in eps])), "step2_done": float(np.mean([e["stage_reached"] >= 2 for e in eps])),
            "plan_ok": float(np.mean([bool(e["plan_ok"]) for e in eps])),
            "memory_correct": float(np.mean([bool((e["mem"] or {}).get("correct")) for e in eps if e["mem"] is not None])) if any(e["mem"] for e in eps) else None}
    json.dump(summ, open(f"{OUT}/summary.json", "w"), indent=1)
    print(json.dumps(summ), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
