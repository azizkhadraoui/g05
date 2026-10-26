"""Showcase episode: one LIBERO kitchen mission that needs the high-level policy, memory and the world model together.

Mission (user request):
  "Warm up the bowl: put it on the stove and turn the stove on. Then tidy up so the kitchen looks like this picture."
  0. HIGH-LEVEL POLICY  RoboBrain 2.5 turns the spoken part into a sequence of G0.5 commands (zero-shot, text + scene).
  1. MEMORY             G0.5 runs command 1; the command stream drops at 2 s; the memory module (task decoder over the last
                        8 s of DINOv3 features, MEM-1) recognises the ongoing task and re-issues it.
  2. EXECUTION          command 2.
  3. WORLD MODEL        the last step exists only as a picture/goal video: every 0.5 s G0.5 proposes one chunk per kitchen
                        command, the 43M world model imagines each 1.6 s ahead, the chunk closest to the goal video runs.
Stage completion uses the simulator's goal check for the current step (moving on to the next stage only).
Logs every step for 74_showcase_video.py. Output: results/showcase/run_<k>.npz
"""
import argparse, asyncio, glob, json, os, sys, time
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

p = argparse.ArgumentParser()
p.add_argument("--server_uri", required=True)
p.add_argument("--episodes", type=int, default=6)
p.add_argument("--num_parallel", type=int, default=3)
p.add_argument("--vlm", default=f"{G}/models/RoboBrain2.5-4B")
p.add_argument("--vlm_device", default="cuda:1")
p.add_argument("--drop_at", type=int, default=40, help="step (within stage 1) at which the command stream is lost")
p.add_argument("--goal_task", type=int, default=2, help="LIBERO-Goal task shown only as a goal video (stage 3)")
p.add_argument("--budget", type=int, default=320)
p.add_argument("--output_dir", default=f"{G}/results/showcase")
p.add_argument("--mission", default="Put the wine bottle away on top of the cabinet and switch on the stove. "
               "Then set the table so it looks like the picture I'm showing you.")
p.add_argument("--init_task", type=int, default=2, help="initial state taken from this LIBERO-Goal task (shared scene)")
p.add_argument("--home", default="goal", choices=["none", "goal", "all"],
               help="scripted retract to the rest pose before the picture step (goal) or before every new step (all)")
p.add_argument("--align", default="time", choices=["time", "feat"],
               help="goal video index: time since the step started (as 49_wm_planner.py) or nearest demo moment")
a = p.parse_args()
os.makedirs(a.output_dir, exist_ok=True)
dev = torch.device("cuda:1" if torch.cuda.device_count() > 1 else "cuda:0")
MISSION = a.mission
AMP = (lambda: torch.autocast("cuda", dtype=torch.float16)) if t4_fp16.T4_MODE else (lambda: torch.autocast("cuda", dtype=torch.bfloat16))


class WM(nn.Module):  # 23_train_wm.py architecture
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


def make_env_fn(bddl, goal_bddls, seed):
    def _fn():
        from libero.libero.envs import OffScreenRenderEnv
        import libero.libero.envs.bddl_utils as BDDLUtils
        env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256); env.seed(seed)
        P_ = env.env; goals = [BDDLUtils.robosuite_parse_problem(g)["goal_state"] for g in goal_bddls]; _orig = env.step

        def _step(act):
            obs, r, done, info = _orig(act); info = dict(info or {})
            info["goal_sat"] = [bool(all(P_._eval_predicate(s) for s in gs)) for gs in goals]
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
    # ---- models: world model, DINOv3 (features), memory decoder, RoboBrain
    ck = torch.load(f"{G}/results/wm_large8k/wm.pt", map_location="cpu", weights_only=False); KW = ck["args"]["k"]
    wm = WM(ck["args"]["width"], ck["args"]["depth"], KW).to(dev).eval(); wm.load_state_dict(ck["model"])
    fm, fs = ck["f_mean"].to(dev), ck["f_std"].to(dev); am, as_ = torch.tensor(ck["a_mean"], device=dev), torch.tensor(ck["a_std"], device=dev)
    lam = H.load_lawm(device="cpu"); lam.vision_encoder.to(dev)
    mck = torch.load(f"{G}/results/mem_intent/offsets_decoder.pt", map_location="cpu", weights_only=False)
    mdec = Decoder(mck["n_frames"], mck["n_tasks"]).to(dev).eval(); mdec.load_state_dict(mck["model"])
    goal_suite = mck["suites"].index("libero_goal"); allowed = torch.tensor([s == goal_suite for s in mck["suite_id"]], device=dev)
    rb = RoboBrain(a.vlm, device=a.vlm_device)

    def feats(ims):
        x = torch.from_numpy(np.stack(ims)).permute(0, 3, 1, 2).to(dev)
        with torch.no_grad():
            f = H.dino_from_uint8(lam, x)
        b = f.shape[0]
        return F.avg_pool2d(f.view(b, 16, 16, 768).permute(0, 3, 1, 2), 2).permute(0, 2, 3, 1).reshape(b, 64, 768).float()

    # goal video (time-indexed demo features) for the picture-only step, as in 49_wm_planner.py
    C = f"{G}/results/feat_cache"; DS = f"{G}/data/libero_merged_g05grip"; m = np.load(f"{C}/meta.npz")
    dd = pd.concat([pd.read_parquet(f, columns=["index", "task_index"]) for f in sorted(glob.glob(f"{DS}/data/*/*.parquet"))]).sort_values("index")
    tidx = dd["task_index"].values; tp = pd.read_parquet(f"{DS}/meta/tasks.parquet").reset_index(); tcol = [c for c in tp.columns if tp[c].dtype == object][0]
    nrm = lambda s: " ".join(str(s).lower().replace("_", " ").split())  # noqa: E731
    name2idx = {nrm(r[tcol]): int(r["task_index"]) for _, r in tp.iterrows()}
    fc = np.load(f"{C}/feats.npy", mmap_mode="r"); L = m["ep_length"]; fr0 = m["frame_index"]
    rows_t = np.nonzero(tidx == name2idx[nrm(langs[a.goal_task])])[0]; starts = rows_t[fr0[rows_t] == 0]
    ref = torch.from_numpy(np.stack([np.asarray(fc[np.sort(starts + np.minimum(f_, L[starts] - 1))], np.float32).mean(0)
                                     for f_ in range(0, int(L[starts].max()), 8)])).to(dev)
    goal_rows = (starts[0] + np.minimum(np.arange(0, int(L[starts[0]]), 8), L[starts[0]] - 1)).astype(np.int64)  # one demo, for display
    names_mem = mck["task_names"]

    def mem_infer(buf):
        t = len(buf) - 1
        idx = [max(t - d, 0) for d in (160, 80, 40, 20)] + [t]
        x = torch.stack([buf[k] for k in idx]).half()[None]
        with torch.no_grad(), AMP():
            lg = mdec(x).float()[0].masked_fill(~allowed, -1e9)
        k = int(lg.argmax()); pr = torch.softmax(lg, -1)[k].item()
        return names_mem[k], pr, idx

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

    # ---- stage 0: high-level policy plans the spoken part of the mission (shared by all episodes)
    obs0_env = make_env_fn(bddls[0], bddls, 7)()
    o0 = obs0_env.set_init_state(suite.get_task_init_states(a.init_task)[0]); img0 = extract_libero_images(o0)["image"]; obs0_env.close()
    q = ("You control a robot arm in a kitchen. The user says: \"" + MISSION + "\"\nThe robot can run these skills:\n" +
         "\n".join(f"{k + 1}. {c}" for k, c in enumerate(langs)) +
         "\nThe request has exactly two steps given in words (the picture is handled separately). Which two skills carry them out, in order? Answer with exactly two skill numbers separated by a comma.")
    plan_txt = rb.ask([([img0], q)], max_new_tokens=16)[0]
    import re
    plan = [int(x) - 1 for x in re.findall(r"\d+", plan_txt) if 1 <= int(x) <= len(langs)][:2]
    print(json.dumps({"robobrain_plan_answer": plan_txt, "plan": [langs[k] for k in plan]}), flush=True)
    stages = [("lang", plan[0] if plan else 2, a.drop_at), ("lang", plan[1] if len(plan) > 1 else 7, None), ("goal", a.goal_task, None)]

    P = a.num_parallel
    vec = await asyncio.to_thread(LiberoAsyncVectorEnv, env_fns=[make_env_fn(bddls[1], bddls, 7)] * P, dummy_env_fn=make_env_fn(bddls[1], bddls, 7))
    async with MultiPolicyWSClient(a.server_uri, P * len(langs), timeout_s=600) as client:
        for e0 in range(0, a.episodes, P):
            n = min(P, a.episodes - e0)
            inits = [suite.get_task_init_states(a.init_task)[(e0 + j) % 50] for j in range(n)] + [suite.get_task_init_states(a.init_task)[0]] * (P - n)
            obs = await asyncio.to_thread(vec.set_init_state_each, inits)
            gst = [LiberoGripperCommandState.from_obs(o) for o in obs]
            act = [j < n for j in range(P)]
            for _ in range(20):
                ob, _, _, _ = await asyncio.to_thread(vec.step_selective, act, [LIBERO_DUMMY_ACTION] * P)
                obs = [x if x is not None else o for x, o in zip(ob, obs)]
            st = [dict(stage=0, step=0, held=0, total=0, cmd=langs[stages[0][1]], lost=False, recalled=None, align=0, home=False) for _ in range(P)]  # noqa: E501
            log = [dict(frames=[], stage=[], cmd=[], events=[], wm=[], mem=None) for _ in range(P)]
            buf = [[] for _ in range(P)]
            rest = [(np.asarray(o["robot0_eef_pos"], np.float64).copy(), np.asarray(o["robot0_eef_quat"], np.float64).copy()) for o in obs]

            async def go_home(js):
                # scripted retract between steps (no learning): lift, move over the rest pose, descend; gripper open
                wps = {}
                for j in js:
                    c = np.asarray(obs[j]["robot0_eef_pos"], np.float64); h = rest[j][0]; zs = max(c[2] + 0.05, h[2])
                    wps[j] = [np.array([c[0], c[1], zs]), np.array([h[0], h[1], zs]), h.copy()]
                    log[j]["events"].append((st[j]["total"], "home", "returning to rest pose"))
                for _ in range(140):
                    mask = [False] * P; acts = [LIBERO_DUMMY_ACTION] * P
                    for j in js:
                        if not wps[j]:
                            continue
                        c = np.asarray(obs[j]["robot0_eef_pos"], np.float64); e = wps[j][0] - c
                        while np.linalg.norm(e) < 0.015 and len(wps[j]) > 1:
                            wps[j].pop(0); e = wps[j][0] - c
                        er = TU.quat2axisangle(TU.mat2quat(TU.quat2mat(rest[j][1]) @ TU.quat2mat(np.asarray(obs[j]["robot0_eef_quat"])).T))
                        if len(wps[j]) == 1 and np.linalg.norm(e) < 0.01 and np.linalg.norm(er) < 0.05:
                            wps[j] = []; continue
                        acts[j] = np.concatenate([np.clip(e / 0.05, -1, 1), np.clip(er / 0.5, -1, 1), [-1.0]]).astype(np.float32).tolist()
                        mask[j] = True
                    if not any(mask):
                        break
                    ob, _, _, _ = await asyncio.to_thread(vec.step_selective, mask, acts)
                    for j in range(P):
                        if mask[j]:
                            obs[j] = ob[j]; st[j]["total"] += 1
                            log[j]["frames"].append(extract_libero_images(ob[j])["image"]); log[j]["stage"].append(st[j]["stage"])
                            log[j]["cmd"].append("(returning to rest pose)")
                for j in js:
                    gst[j] = LiberoGripperCommandState.from_obs(obs[j]); st[j]["home"] = False
                    pe = float(np.linalg.norm(rest[j][0] - np.asarray(obs[j]["robot0_eef_pos"])))
                    re_ = float(np.linalg.norm(TU.quat2axisangle(TU.mat2quat(TU.quat2mat(rest[j][1]) @ TU.quat2mat(np.asarray(obs[j]["robot0_eef_quat"])).T))))
                    log[j]["events"].append((st[j]["total"], "homed", f"rest pose reached: {pe * 100:.1f} cm, {np.degrees(re_):.0f} deg off"))

            while any(act):
                if a.home != "none" and any(st[j]["home"] for j in range(P) if act[j]):
                    await go_home([j for j in range(P) if act[j] and st[j]["home"]])
                live = [j for j in range(P) if act[j]]
                cand = {}
                for j in live:
                    kind, task, drop = stages[st[j]["stage"]]
                    if kind == "goal":
                        cand[j] = list(range(len(langs)))
                    else:
                        if drop is not None and st[j]["step"] >= drop and not st[j]["lost"]:
                            st[j]["lost"] = True
                            name, pr, idx = mem_infer(buf[j])
                            st[j]["recalled"] = name; st[j]["cmd"] = name
                            log[j]["mem"] = dict(total=st[j]["total"], inferred=name, conf=pr, frames=[int(st[j]["total"] - (len(buf[j]) - 1 - i)) for i in idx])
                            log[j]["events"].append((st[j]["total"], "lost", "command stream lost"))
                            log[j]["events"].append((st[j]["total"], "memory", f"memory recalls: {name}"))
                        cand[j] = [st[j]["cmd"]]
                idx_c, raws = [], []
                for j in live:
                    for c_i, c in enumerate(cand[j]):
                        text = langs[c] if isinstance(c, int) else c
                        raws.append(build_libero_raw_obs(obs=obs[j], task_description=text, embodiment_type="libero")[0]); idx_c.append(j * len(langs) + c_i)
                await client.infer_batch(idx_c, raws)
                chosen = {}
                for j in live:
                    chunks = [client.action_chunks[j * len(langs) + c_i] for c_i in range(len(cand[j]))]
                    if len(chunks) == 1:
                        chosen[j] = chunks[0]; continue
                    cur = feats([extract_libero_images(obs[j])["image"]])[0]
                    an = (torch.from_numpy(np.stack([chunk_to_wm(c, obs[j]) for c in chunks])).to(dev) - am) / as_
                    with torch.no_grad(), AMP():
                        u = ((cur - fm) / fs)[None].expand(len(chunks), -1, -1)
                        for s in range(4):
                            u = wm(u, an[:, s * KW:(s + 1) * KW].reshape(len(chunks), -1)).float()
                        if a.align == "time":  # goal video 1.6 s ahead of the time since this step started
                            gi = min((st[j]["step"] + 32) // 8, len(ref) - 1)
                        else:  # re-alignment: the demo moment most like the current scene (never moving backwards), then 1.6 s ahead
                            dist = (((ref - fm) / fs - ((cur - fm) / fs)[None]) ** 2).mean((1, 2))
                            st[j]["align"] = max(st[j]["align"], int(dist.argmin()))
                            gi = min(st[j]["align"] + 4, len(ref) - 1)
                        g = (ref[gi] - fm) / fs
                        sc = ((u - g[None]) ** 2).mean((1, 2)).cpu().numpy()
                    b = int(np.argmin(sc)); chosen[j] = chunks[b]; st[j]["cmd"] = langs[b]
                    log[j]["wm"].append(dict(total=st[j]["total"], scores=[float(x) for x in sc], choice=b,
                                             goal_row=int(goal_rows[min(gi, len(goal_rows) - 1)])))
                for s in range(10):
                    acts = [LIBERO_DUMMY_ACTION] * P; mask = [False] * P
                    for j in live:
                        if act[j]:
                            single = {k: (np.asarray(v)[min(s, np.asarray(v).shape[0] - 1)] if np.asarray(v).ndim >= 1 else v) for k, v in chosen[j].items()}
                            acts[j] = single_action_to_libero_action(single, obs=obs[j], gripper_state=gst[j]); mask[j] = True
                    if not any(mask):
                        break
                    ob, _, _, infos = await asyncio.to_thread(vec.step_selective, mask, acts)
                    imgs = [extract_libero_images(ob[j])["image"] if mask[j] else None for j in range(P)]
                    fe = feats([imgs[j] for j in range(P) if mask[j]]); fk = 0
                    advanced = False
                    for j in range(P):
                        if not mask[j]:
                            continue
                        obs[j] = ob[j]; buf[j].append(fe[fk].half()); fk += 1
                        st[j]["step"] += 1; st[j]["total"] += 1
                        log[j]["frames"].append(imgs[j]); log[j]["stage"].append(st[j]["stage"])
                        log[j]["cmd"].append("" if (st[j]["lost"] and stages[st[j]["stage"]][2] is not None and st[j]["recalled"] is None) else st[j]["cmd"])
                        task = stages[st[j]["stage"]][1]
                        st[j]["held"] = st[j]["held"] + 1 if infos[j]["goal_sat"][task] else 0
                        if st[j]["held"] >= 5:
                            log[j]["events"].append((st[j]["total"], "done", f"step {st[j]['stage'] + 1} complete: {langs[task]}"))
                            st[j]["stage"] += 1; st[j]["step"] = 0; st[j]["held"] = 0; st[j]["align"] = 0; advanced = True
                            buf[j] = []  # the memory module reads the frames since the current command started
                            if st[j]["stage"] >= len(stages):
                                act[j] = False
                            else:
                                st[j]["cmd"] = langs[stages[st[j]["stage"]][1]]
                                st[j]["home"] = a.home == "all" or stages[st[j]["stage"]][0] == "goal"
                        elif st[j]["step"] >= a.budget:
                            act[j] = False
                    if advanced:
                        break
            for j in range(n):
                k = e0 + j; ok = st[j]["stage"] >= len(stages)
                np.savez_compressed(f"{a.output_dir}/run_{k}.npz", frames=np.stack(log[j]["frames"]), stage=np.array(log[j]["stage"]),
                                    cmd=np.array(log[j]["cmd"]), success=ok, meta=json.dumps(dict(
                                        events=log[j]["events"], wm=log[j]["wm"], mem=log[j]["mem"], plan_answer=plan_txt,
                                        plan=[langs[x] for x in plan], stages=[[s_[0], langs[s_[1]], s_[2]] for s_ in stages],
                                        mission=MISSION, langs=langs, goal_task=langs[a.goal_task], stage_reached=st[j]["stage"])))
                print(json.dumps({"episode": k, "success": ok, "stage_reached": st[j]["stage"], "steps": st[j]["total"],
                                  "memory": log[j]["mem"]["inferred"] if log[j]["mem"] else None}), flush=True)
    await asyncio.to_thread(vec.close)


if __name__ == "__main__":
    asyncio.run(main())
