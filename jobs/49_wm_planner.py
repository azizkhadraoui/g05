"""P01 closed loop: planning with our small latent world model on top of frozen G0.5 (LIBERO-Goal).

The robot is given a GOAL IMAGE of the target task instead of its instruction. Every 10 steps:
  1. G0.5 proposes one 32-step action chunk per candidate instruction (the 10 LIBERO-Goal commands);
  2. the world model (23_train_wm.py) rolls each chunk forward 4 x 0.4 s from the current DINOv3 features;
  3. the chunk whose predicted 1.6 s future is closest to the goal features is executed for 10 steps.
Modes:
  wm         world-model selection at every replan
  wm_commit  world-model selection for the first --commit_after replans, then commit to the best-scoring command
  random     a random candidate command at every replan (same pipeline, no world model)
  blank      G0.5 with an empty instruction (no goal information)
  oracle     G0.5 with the correct instruction (upper bound)
Goal features: mean pooled DINOv3 features of the final frames of the target task's training demos.
"""
import argparse
import asyncio
import glob
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

JOBS = os.path.dirname(os.path.abspath(__file__))
G = os.environ.get("G05_ROOT", "/export/home/kaziz/g05")
PROJECT = f"{G}/GalaxeaVLA"
sys.path.insert(0, JOBS)
sys.path.insert(0, PROJECT)
os.chdir(PROJECT)
import t4_fp16  # noqa: E402,F401
import lawam_hybrid as H  # noqa: E402
from experiments.libero.libero_eval_utils import (  # noqa: E402
    LIBERO_DUMMY_ACTION, LiberoGripperCommandState, build_libero_raw_obs, ensure_libero_config, extract_libero_images,
    libero_gripper_state_to_command, single_action_to_libero_action, LIBERO_GRIPPER_OPEN_COMMAND)
from experiments.libero.libero_eval_utils import _is_gripper_hold_sentinel  # noqa: E402
from experiments.libero.libero_vector_env import LiberoAsyncVectorEnv  # noqa: E402
from scripts.utils.policy_ws_client import MultiPolicyWSClient  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--server_uri", required=True)
p.add_argument("--mode", required=True, choices=["wm", "wm_commit", "random", "blank", "oracle"])
p.add_argument("--task_ids", default="0,1,2,3,4,5,6,7,8,9")
p.add_argument("--trials", type=int, default=10)
p.add_argument("--num_parallel", type=int, default=5)
p.add_argument("--max_steps", type=int, default=300)
p.add_argument("--num_steps_wait", type=int, default=20)
p.add_argument("--replan", type=int, default=10)
p.add_argument("--commit_after", type=int, default=3)
p.add_argument("--wm_ckpt", default=f"{G}/results/wm_large8k/wm.pt")
p.add_argument("--goal_type", default="timed", choices=["timed", "final"],
               help="timed: demo-average features 1.6 s ahead of the current time (a goal video); final: last frame only")
p.add_argument("--seed", type=int, default=7)
p.add_argument("--switch_offset", type=int, default=None, help="steering probe: start from task s=(t+off)%%10, goal of s until --switch_step, then t")
p.add_argument("--switch_step", type=int, default=40)
p.add_argument("--save_gifs", type=int, default=0, help="write captioned GIFs for the first N trials of each task")
p.add_argument("--output_dir", required=True)
a = p.parse_args()
os.makedirs(a.output_dir, exist_ok=True)
random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
dev = torch.device("cuda")
AMP = (lambda: torch.autocast("cuda", dtype=torch.float16)) if t4_fp16.T4_MODE else (lambda: torch.autocast("cuda", dtype=torch.bfloat16))


class WM(nn.Module):  # identical to 23_train_wm.py
    def __init__(self, width, depth, k):
        super().__init__()
        self.inp = nn.Linear(768, width)
        self.pos = nn.Parameter(torch.randn(64, width) * 0.02)
        self.act = nn.Sequential(nn.Linear(7 * k, width * 4), nn.GELU(), nn.Linear(width * 4, width * 4))
        self.act_pos = nn.Parameter(torch.randn(4, width) * 0.02)
        layer = nn.TransformerEncoderLayer(width, max(4, width // 64), width * 4, dropout=0.0, batch_first=True, norm_first=True)
        self.tf = nn.TransformerEncoder(layer, depth)
        self.norm = nn.LayerNorm(width)
        self.out = nn.Linear(width, 768)

    def forward(self, u, chunk):
        x = self.inp(u) + self.pos
        at = self.act(chunk).view(u.shape[0], 4, -1) + self.act_pos
        h = self.tf(torch.cat([x, at], 1))[:, :64]
        return u + self.out(self.norm(h))


K_WM = wm = f_mean = f_std = a_mean = a_std = lam = None


def init_models():
    """Loaded in the main process only (the vector env's spawned workers re-import this module)."""
    global K_WM, wm, f_mean, f_std, a_mean, a_std, lam
    ck = torch.load(a.wm_ckpt, map_location="cpu", weights_only=False)
    K_WM = ck["args"]["k"]
    wm = WM(ck["args"]["width"], ck["args"]["depth"], K_WM).to(dev).eval()
    wm.load_state_dict(ck["model"])
    f_mean, f_std = ck["f_mean"].to(dev), ck["f_std"].to(dev)
    a_mean, a_std = torch.tensor(ck["a_mean"], device=dev), torch.tensor(ck["a_std"], device=dev)
    lam = H.load_lawm()


def feats_of(images):
    x = torch.from_numpy(np.stack(images)).permute(0, 3, 1, 2).to(dev)
    with torch.no_grad():
        f = H.dino_from_uint8(lam, x)
    b = f.shape[0]
    return F.avg_pool2d(f.view(b, 16, 16, 768).permute(0, 3, 1, 2), 2).permute(0, 2, 3, 1).reshape(b, 64, 768).float()


# ---------------- goal features: final frames of each LIBERO-Goal task's demos (feature cache rows) ----------------
def goal_features(task_langs):
    import pandas as pd
    C = f"{G}/results/feat_cache"; DS = f"{G}/data/libero_merged_g05grip"
    m = np.load(f"{C}/meta.npz")
    d = pd.concat([pd.read_parquet(f, columns=["index", "task_index"]) for f in sorted(glob.glob(f"{DS}/data/*/*.parquet"))]).sort_values("index")
    tidx = d["task_index"].values
    tp = pd.read_parquet(f"{DS}/meta/tasks.parquet").reset_index()
    tcol = [c for c in tp.columns if tp[c].dtype == object][0]
    norm = lambda s: " ".join(str(s).lower().replace("_", " ").split())  # noqa: E731
    name2idx = {norm(r[tcol]): int(r["task_index"]) for _, r in tp.iterrows()}
    feats = np.load(f"{C}/feats.npy", mmap_mode="r")
    ep, fr, L = m["episode_index"], m["frame_index"], m["ep_length"]
    out = {}
    for lang in task_langs:
        rows_t = np.nonzero(tidx == name2idx[norm(lang)])[0]
        starts = rows_t[fr[rows_t] == 0]  # first row of each demo of this task (episodes are contiguous)
        if a.goal_type == "final":
            out[lang] = torch.from_numpy(np.asarray(feats[starts + L[starts] - 1], dtype=np.float32).mean(0))[None].to(dev)
        else:  # time-indexed reference every 0.4 s: mean over demos of the frame at that time (clamped to each demo's end)
            T = int(L[starts].max())
            ref = [np.asarray(feats[np.sort(starts + np.minimum(f_, L[starts] - 1))], dtype=np.float32).mean(0) for f_ in range(0, T, 8)]
            out[lang] = torch.from_numpy(np.stack(ref)).to(dev)
        print(f"goal '{lang}': {len(starts)} demos, {out[lang].shape[0]} reference points", flush=True)
    return out


def goal_at(goals, lang, step):
    ref = goals[lang]
    return ref[min((step + 32) // 8, ref.shape[0] - 1)]


def chunk_to_wm(chunk, obs, steps=32):
    """G0.5 action chunk (dict of [T,...]) -> [steps, 7] in the dataset convention (gripper 1 open / 0 close)."""
    pk = next(k for k in chunk if k.endswith("ee_pose")); gk = next((k for k in chunk if k.endswith("gripper")), None)
    pose = np.asarray(chunk[pk], np.float32).reshape(-1, 6)
    cur = 1.0 if libero_gripper_state_to_command(obs) == LIBERO_GRIPPER_OPEN_COMMAND else 0.0
    if gk is None:
        grip = np.full((len(pose),), cur, np.float32)
    else:
        grip = np.asarray(chunk[gk], np.float32).reshape(len(pose), -1)[:, 0].copy()
        last = cur
        for i, g in enumerate(grip):
            if _is_gripper_hold_sentinel(float(g)):
                grip[i] = last
            else:
                grip[i] = float(np.clip(g, 0, 1)); last = grip[i]
    act = np.concatenate([pose, grip[:, None]], 1)
    if len(act) < steps:
        act = np.concatenate([act, np.repeat(act[-1:], steps - len(act), 0)])
    return act[:steps]


def wm_scores(cur_feat, chunks_wm, goal):
    """cur_feat [64,768], chunks_wm [K,32,7] -> distance of the predicted 1.6 s future to the goal, [K]."""
    Kc = chunks_wm.shape[0]
    with torch.no_grad(), AMP():
        u = ((cur_feat - f_mean) / f_std)[None].expand(Kc, -1, -1)
        an = (torch.from_numpy(chunks_wm).to(dev) - a_mean) / a_std
        n_steps = chunks_wm.shape[1] // K_WM
        for s in range(n_steps):
            u = wm(u, an[:, s * K_WM:(s + 1) * K_WM].reshape(Kc, -1)).float()
        g = (goal - f_mean) / f_std
        return ((u - g[None]) ** 2).mean((1, 2)).cpu().numpy()


def write_gif(path, frames, caps, header, success):
    from PIL import Image, ImageDraw
    out = []
    for k, (fr_, cap) in enumerate(zip(frames, caps)):
        im = Image.fromarray(fr_).resize((288, 288))
        canvas = Image.new("RGB", (288, 288 + 44), (255, 255, 255))
        canvas.paste(im, (0, 44))
        d = ImageDraw.Draw(canvas)
        d.text((4, 2), header[:52], fill=(0, 0, 0))
        d.text((4, 16), cap[:52], fill=(0, 70, 160))
        if k == len(frames) - 1:
            d.text((4, 30), "SUCCESS" if success else "FAILURE", fill=(0, 140, 0) if success else (190, 0, 0))
        out.append(canvas)
    out += [out[-1]] * 8
    out[0].save(path, save_all=True, append_images=out[1:], duration=100, loop=0, optimize=True)


def single_from_chunk(chunk, s):
    out = {}
    for k, arr in chunk.items():
        arr = np.asarray(arr)
        out[k] = arr[min(s, arr.shape[0] - 1)] if arr.ndim >= 1 else arr
    return out


async def main():
    init_models()
    ensure_libero_config()
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    suite = benchmark.get_benchmark_dict()["libero_goal"]()
    langs = [suite.get_task(i).language for i in range(suite.n_tasks)]
    goals = goal_features(langs)
    P = a.num_parallel
    Kc = len(langs) if a.mode in ("wm", "wm_commit") else 1
    summary = {"mode": a.mode, "wm_ckpt": a.wm_ckpt, "tasks": {}}
    async with MultiPolicyWSClient(a.server_uri, P * len(langs), timeout_s=600) as client:
        for tid in [int(x) for x in a.task_ids.split(",")]:
            task = suite.get_task(tid)
            bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file

            def env_fn():
                e = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256); e.seed(a.seed); return e

            def dummy_fn():
                e = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256, has_renderer=False); e.seed(a.seed); return e
            src = (tid + a.switch_offset) % len(langs) if a.switch_offset is not None else tid
            init_states = suite.get_task_init_states(src)
            vec = await asyncio.to_thread(LiberoAsyncVectorEnv, env_fns=[env_fn] * P, dummy_env_fn=dummy_fn)
            results = []
            try:
                for t0 in range(0, a.trials, P):
                    ids = list(range(t0, min(t0 + P, a.trials)))
                    n = len(ids)
                    obs = await asyncio.to_thread(vec.set_init_state_each, [init_states[i % len(init_states)] for i in ids] + [init_states[0]] * (P - n))
                    gstate = [LiberoGripperCommandState.from_obs(o) for o in obs]
                    active = [i < n for i in range(P)]
                    for _ in range(a.num_steps_wait):
                        obs_b, _, _, _ = await asyncio.to_thread(vec.step_selective, active, [LIBERO_DUMMY_ACTION] * P)
                        obs = [ob if ob is not None else o for ob, o in zip(obs_b, obs)]
                    done = [False] * P; step = 0
                    gif_fr = [[] for _ in range(P)]; gif_cap = [[] for _ in range(P)]
                    picks = [[] for _ in range(P)]; score_sum = [np.zeros(len(langs)) for _ in range(P)]; committed = [None] * P
                    while step < a.max_steps and any(active[i] and not done[i] for i in range(P)):
                        live = [i for i in range(P) if active[i] and not done[i]]
                        # candidate commands per live env
                        cand = {}
                        for i in live:
                            if a.mode == "oracle":
                                cand[i] = [src if step < a.switch_step and src != tid else tid]
                            elif a.mode == "blank":
                                cand[i] = [None]
                            elif a.mode == "random":
                                cand[i] = [random.randrange(len(langs))]
                            elif a.mode == "wm_commit" and committed[i] is not None:
                                cand[i] = [committed[i]]
                            else:
                                cand[i] = list(range(len(langs)))
                        idx, raws = [], []
                        for i in live:
                            for j, c in enumerate(cand[i]):
                                raw, _ = build_libero_raw_obs(obs=obs[i], task_description="" if c is None else langs[c], embodiment_type="libero")
                                idx.append(i * len(langs) + j); raws.append(raw)
                        await client.infer_batch(idx, raws)
                        chosen = {}
                        multi = [i for i in live if len(cand[i]) > 1]
                        cur = dict(zip(multi, feats_of([extract_libero_images(obs[i])["image"] for i in multi]))) if multi else {}
                        for i in live:
                            chunks = [client.action_chunks[i * len(langs) + j] for j in range(len(cand[i]))]
                            if len(chunks) == 1:
                                chosen[i] = (chunks[0], cand[i][0])
                            else:
                                g_now = goal_at(goals, langs[src], step) if (src != tid and step < a.switch_step) else                                     goal_at(goals, langs[tid], step - (a.switch_step if src != tid else 0))
                                sc = wm_scores(cur[i], np.stack([chunk_to_wm(c, obs[i]) for c in chunks]).astype(np.float32), g_now)
                                score_sum[i] += (sc - sc.mean()) / (sc.std() + 1e-8)
                                b = int(np.argmin(sc))
                                chosen[i] = (chunks[b], cand[i][b])
                                if a.mode == "wm_commit" and len(picks[i]) + 1 >= a.commit_after:
                                    committed[i] = int(np.argmin(score_sum[i]))
                            picks[i].append(chosen[i][1])
                        for s in range(a.replan):
                            acts = [LIBERO_DUMMY_ACTION] * P; mask = [False] * P
                            for i in live:
                                if done[i]:
                                    continue
                                acts[i] = single_action_to_libero_action(single_from_chunk(chosen[i][0], s), obs=obs[i], gripper_state=gstate[i])
                                mask[i] = True
                            if not any(mask):
                                break
                            obs_b, _, dones, infos = await asyncio.to_thread(vec.step_selective, mask, acts)
                            for i in range(P):
                                if mask[i]:
                                    obs[i] = obs_b[i]
                                    if i < n and ids[i] % max(a.trials, 1) < a.save_gifs and step % 2 == 0:
                                        c_ = chosen[i][1]
                                        gif_fr[i].append(extract_libero_images(obs[i])["image"])
                                        gif_cap[i].append("G0.5 runs: " + ("(no instruction)" if c_ is None else langs[c_]))
                                    if dones[i] or infos[i].get("_done"):
                                        done[i] = True
                            step += 1
                    for i in range(n):
                        if gif_fr[i]:
                            os.makedirs(f"{a.output_dir}/gifs", exist_ok=True)
                            hdr = (f"GOAL VIDEO: {langs[tid]}" if a.mode.startswith("wm") else f"{a.mode.upper()}: target = {langs[tid]}")
                            write_gif(f"{a.output_dir}/gifs/{a.mode}_task{tid}_trial{ids[i]}_{'success' if done[i] else 'failure'}.gif",
                                      gif_fr[i], gif_cap[i], hdr, done[i])
                    for k, i in enumerate(range(n)):
                        pk = [x for x in picks[i] if x is not None]
                        results.append({"trial": ids[k], "success": bool(done[i]), "n_replans": len(picks[i]),
                                        "frac_target_chosen": (float(np.mean([x == tid for x in pk])) if pk else None),
                                        "picks": pk[:40]})
                    print(json.dumps({"task": tid, "trials_done": len(results), "succ": sum(r["success"] for r in results)}), flush=True)
            finally:
                await asyncio.to_thread(vec.close)
            summary["tasks"][tid] = {"task": langs[tid], "successes": sum(r["success"] for r in results), "total": len(results),
                                     "frac_target_chosen": float(np.mean([r["frac_target_chosen"] for r in results if r["frac_target_chosen"] is not None])) if a.mode not in ("blank",) else None,
                                     "trials": results}
            json.dump(summary, open(f"{a.output_dir}/planner_results.json", "w"), indent=2)
    tot = sum(v["successes"] for v in summary["tasks"].values()); n = sum(v["total"] for v in summary["tasks"].values())
    summary["success_rate"] = tot / max(n, 1)
    json.dump(summary, open(f"{a.output_dir}/planner_results.json", "w"), indent=2)
    print(json.dumps({"mode": a.mode, "success": f"{tot}/{n}"}), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
