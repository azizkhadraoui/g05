"""MEM-1: does episodic visual memory carry task information that the current frame lacks?

A small transformer reads pooled DINOv3 features (8x8 tokens per frame, from 22_cache_feats.py) and predicts which
LIBERO task is being executed. Evaluated within-suite (10-way; suite known), at frames >= 2 s into the episode
(the point where the instruction is withdrawn in the M02 probe), on held-out episodes (episode_index % 5 == 0).

Conditions (pre-registered):
  current   current frame only
  first     current + first frame of the episode
  offsets   current + frames 1, 2, 4, 8 s back                      (5 frames)
  snapshot  current + SM1 event snapshots (first + 3 most recent)  (5 frames, 3D-Mem-style selection)
Positive result: memory conditions beat `current` on the shared-scene suites (goal, object, spatial).
"""
import argparse
import glob
import json
import math
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mem_decoder import Decoder  # noqa: E402

G = os.environ.get("G05_ROOT", "/export/home/kaziz/g05")
C = f"{G}/results/feat_cache"
DS = f"{G}/data/libero_merged_g05grip"
SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]

p = argparse.ArgumentParser()
p.add_argument("--cond", required=True, choices=["current", "first", "offsets", "snapshot"])
p.add_argument("--steps", type=int, default=6000)
p.add_argument("--batch", type=int, default=256)
p.add_argument("--lr", type=float, default=3e-4)
p.add_argument("--min_frame", type=int, default=40)
p.add_argument("--heldout_mod", type=int, default=5)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--out", default=f"{G}/results/mem_intent")
p.add_argument("--eval_only", action="store_true", help="evaluate the saved decoder (no training)")
a = p.parse_args()
os.makedirs(a.out, exist_ok=True)
torch.manual_seed(a.seed); np.random.seed(a.seed)
dev = torch.device("cuda")
AMP16 = torch.cuda.get_device_capability()[0] < 8
t0 = time.time()

# ---------------- labels: task per frame, suite per task ----------------
m = np.load(f"{C}/meta.npz")
ep, fr = m["episode_index"], m["frame_index"]
N = len(ep)
d = pd.concat([pd.read_parquet(f, columns=["index", "task_index"]) for f in sorted(glob.glob(f"{DS}/data/*/*.parquet"))])
d = d.sort_values("index")
task_idx = d["task_index"].values.astype(np.int64)
tp = pd.read_parquet(f"{DS}/meta/tasks.parquet").reset_index()
tcol = [c for c in tp.columns if tp[c].dtype == object][0]  # the task string (LeRobot v3 keeps it as the index)
names = {int(r["task_index"]): r[tcol] for _, r in tp.iterrows()}
from libero.libero import benchmark  # noqa: E402

bd = benchmark.get_benchmark_dict()
norm = lambda s: " ".join(str(s).lower().replace("_", " ").split())  # noqa: E731
lang2suite = {}
for sname in SUITES:
    for t in bd[sname]().tasks:
        lang2suite[norm(t.language)] = sname
suite_of_task = {}
for ti, nm in names.items():
    suite_of_task[ti] = lang2suite.get(norm(nm))
unmatched = [names[t] for t, s in suite_of_task.items() if s is None]
n_tasks = max(names) + 1
suite_id = np.array([SUITES.index(suite_of_task[t]) if suite_of_task[t] else -1 for t in range(n_tasks)])
row_suite = suite_id[task_idx]
print(f"tasks={n_tasks} unmatched={unmatched[:5]}", flush=True)

# ---------------- memory frame selection ----------------
snap = np.load(f"{C}/snapshot_mem_rows.npy")


def frames_for(rows):
    if a.cond == "current":
        return rows[:, None]
    if a.cond == "first":
        return np.stack([rows - fr[rows], rows], 1)
    if a.cond == "offsets":
        return np.stack([rows - np.minimum(k, fr[rows]) for k in (160, 80, 40, 20)] + [rows], 1)
    return np.concatenate([snap[rows], rows[:, None]], 1)


n_frames = frames_for(np.array([1000])).shape[1]
feats = np.ascontiguousarray(np.load(f"{C}/feats.npy", mmap_mode="r"))
print(f"features in RAM {feats.shape} ({time.time() - t0:.0f}s)", flush=True)
elig = (fr >= a.min_frame) & (row_suite >= 0)
held = ep % a.heldout_mod == 0
train_rows = np.nonzero(elig & ~held)[0]
eval_rows = np.nonzero(elig & held)[0]


def batch_x(rows):
    idx = frames_for(rows)
    return torch.from_numpy(feats[idx.reshape(-1)].reshape(len(rows), idx.shape[1], 64, 768)).to(dev, non_blocking=True)


model = Decoder(n_frames, n_tasks).to(dev)
tag = a.cond if a.seed == 0 else f"{a.cond}_seed{a.seed}"
if a.eval_only:
    model.load_state_dict(torch.load(f"{a.out}/{tag}_decoder.pt", map_location="cpu", weights_only=False)["model"])
    a.steps = 0
opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.05)
scaler = torch.cuda.amp.GradScaler(enabled=AMP16)
amp = (lambda: torch.autocast("cuda", dtype=torch.float16)) if AMP16 else (lambda: torch.autocast("cuda", dtype=torch.bfloat16))
labels = torch.from_numpy(task_idx)
for step in range(1, a.steps + 1):
    lr = a.lr * min(1, step / 300) * 0.5 * (1 + math.cos(math.pi * step / a.steps))
    for g in opt.param_groups:
        g["lr"] = lr
    rows = np.random.choice(train_rows, a.batch)
    with amp():
        loss = F.cross_entropy(model(batch_x(rows)).float(), labels[rows].to(dev))
    opt.zero_grad(set_to_none=True)
    scaler.scale(loss).backward(); scaler.unscale_(opt)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(opt); scaler.update()
    if step % 500 == 0:
        print(json.dumps({"step": step, "loss": round(loss.item(), 4), "t": round(time.time() - t0)}), flush=True)

# ---------------- evaluation: within-suite accuracy ----------------
model.eval()
rng = np.random.RandomState(1)
ev = eval_rows if len(eval_rows) <= 40000 else rng.choice(eval_rows, 40000, replace=False)
correct = np.zeros(len(ev), bool)
with torch.no_grad():
    for s in range(0, len(ev), 512):
        r = ev[s:s + 512]
        with amp():
            lg = model(batch_x(r)).float()
        mask = torch.from_numpy(suite_id[None, :] != row_suite[r][:, None]).to(dev)
        lg = lg.masked_fill(mask, -1e9)
        correct[s:s + 512] = (lg.argmax(1).cpu().numpy() == task_idx[r])
bins = [(40, 80), (80, 160), (160, 320), (320, 10_000)]
res = {"cond": a.cond, "n_frames": n_frames, "n_eval": int(len(ev)), "n_eval_episodes": int(len(np.unique(ep[ev]))),
       "overall": float(correct.mean()), "per_suite": {}, "per_time_bin_s": {}}
for si, sname in enumerate(SUITES):
    msk = row_suite[ev] == si
    res["per_suite"][sname] = {"acc": float(correct[msk].mean()), "n": int(msk.sum())}
shared = np.isin(row_suite[ev], [0, 1, 2])
res["shared_scene_suites_acc"] = float(correct[shared].mean())
for lo, hi in bins:
    msk = shared & (fr[ev] >= lo) & (fr[ev] < hi)
    res["per_time_bin_s"][f"{lo / 20:.0f}-{min(hi, 9999) / 20:.0f}s"] = {"acc": float(correct[msk].mean()), "n": int(msk.sum())}
np.savez(f"{a.out}/{tag}_rows.npz", rows=ev, correct=correct, episode=ep[ev], frame=fr[ev], suite=row_suite[ev])
json.dump(res, open(f"{a.out}/{tag}.json", "w"), indent=2)
if not a.eval_only:
    torch.save({"model": model.state_dict(), "cond": a.cond, "n_frames": n_frames, "n_tasks": n_tasks, "task_names": names,
            "suite_id": suite_id.tolist(), "suites": SUITES}, f"{a.out}/{tag}_decoder.pt")
print(json.dumps(res), flush=True)
