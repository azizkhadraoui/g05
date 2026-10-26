"""LIBERO-Mem offline memory test, step 2: how many pick-and-place cycles have been completed? (from 45_lmem_feats.py)

Train on demos 0-99 of every task, evaluate on held-out demos 100-119 (the ones used for closed-loop evaluation).
Conditions (frames given to the decoder; 5 Hz):
  current   current frame only (Markovian)
  first     first frame + current
  offsets   1, 2, 4, 8 s back + current
  uniform8  8 frames evenly spaced over the whole episode so far + current (full-episode episodic memory)
  snapshot  3D-Mem-style: first + 3 most recent event snapshots + current
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import t4_fp16  # noqa: E402,F401
from mem_decoder import Decoder  # noqa: E402

G = "/export/home/kaziz/g05"
C = f"{G}/results/lmem_feats"
p = argparse.ArgumentParser()
p.add_argument("--cond", required=True, choices=["current", "first", "offsets", "uniform8", "snapshot"])
p.add_argument("--steps", type=int, default=4000)
p.add_argument("--batch", type=int, default=192)
p.add_argument("--lr", type=float, default=3e-4)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--tau", type=float, default=0.15)
p.add_argument("--labels", default=None, help="alternative per-row labels (.npy, -1 = unused), e.g. labels_cycles.npy")
p.add_argument("--tasks", default=None, help="comma list of task ids to train/evaluate on")
p.add_argument("--tag", default=None)
p.add_argument("--n_classes", type=int, default=9, help="output classes (9 = counts 0-8; labels_instr.npy: 22 step instructions)")
p.add_argument("--classes", default=None, help="json list of class names saved with the decoder (e.g. labels_instr_classes.json)")
a = p.parse_args()
torch.manual_seed(a.seed); np.random.seed(a.seed)
dev = torch.device("cuda")
AMP16 = torch.cuda.get_device_capability()[0] < 8
amp = (lambda: torch.autocast("cuda", dtype=torch.float16)) if AMP16 else (lambda: torch.autocast("cuda", dtype=torch.bfloat16))
t0 = time.time()
m = np.load(f"{C}/meta.npz")
task, demo, fr, count = m["task"], m["demo"], m["frame"], np.minimum(m["count"], a.n_classes - 1)
if a.labels:
    count = np.minimum(np.load(a.labels), a.n_classes - 1)
use = np.ones(len(task), bool) if a.tasks is None else np.isin(task, [int(x) for x in a.tasks.split(",")])
use &= count >= 0
feats = np.ascontiguousarray(np.load(f"{C}/feats.npy", mmap_mode="r"))
N = len(task)
start = np.arange(N) - fr  # episodes are stored contiguously
print(f"loaded {feats.shape} ({time.time() - t0:.0f}s)", flush=True)

snap = None
if a.cond == "snapshot":  # event-based writes: new snapshot when features move > tau from the last snapshot
    snap = np.zeros((N, 4), np.int64)
    ep_id = task * 1000 + demo
    bounds = np.flatnonzero(np.r_[True, ep_id[1:] != ep_id[:-1], True])
    n_snaps = []
    for s, e in zip(bounds[:-1], bounds[1:]):
        fe = feats[s:e].astype(np.float32)
        snaps, last = [0], fe[0]
        for j in range(1, e - s):
            if ((fe[j] - last) ** 2).mean() > a.tau:
                snaps.append(j); last = fe[j]
        n_snaps.append(len(snaps)); snaps = np.array(snaps)
        for t in range(e - s):
            w = snaps[snaps < t]
            sel = [0, 0, 0, 0] if len(w) == 0 else sorted(([int(w[0])] + [int(x) for x in w[-3:]] + [int(w[-1])] * 3)[:4])
            snap[s + t] = s + np.array(sel)
    print(f"snapshots per episode: mean {np.mean(n_snaps):.1f}", flush=True)


def frames_for(rows):
    if a.cond == "current":
        return rows[:, None]
    if a.cond == "first":
        return np.stack([start[rows], rows], 1)
    if a.cond == "offsets":
        return np.stack([rows - np.minimum(k, fr[rows]) for k in (40, 20, 10, 5)] + [rows], 1)
    if a.cond == "uniform8":
        return np.stack([start[rows] + (fr[rows] * j) // 8 for j in range(8)] + [rows], 1)
    return np.concatenate([snap[rows], rows[:, None]], 1)


nf = frames_for(np.array([100])).shape[1]
train_rows = np.nonzero((demo < 100) & use)[0]
eval_rows = np.nonzero((demo >= 100) & use)[0]
model = Decoder(nf, a.n_classes).to(dev)
opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.05)
scaler = torch.cuda.amp.GradScaler(enabled=AMP16)
lab = torch.from_numpy(count)


def batch_x(rows):
    idx = frames_for(rows)
    return torch.from_numpy(feats[idx.reshape(-1)].reshape(len(rows), nf, 64, 768)).to(dev)


for step in range(1, a.steps + 1):
    lr = a.lr * min(1, step / 300) * 0.5 * (1 + math.cos(math.pi * step / a.steps))
    for g in opt.param_groups:
        g["lr"] = lr
    rows = np.random.choice(train_rows, a.batch)
    with amp():
        loss = F.cross_entropy(model(batch_x(rows)).float(), lab[rows].to(dev))
    opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.unscale_(opt)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); scaler.step(opt); scaler.update()
    if step % 500 == 0:
        print(json.dumps({"step": step, "loss": round(loss.item(), 4), "t": round(time.time() - t0)}), flush=True)

model.eval()
pred = np.zeros(len(eval_rows), np.int64)
with torch.no_grad():
    for s in range(0, len(eval_rows), 384):
        r = eval_rows[s:s + 384]
        with amp():
            pred[s:s + 384] = model(batch_x(r)).float().argmax(1).cpu().numpy()
y = count[eval_rows]; ok = pred == y; tk = task[eval_rows]
res = {"cond": a.cond, "n_frames_in": nf, "n_eval_frames": int(len(eval_rows)), "acc_all": float(ok.mean()),
       "acc_after_first_cycle": float(ok[y >= 1].mean()), "per_task": {}, "per_count": {}}
for t in sorted(set(tk.tolist())):
    mk = tk == t
    maj = np.bincount(y[mk]).max() / mk.sum()
    res["per_task"][int(t)] = {"acc": float(ok[mk].mean()), "acc_after_first_cycle": float(ok[mk & (y >= 1)].mean()) if (mk & (y >= 1)).any() else None,
                               "majority_baseline": float(maj), "max_count": int(y[mk].max())}
for c in range(a.n_classes):
    mk = y == c
    if mk.any():
        res["per_count"][int(c)] = {"acc": float(ok[mk].mean()), "n": int(mk.sum())}
os.makedirs(f"{G}/results/lmem_count", exist_ok=True)
tag = a.tag or (a.cond if a.seed == 0 else f"{a.cond}_seed{a.seed}")
json.dump(res, open(f"{G}/results/lmem_count/{tag}.json", "w"), indent=2)
np.savez(f"{G}/results/lmem_count/{tag}_rows.npz", rows=eval_rows, pred=pred, y=y, task=tk, demo=demo[eval_rows], frame=fr[eval_rows])
torch.save({"model": model.state_dict(), "cond": a.cond, "n_frames": nf, "tau": a.tau, "n_classes": a.n_classes,
            "classes": json.load(open(a.classes)) if a.classes else None}, f"{G}/results/lmem_count/{tag}_decoder.pt")
print(json.dumps(res), flush=True)
