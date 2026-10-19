"""MEM-3: memory when the scene is ambiguous. Task-intent decoders on LIBERO-Goal, trained on standard demos plus
counterfactual-scene recordings (G0.5 executing task t from the scene of task s; 33_record_cf.sh), so the scene layout
no longer identifies the task and only the robot's history can.

Train: standard LIBERO-Goal demos (non-held-out) + cf recordings with s = t+1, t+5, t+7.
Eval:  cf recordings with s = t+3 (the steering/memory probe pairs, never trained on) and held-out standard demos,
       at frames >= 2 s. Within-suite (10-way) accuracy.
Conditions (as MEM-1): current | first | offsets | snapshot.  Positive: memory conditions beat `current` on cf eval.
"""
import argparse
import glob
import json
import math
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

JOBS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, JOBS)
import t4_fp16  # noqa: E402,F401
import lawam_hybrid as H  # noqa: E402
from mem_decoder import Decoder  # noqa: E402

G = os.environ.get("G05_ROOT", "/export/home/kaziz/g05")
C = f"{G}/results/feat_cache"
DS = f"{G}/data/libero_merged_g05grip"
p = argparse.ArgumentParser()
p.add_argument("--conds", default="current,first,offsets,snapshot")
p.add_argument("--steps", type=int, default=5000)
p.add_argument("--batch", type=int, default=256)
p.add_argument("--lr", type=float, default=3e-4)
p.add_argument("--min_frame", type=int, default=40)
p.add_argument("--out", default=f"{G}/results/mem3")
a = p.parse_args()
os.makedirs(a.out, exist_ok=True)
dev = torch.device("cuda")
AMP16 = torch.cuda.get_device_capability()[0] < 8
amp = (lambda: torch.autocast("cuda", dtype=torch.float16)) if AMP16 else (lambda: torch.autocast("cuda", dtype=torch.bfloat16))
t0 = time.time()
TAU = json.load(open(f"{C}/snapshot_stats.json"))["tau"]

# ---------------- task names / LIBERO-Goal subset ----------------
d = pd.concat([pd.read_parquet(f, columns=["index", "task_index"]) for f in sorted(glob.glob(f"{DS}/data/*/*.parquet"))]).sort_values("index")
task_idx_std = d["task_index"].values.astype(np.int64)
tp = pd.read_parquet(f"{DS}/meta/tasks.parquet").reset_index()
tcol = [c for c in tp.columns if tp[c].dtype == object][0]
names = {int(r["task_index"]): r[tcol] for _, r in tp.iterrows()}
norm = lambda s: " ".join(str(s).lower().replace("_", " ").split())  # noqa: E731
from libero.libero import benchmark  # noqa: E402

bd = benchmark.get_benchmark_dict()
SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
lang2suite = {norm(t.language): s for s in SUITES for t in bd[s]().tasks}
n_tasks = max(names) + 1
suite_id = np.array([SUITES.index(lang2suite[norm(names[t])]) if norm(names[t]) in lang2suite else -1 for t in range(n_tasks)])
GOAL = SUITES.index("libero_goal")
name2idx = {norm(v): k for k, v in names.items()}

# ---------------- unified frame store: standard goal demos + cf recordings ----------------
m = np.load(f"{C}/meta.npz")
ep_std, fr_std = m["episode_index"], m["frame_index"]
goal_rows = np.nonzero(suite_id[task_idx_std] == GOAL)[0]
feats_std = np.load(f"{C}/feats.npy", mmap_mode="r")
F_list, ep_list, fr_list, lab_list, split_list = [], [], [], [], []
F_list.append(np.asarray(feats_std[goal_rows]))
ep_list.append(ep_std[goal_rows]); fr_list.append(fr_std[goal_rows]); lab_list.append(task_idx_std[goal_rows])
split_list.append(np.where(ep_std[goal_rows] % 5 == 0, "std_eval", "std_train"))
print(f"standard goal frames {len(goal_rows)} ({time.time() - t0:.0f}s)", flush=True)

lam = H.load_lawm()
cf_files = sorted(glob.glob(f"{G}/results/cf_record/t*_s*/rec/*.npz"))
next_ep = int(ep_std.max()) + 1
n_cf = {"cf_train": 0, "cf_eval": 0}
for f in cf_files:
    pair = os.path.basename(os.path.dirname(os.path.dirname(f)))  # t{t}_s{s}
    t, s = int(pair.split("_")[0][1:]), int(pair.split("_")[1][1:])
    split = "cf_eval" if (s - t) % 10 == 3 else "cf_train"
    z = np.load(f, allow_pickle=True)
    lab = name2idx.get(norm(str(z["task_description"])))
    if lab is None:
        continue
    imgs = z["image"]
    fe = []
    for k in range(0, len(imgs), 256):
        x = torch.from_numpy(imgs[k:k + 256]).permute(0, 3, 1, 2).cuda()
        g = H.dino_from_uint8(lam, x)
        b = g.shape[0]
        g = F.avg_pool2d(g.view(b, 16, 16, 768).permute(0, 3, 1, 2), 2).permute(0, 2, 3, 1).reshape(b, 64, 768)
        fe.append(g.half().cpu().numpy())
    fe = np.concatenate(fe)
    F_list.append(fe); ep_list.append(np.full(len(fe), next_ep)); fr_list.append(np.arange(len(fe)))
    lab_list.append(np.full(len(fe), lab)); split_list.append(np.full(len(fe), split)); next_ep += 1
    n_cf[split] += 1
del lam
torch.cuda.empty_cache()
FE = np.concatenate(F_list); EP = np.concatenate(ep_list); FR = np.concatenate(fr_list)
LAB = np.concatenate(lab_list); SPLIT = np.concatenate(split_list)
print(f"frames total {len(FE)}  cf episodes {n_cf}  ({time.time() - t0:.0f}s)", flush=True)

# snapshot memory rows (same rule as 27_snapshot_memory.py), episodes are contiguous
starts = np.flatnonzero(np.r_[True, EP[1:] != EP[:-1]]); ends = np.r_[starts[1:], len(EP)]
SNAP = np.zeros((len(EP), 4), dtype=np.int64)
for st, en in zip(starts, ends):
    Fe = FE[st:en].astype(np.float32)
    snaps, last = [0], Fe[0]
    for j in range(1, len(Fe)):
        if ((Fe[j] - last) ** 2).mean() > TAU:
            snaps.append(j); last = Fe[j]
    snaps = np.array(snaps)
    for tt in range(en - st):
        w = snaps[snaps < tt]
        if len(w) == 0:
            sel = [0, 0, 0, 0]
        else:
            sel = [int(w[0])] + [int(x) for x in w[-3:]]
            while len(sel) < 4:
                sel.insert(1, sel[1] if len(sel) > 1 else sel[0])
            sel = sorted(sel[:4])
        SNAP[st + tt] = st + np.array(sel)
print(f"snapshots done ({time.time() - t0:.0f}s)", flush=True)


def frames_for(cond, rows):
    if cond == "current":
        return rows[:, None]
    if cond == "first":
        return np.stack([rows - FR[rows], rows], 1)
    if cond == "offsets":
        return np.stack([rows - np.minimum(k, FR[rows]) for k in (160, 80, 40, 20)] + [rows], 1)
    return np.concatenate([SNAP[rows], rows[:, None]], 1)


elig = FR >= a.min_frame
train_rows = np.nonzero(elig & np.isin(SPLIT, ["std_train", "cf_train"]))[0]
w_train = np.where(SPLIT[train_rows] == "cf_train", 1.0, 1.0)
cf_tr = SPLIT[train_rows] == "cf_train"
if cf_tr.any():  # balance standard and counterfactual frames 50/50
    w_train = np.where(cf_tr, 0.5 / cf_tr.sum(), 0.5 / (~cf_tr).sum())
else:
    w_train = w_train / w_train.sum()
allowed = torch.from_numpy(suite_id == GOAL).to(dev)
results = {"n_cf_episodes": n_cf, "tau": TAU, "conds": {}}
for cond in a.conds.split(","):
    torch.manual_seed(0); np.random.seed(0)
    nf = frames_for(cond, np.array([starts[0] + 50])).shape[1]
    model = Decoder(nf, n_tasks).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.05)
    scaler = torch.cuda.amp.GradScaler(enabled=AMP16)
    for step in range(1, a.steps + 1):
        lr = a.lr * min(1, step / 300) * 0.5 * (1 + math.cos(math.pi * step / a.steps))
        for g_ in opt.param_groups:
            g_["lr"] = lr
        rows = np.random.choice(train_rows, a.batch, p=w_train)
        idx = frames_for(cond, rows)
        x = torch.from_numpy(FE[idx.reshape(-1)].reshape(len(rows), nf, 64, 768)).to(dev)
        with amp():
            lg = model(x).float().masked_fill(~allowed, -1e9)
            loss = F.cross_entropy(lg, torch.from_numpy(LAB[rows]).to(dev))
        opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); scaler.step(opt); scaler.update()
        if step % 1000 == 0:
            print(json.dumps({"cond": cond, "step": step, "loss": round(loss.item(), 4), "t": round(time.time() - t0)}), flush=True)
    model.eval()
    res = {}
    for split in ("cf_eval", "std_eval", "cf_train"):
        ev = np.nonzero(elig & (SPLIT == split))[0]
        if len(ev) == 0:
            continue
        corr = np.zeros(len(ev), bool)
        with torch.no_grad():
            for k in range(0, len(ev), 512):
                r = ev[k:k + 512]; idx = frames_for(cond, r)
                x = torch.from_numpy(FE[idx.reshape(-1)].reshape(len(r), nf, 64, 768)).to(dev)
                with amp():
                    lg = model(x).float().masked_fill(~allowed, -1e9)
                corr[k:k + 512] = lg.argmax(1).cpu().numpy() == LAB[r]
        bins = {}
        for lo, hi in ((40, 80), (80, 160), (160, 10_000)):
            mk = (FR[ev] >= lo) & (FR[ev] < hi)
            bins[f"{lo / 20:.0f}-{min(hi, 9999) / 20:.0f}s"] = float(corr[mk].mean()) if mk.any() else None
        # episode-level: accuracy at exactly t = 2 s (the moment the instruction is withdrawn in the probes)
        at2 = ev[FR[ev] == a.min_frame]
        res[split] = {"acc": float(corr.mean()), "n_frames": int(len(ev)), "n_episodes": int(len(np.unique(EP[ev]))),
                      "acc_at_2s": float(corr[np.isin(ev, at2)].mean()) if len(at2) else None, "by_time": bins}
    results["conds"][cond] = res
    print(json.dumps({"cond": cond, **res}), flush=True)
    torch.save({"model": model.state_dict(), "cond": cond, "n_frames": nf, "n_tasks": n_tasks, "task_names": names,
                "suite_id": suite_id.tolist(), "suites": SUITES}, f"{a.out}/{cond}_decoder.pt")
json.dump(results, open(f"{a.out}/mem3_results.json", "w"), indent=2)
print("saved", f"{a.out}/mem3_results.json")
