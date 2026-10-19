"""SM1 (3D-Mem-inspired snapshot memory): event-based snapshot writing + retrieval table for every frame.

Write: per episode, a new snapshot is stored when the pooled DINOv3 features differ from the last stored snapshot by
more than tau (mean squared difference over the 8x8 tokens). Frame 0 is always stored. tau is calibrated so that
episodes hold ~TARGET snapshots on average (3D-Mem reports ~11 per scene on A-EQA).
Read (at frame t): the first snapshot + the 3 most recent snapshots written strictly before t (padded by repetition).
Output: results/feat_cache/snapshot_mem_rows.npy [N, 4] (row indices, time-ordered) and snapshot_stats.json.
"""
import json
import os

import numpy as np

C = "/export/home/kaziz/g05/results/feat_cache"
TARGET = float(os.environ.get("TARGET_SNAPSHOTS", 10))
m = np.load(f"{C}/meta.npz")
ep, fr = m["episode_index"], m["frame_index"]
feats = np.load(f"{C}/feats.npy", mmap_mode="r")
N = len(ep)
starts = np.flatnonzero(np.r_[True, ep[1:] != ep[:-1]])
ends = np.r_[starts[1:], N]


def snaps_for(F, tau):
    out, last = [0], F[0]
    for j in range(1, len(F)):
        if ((F[j] - last) ** 2).mean() > tau:
            out.append(j)
            last = F[j]
    return out


rng = np.random.RandomState(0)
calib = rng.choice(len(starts), 120, replace=False)
cal_F = [np.asarray(feats[starts[i]:ends[i]], dtype=np.float32) for i in calib]
lo, hi = 1e-4, 10.0
for _ in range(30):  # geometric bisection on tau
    tau = (lo * hi) ** 0.5
    mean_n = np.mean([len(snaps_for(F, tau)) for F in cal_F])
    lo, hi = (tau, hi) if mean_n > TARGET else (lo, tau)
tau = (lo * hi) ** 0.5
print(f"tau={tau:.5f} -> {np.mean([len(snaps_for(F, tau)) for F in cal_F]):.2f} snapshots/episode (calibration)", flush=True)

mem_rows = np.zeros((N, 4), dtype=np.int64)
n_snaps, ages = [], []
for k, (s, e) in enumerate(zip(starts, ends)):
    F = np.asarray(feats[s:e], dtype=np.float32)
    snaps = snaps_for(F, tau)
    n_snaps.append(len(snaps))
    snaps = np.array(snaps)
    for t in range(e - s):
        written = snaps[snaps < t]
        if len(written) == 0:
            sel = [0, 0, 0, 0]
        else:
            recent = list(written[-3:])
            sel = [int(written[0])] + [int(x) for x in recent]
            while len(sel) < 4:
                sel.insert(1, sel[1] if len(sel) > 1 else sel[0])
            sel = sorted(sel[:4])
        mem_rows[s + t] = s + np.array(sel)
        ages.append(t - sel[-1])
    if (k + 1) % 200 == 0:
        print(f"  {k + 1}/{len(starts)} episodes", flush=True)
np.save(f"{C}/snapshot_mem_rows.npy", mem_rows)
stats = {"tau": tau, "target": TARGET, "snapshots_per_episode_mean": float(np.mean(n_snaps)),
         "snapshots_per_episode_median": float(np.median(n_snaps)), "mean_age_of_latest_snapshot_frames": float(np.mean(ages)),
         "fixed_offset_baseline_frames": [20, 40, 80, 160]}
json.dump(stats, open(f"{C}/snapshot_stats.json", "w"), indent=2)
print(json.dumps(stats))
