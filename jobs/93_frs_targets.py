"""Coarse 'pointing' targets for Flow Reversal Steering references (95_frs_client.py), from the LIBERO-Goal demos.

Per LIBERO-Goal task: the mean end-effector position where the gripper first closes (the contact point a human would
point at; tasks without a close event use the position at 40% of the episode), the typical approach speed in action
units, the action -> displacement scale (least squares of the per-step position change on the position action), and the
dataset's 'open' gripper value (read from the first frames of every demo, where the gripper is open).
State and actions are in the same world-frame convention the live client sends (build_libero_raw_obs).
Output: results/frs/targets.json
"""
import glob
import json
import os

import numpy as np
import pandas as pd

G = "/export/home/kaziz/g05"; DS = f"{G}/data/libero_merged_g05grip"
GOAL = ["open the middle drawer of the cabinet", "put the bowl on the stove", "put the wine bottle on top of the cabinet",
        "open the top drawer and put the bowl inside", "put the bowl on top of the cabinet", "push the plate to the front of the stove",
        "put the cream cheese in the bowl", "turn on the stove", "put the bowl on the plate", "put the wine bottle on the rack"]
nrm = lambda s: " ".join(str(s).lower().replace("_", " ").split())  # noqa: E731
df = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"{DS}/data/*/*.parquet"))]).sort_values("index")
tp = pd.read_parquet(f"{DS}/meta/tasks.parquet").reset_index()
tcol = [c for c in tp.columns if tp[c].dtype == object][0]
name2idx = {nrm(r[tcol]): int(r["task_index"]) for _, r in tp.iterrows()}
S = np.stack(df["observation.state"].values).astype(np.float64); A = np.stack(df["action"].values).astype(np.float64)
ep = df["episode_index"].values; ti = df["task_index"].values

# gripper 'open' value: the first 5 frames of every demo (all demos start open)
first = np.concatenate([np.nonzero(ep == e)[0][:5] for e in np.unique(ep)])
g_open = float(np.median(A[first, 6])); g_vals = np.unique(np.round(A[:, 6], 3))
print("gripper action values:", g_vals[:10], "open =", g_open)

# action -> displacement scale (per axis), from consecutive frames of the same episode
same = ep[1:] == ep[:-1]
dpos, apos = (S[1:, :3] - S[:-1, :3])[same], A[:-1, :3][same]
k = np.array([float(np.dot(apos[:, j], dpos[:, j]) / max(np.dot(apos[:, j], apos[:, j]), 1e-9)) for j in range(3)])
print("metres per action unit per step:", k.round(5))

out = {"gripper_open": g_open, "k": k.tolist(), "tasks": {}}
for name in GOAL:
    t = name2idx[nrm(name)]; pts, speeds = [], []
    for e in np.unique(ep[ti == t]):
        rows = np.nonzero(ep == e)[0]; a, s = A[rows], S[rows]
        closed = np.nonzero(np.abs(a[:, 6] - g_open) > 0.5 * (np.ptp(g_vals) or 1.0))[0]
        c = int(closed[0]) if len(closed) else int(0.4 * len(rows))
        pts.append(s[c, :3]); speeds.append(np.median(np.linalg.norm(a[: max(c, 2), :3], axis=1)))
    pts = np.array(pts)
    out["tasks"][name] = {"point": pts.mean(0).tolist(), "point_std_cm": (pts.std(0) * 100).round(2).tolist(),
                          "speed": float(np.median(speeds)), "n_demos": int(len(pts))}
    print(f"{name:45s} point {pts.mean(0).round(3)} (std cm {(pts.std(0) * 100).round(1)}) speed {np.median(speeds):.3f} n={len(pts)}")
os.makedirs(f"{G}/results/frs", exist_ok=True)
json.dump(out, open(f"{G}/results/frs/targets.json", "w"), indent=1)
