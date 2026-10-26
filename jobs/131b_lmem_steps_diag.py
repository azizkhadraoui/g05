"""Diagnostic for 131_lmem_steps.py's move detection: object geometry over a few demos of LIBERO-Mem tasks 6-9 (no rendering).
Bowls: at every rest period (low and still for >= 3 steps), the nearest plate and its horizontal distance.
Cheese / baskets: final cheese-basket offsets, basket displacement, height change and still steps at the end of the demo.
"""
import glob
import os
import sys

import numpy as np

G = "/export/home/kaziz/g05"
sys.path.insert(0, f"{G}/jobs")
import lmem_steps as S  # noqa: E402
from libero.libero import benchmark, get_libero_path  # noqa: E402
from libero.libero.envs.env_wrapper import ControlEnv  # noqa: E402
import h5py  # noqa: E402

suite = benchmark.get_benchmark_dict()["libero_mem"]()
for tid in S.TASKS:
    task = suite.get_task(tid)
    env = ControlEnv(bddl_file_name=os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file),
                     has_renderer=False, has_offscreen_renderer=False, use_camera_obs=False)
    env.reset(); P = env.env; names = S.objects(tid)
    print(f"== task {tid}: {task.language}", flush=True)
    with h5py.File(glob.glob(f"{G}/libero_mem/raw/{task.name}_demo.hdf5")[0], "r") as f:
        keys = sorted(f["data"].keys(), key=lambda s: int(s.split("_")[-1]))
        for k in keys[:5]:
            states = f["data"][k]["states"][()]
            X = []
            for s in states:
                P.sim.set_state_from_flattened(s); X.append({n: P.sim.data.get_joint_qpos(n + "_joint0")[:3].copy() for n in names})
            T = len(X)
            if tid in (6, 7):
                plates = [n for n in names if n.startswith("plate")]; pxy = np.stack([X[0][p][:2] for p in plates])
                out = []
                for b in [n for n in names if "bowl" in n]:
                    z = np.array([x[b][2] for x in X]); xy = np.stack([x[b][:2] for x in X]); v = np.r_[0, np.linalg.norm(np.diff(xy, axis=0), axis=1)]
                    rest = (z < z[0] + 0.02) & (v < 0.002); segs = []; t = 0
                    while t < T:
                        if rest[t]:
                            u = t
                            while u < T and rest[u]:
                                u += 1
                            if u - t >= 2:
                                d = np.linalg.norm(pxy - xy[(t + u) // 2], axis=1); j = int(d.argmin())
                                segs.append((t, u - t, plates[j], round(float(d[j]) * 100, 1)))
                            t = u
                        else:
                            t += 1
                    out.append(f"{b}: z0 {z[0]:.3f} zmax {z.max():.3f} rests(t,len,plate,cm)={segs[:6]}")
                print(k, T, "\n   " + "\n   ".join(out), flush=True)
            else:
                ch = X[-1]["cream_cheese_1"]
                info = []
                for b in ("basket_1", "basket_2"):
                    p0, p1 = X[0][b], X[-1][b]
                    mv = [t for t in range(1, T) if np.linalg.norm(X[t][b] - X[t - 1][b]) > 0.001]
                    info.append(f"{b}: y0 {p0[1]:.3f} disp {np.linalg.norm(p1[:2] - p0[:2]) * 100:.1f}cm dz {(p1[2] - p0[2]) * 100:.1f}cm "
                                f"last_move_t {mv[-1] if mv else None} cheese_xy {np.linalg.norm(ch[:2] - p1[:2]) * 100:.1f}cm cheese_dz {(ch[2] - p1[2]) * 100:.1f}cm")
                print(k, T, "\n   " + "\n   ".join(info), flush=True)
    env.close()
