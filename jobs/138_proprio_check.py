"""Offline check of the activity-log counter (lmem_proprio.py) on LIBERO-Mem's repetition tasks 0-5: replay every demo's
simulator states (no cameras) to get exactly the finger opening and hand height the robot observes in closed loop
(robot0_gripper_qpos, robot0_eef_pos), and compare the count with (a) the task's N at the end of the demo and (b) the
object-motion cycle labels of 54_lmem_relabel.py frame by frame (5 Hz rows of results/lmem_feats), on the held-out demos
100-119 that the closed-loop evaluation uses and on the training demos. Needs a GPU node (robosuite initialises EGL).
"""
import glob
import json
import os
import sys

import h5py
import numpy as np

G = "/export/home/kaziz/g05"
sys.path.insert(0, f"{G}/jobs")
from lmem_proprio import ProprioCounter  # noqa: E402

from libero.libero import benchmark, get_libero_path  # noqa: E402
from libero.libero.envs.env_wrapper import ControlEnv  # noqa: E402

NREP = {0: 1, 1: 1, 2: 3, 3: 3, 4: 5, 5: 7}
suite = benchmark.get_benchmark_dict()["libero_mem"]()
m = np.load(f"{G}/results/lmem_feats/meta.npz"); lab = np.load(f"{G}/results/lmem_feats/labels_cycles.npy")
out = {}
for tid in range(6):
    task = suite.get_task(tid)
    env = ControlEnv(bddl_file_name=os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file),
                     has_renderer=False, has_offscreen_renderer=False, use_camera_obs=False)
    env.reset()
    with h5py.File(glob.glob(f"{G}/libero_mem/raw/{task.name}_demo.hdf5")[0], "r") as f:
        keys = sorted(f["data"].keys(), key=lambda s: int(s.split("_")[-1]))
        fin, agree, n_fr = {"train": [], "heldout": []}, [], 0
        for di, k in enumerate(keys):
            states = f["data"][k]["states"][()]
            c = ProprioCounter(); counts = []
            for s in states:
                ob = env.set_init_state(s)
                counts.append(c.update(ob["robot0_gripper_qpos"], ob["robot0_eef_pos"][2]))
            counts = np.array(counts)
            fin["heldout" if di >= 100 else "train"].append(int(counts[-1]))
            if di >= 100:
                rows = np.nonzero((m["task"] == tid) & (m["demo"] == di))[0]
                y = lab[rows]; p = counts[np.minimum(m["frame"][rows] * 4, len(counts) - 1)]
                ok = y >= 0; agree.append((p[ok] == y[ok]).sum()); n_fr += int(ok.sum())
        out[tid] = {"N": NREP[tid],
                    "final_count_eq_N_heldout": float(np.mean(np.array(fin["heldout"]) == NREP[tid])),
                    "final_count_eq_N_train": float(np.mean(np.array(fin["train"]) == NREP[tid])),
                    "final_hist_heldout": {int(v): int((np.array(fin["heldout"]) == v).sum()) for v in np.unique(fin["heldout"])},
                    "frame_agreement_with_geometric_labels_heldout": float(sum(agree) / max(n_fr, 1))}
        print(tid, json.dumps(out[tid]), flush=True)
    env.close()
json.dump(out, open(f"{G}/results/lmem_feats/proprio_counter_check.json", "w"), indent=1)
print("PROPRIO_CHECK_DONE", flush=True)
