"""Cycle labels from object geometry (replaces gripper-release counting, which teleop re-grasps make noisy).
For the repetition tasks (0-5): object z from the demo's sim states (no rendering). A cycle completes when the object,
after being lifted (> 4 cm above its start for >= 5 steps), comes back down (< 3.5 cm for >= 3 steps).
Labels are aligned with results/lmem_feats/meta.npz rows (5 Hz: frame f <-> step 4f); other tasks get -1.
"""
import glob, json, os
import h5py, numpy as np
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

G = "/export/home/kaziz/g05"; RAW = f"{G}/libero_mem/raw"
OBJ = {0: "akita_black_bowl_1", 1: "wine_bottle_1", 2: "akita_black_bowl_1", 3: "wine_bottle_1", 4: "akita_black_bowl_1", 5: "akita_black_bowl_1"}
NREP = {0: 1, 1: 1, 2: 3, 3: 3, 4: 5, 5: 7}
m = np.load(f"{G}/results/lmem_feats/meta.npz")
labels = np.full(len(m["task"]), -1, np.int64)
suite = benchmark.get_benchmark_dict()["libero_mem"]()
report = {}
for tid in range(6):
    task = suite.get_task(tid)
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=64, camera_widths=64); env.seed(0); env.reset(); P = env.env
    finals = []
    with h5py.File(glob.glob(f"{RAW}/{task.name}_demo.hdf5")[0], "r") as f:
        keys = sorted(f["data"].keys(), key=lambda s: int(s.split("_")[-1]))
        for di, k in enumerate(keys):
            states = f["data"][k]["states"][()]
            z = np.empty(len(states))
            for t, s in enumerate(states):
                P.sim.set_state_from_flattened(s)
                z[t] = P.sim.data.get_joint_qpos(OBJ[tid] + "_joint0")[2]
            z0 = np.median(z[:10]); cyc = np.zeros(len(z), np.int64); c = 0; up = still = 0; lifted = False
            for t in range(len(z)):
                up = up + 1 if z[t] > z0 + 0.04 else 0
                if up >= 5:
                    lifted = True
                still = still + 1 if z[t] < z0 + 0.035 else 0  # back down (demos end right after the last placement)
                if lifted and still >= 3:
                    c += 1; lifted = False
                cyc[t] = c
            finals.append(int(c))
            rows = np.nonzero((m["task"] == tid) & (m["demo"] == di))[0]
            labels[rows] = cyc[np.minimum(m["frame"][rows] * 4, len(cyc) - 1)]
    env.close()
    fin = np.array(finals)
    report[tid] = {"N": NREP[tid], "final_count_hist": {int(v): int((fin == v).sum()) for v in np.unique(fin)}, "frac_final_eq_N": float((fin == NREP[tid]).mean())}
    print(tid, report[tid], flush=True)
np.save(f"{G}/results/lmem_feats/labels_cycles.npy", labels)
json.dump(report, open(f"{G}/results/lmem_feats/labels_cycles_report.json", "w"), indent=2)
