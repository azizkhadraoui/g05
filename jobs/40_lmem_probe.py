"""LIBERO-Mem setup check (run with the libero-mem fork first on PYTHONPATH and LIBERO_CONFIG_PATH=libero_mem/cfg).

1. Print the raw demo hdf5 layout.
2. Write init-state files (first state of every demo) to libero_mem/init_files/libero_mem/<task>.pruned_init, so the
   stock benchmark API (get_task_init_states) works. Demos 0-99 = training split, 100-119 = held-out ("unseen").
3. Replay demo 0 of a few tasks with LIBERO-Mem's subgoal-aware success check (_check_success(inc=True) every step).
"""
import glob
import json
import os
import sys
import time

import h5py
import numpy as np
import torch

LM = "/export/home/kaziz/g05/libero_mem"
RAW = f"{LM}/raw"
INIT = f"{LM}/init_files/libero_mem"
os.makedirs(INIT, exist_ok=True)
t0 = time.time()

files = sorted(glob.glob(f"{RAW}/*_demo.hdf5"))
with h5py.File(files[0], "r") as f:
    print("top keys", list(f.keys()), "attrs", list(f.attrs.keys()))
    data = f["data"]
    print("data attrs", {k: (str(v)[:200]) for k, v in data.attrs.items()})
    demos = list(data.keys())
    print("n demos", len(demos), demos[:3])
    d0 = data[demos[0]]
    for k in d0.keys():
        v = d0[k]
        print("  ", k, getattr(v, "shape", None), getattr(v, "dtype", None) if hasattr(v, "dtype") else list(v.keys())[:10])
    print("demo attrs", {k: str(v)[:120] for k, v in d0.attrs.items()})
    a = d0["actions"][()]
    print("gripper values", np.unique(a[:, -1])[:10], "len", len(a))

meta = {}
for fn in files:
    name = os.path.basename(fn).replace("_demo.hdf5", "")
    with h5py.File(fn, "r") as f:
        data = f["data"]
        keys = sorted(data.keys(), key=lambda s: int(s.split("_")[-1]))
        inits = [data[k]["init_state"][()] if "init_state" in data[k] else data[k]["states"][0] for k in keys]
        lens = [len(data[k]["actions"]) for k in keys]
    torch.save(np.stack(inits), f"{INIT}/{name}.pruned_init")
    meta[name] = {"n_demos": len(keys), "len_mean": float(np.mean(lens)), "len_max": int(np.max(lens))}
    print(name, meta[name], flush=True)
json.dump(meta, open(f"{LM}/demo_meta.json", "w"), indent=2)

from libero.libero import benchmark, get_libero_path  # noqa: E402
from libero.libero.envs import OffScreenRenderEnv  # noqa: E402
import libero  # noqa: E402

print("libero package:", libero.__file__)
suite = benchmark.get_benchmark_dict()["libero_mem"]()
print("tasks:", [(i, t.name, t.language) for i, t in enumerate(suite.tasks)])
for tid in [0, 2, 6]:
    task = suite.get_task(tid)
    init = suite.get_task_init_states(tid)
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256)
    env.seed(0); env.reset()
    obs = env.set_init_state(init[0])
    fn = [x for x in files if os.path.basename(x).startswith(task.name)][0]
    with h5py.File(fn, "r") as f:
        k = sorted(f["data"].keys(), key=lambda s: int(s.split("_")[-1]))[0]
        acts = f["data"][k]["actions"][()]
    first_done, n_sub = None, []
    for i, act in enumerate(acts):
        act = act.copy()
        act[-1] = act[-1] if act[-1] != 0 else -1
        env.env._check_success(inc=True)
        obs, r, done, info = env.step(act)
        if done and first_done is None:
            first_done = i
        if i % 50 == 0:
            n_sub.append(len(env.env.get_satisfied_subgoals(task.language)))
    print(json.dumps({"task": tid, "lang": task.language, "demo_len": len(acts), "first_done": first_done,
                      "subgoals_satisfied_every_50": n_sub, "final_sat": str(env.env.get_satisfied_subgoals(task.language))[:300],
                      "img": list(obs["agentview_image"].shape), "t": round(time.time() - t0)}), flush=True)
    env.close()
