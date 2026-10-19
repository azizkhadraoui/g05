"""LIBERO-Mem checker ceiling: replay the held-out demos (100-119) with the benchmark's GT protocol
(set init state, reset subgoal progress, 20 settle steps, strip leading no-ops, _check_success(inc=True) each step)."""
import glob, json, os, re, time
import h5py, numpy as np
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

RAW = "/export/home/kaziz/g05/libero_mem/raw"
suite = benchmark.get_benchmark_dict()["libero_mem"]()
DUMMY = np.array([0, 0, 0, 0, 0, 0, -1.0])
out = {}
for tid in range(suite.n_tasks):
    task = suite.get_task(tid)
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=128, camera_widths=128); env.seed(0); P = env.env
    gl = P.get_goal_sequence_len() if hasattr(P, "get_goal_sequence_len") else len(P.parsed_problem["goal_state"][0]) - 1
    with h5py.File(glob.glob(f"{RAW}/{task.name}_demo.hdf5")[0], "r") as f:
        keys = sorted(f["data"].keys(), key=lambda s: int(s.split("_")[-1]))[100:120]
        demos = [(f["data"][k]["init_state"][()], f["data"][k]["actions"][()]) for k in keys]
    succ, subs = [], []
    for init, acts in demos:
        env.reset(); env.set_init_state(init); P.reset_subgoal_progress()
        if hasattr(P, "_overshot"): P._overshot = False
        for _ in range(20): env.step(DUMMY)
        a0 = 0
        while a0 < len(acts) and np.linalg.norm(acts[a0][:-1]) < 1e-4: a0 += 1
        done_any, mx = False, 0
        for a in acts[a0:]:
            a = a.copy(); a[-1] = a[-1] if a[-1] != 0 else -1
            P._check_success(inc=True); _, _, done, _ = env.step(a)
            done_any |= bool(done); mx = max(mx, len(P._satisfied_subgoals))
        succ.append(done_any); subs.append(min(mx, gl) / gl)
    out[tid] = {"task": re.sub(r"^\d+\s+", "", task.language), "goal_len": gl, "demo_success": float(np.mean(succ)), "demo_subgoal_completion": float(np.mean(subs))}
    print(json.dumps({"tid": tid, **out[tid]}), flush=True)
    env.close()
json.dump(out, open("/export/home/kaziz/g05/results/lmem_ceiling.json", "w"), indent=2)
