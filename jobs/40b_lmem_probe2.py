"""LIBERO-Mem: validate the subgoal checker. A = action replay (their GT protocol), B = state replay (exact)."""
import glob, json, os, time
import h5py, numpy as np
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

RAW = "/export/home/kaziz/g05/libero_mem/raw"
suite = benchmark.get_benchmark_dict()["libero_mem"]()
DUMMY = np.array([0, 0, 0, 0, 0, 0, -1.0])
t0 = time.time()
for tid in [0, 2, 6]:
    task = suite.get_task(tid)
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=128, camera_widths=128)
    env.seed(0)
    P = env.env
    fn = glob.glob(f"{RAW}/{task.name}_demo.hdf5")[0]
    with h5py.File(fn, "r") as f:
        keys = sorted(f["data"].keys(), key=lambda s: int(s.split("_")[-1]))[:3]
        demos = [(f["data"][k]["init_state"][()], f["data"][k]["states"][()], f["data"][k]["actions"][()]) for k in keys]
    goal_len = len(P.parsed_problem["goal_state"][0]) - 1
    for di, (init, states, acts) in enumerate(demos):
        # B: exact state replay
        env.reset(); P.reset_subgoal_progress()
        doneB, nB = False, []
        for t, s in enumerate(states):
            P.sim.set_state_from_flattened(s); P.sim.forward()
            P._check_success(inc=True)
            doneB = P._check_success()
            if t % 40 == 0: nB.append(len(P._satisfied_subgoals))
        nB.append(len(P._satisfied_subgoals))
        # A: action replay with 20 settle steps
        env.reset(); env.set_init_state(init); P.reset_subgoal_progress()
        for _ in range(20): env.step(DUMMY)
        a0 = 0
        while a0 < len(acts) and np.linalg.norm(acts[a0][:-1]) < 1e-4: a0 += 1
        doneA, firstA = False, None
        for t, a in enumerate(acts[a0:]):
            a = a.copy(); a[-1] = a[-1] if a[-1] != 0 else -1
            P._check_success(inc=True)
            obs, r, done, info = env.step(a)
            if done and firstA is None: firstA = t
        print(json.dumps({"task": tid, "demo": keys[di], "goal_len": goal_len, "B_state_replay_nsat_trace": nB, "B_done": bool(doneB),
                          "A_action_replay_nsat": len(P._satisfied_subgoals), "A_first_done": firstA, "len": len(acts), "t": round(time.time() - t0)}), flush=True)
    env.close()
