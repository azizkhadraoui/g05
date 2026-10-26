"""Long-horizon probe: chains of K LIBERO-Goal commands executed back-to-back in one episode (CALVIN-style).

All 10 LIBERO-Goal tasks share one scene and one set of objects, so every task's goal (its BDDL predicates) can be checked
in any episode. A chain starts from the first task's initial state; when the active command's goal holds for 5
consecutive steps, the next command is given. The episode stops at the first command not completed within its step
budget. Metric: number of commands completed in a row (0..K); also success rate at each chain position.

Works with any policy behind the G0.5 websocket protocol (G0.5 server or the cross-model adapters).
"""
import argparse
import asyncio
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np

G = os.environ.get("G05_ROOT", "/export/home/kaziz/g05")
sys.path.insert(0, f"{G}/GalaxeaVLA"); sys.path.insert(0, f"{G}/jobs")
os.chdir(f"{G}/GalaxeaVLA")
from experiments.libero.libero_eval_utils import (  # noqa: E402
    LIBERO_DUMMY_ACTION, LiberoGripperCommandState, build_libero_raw_obs, ensure_libero_config, single_action_to_libero_action)
from experiments.libero.libero_vector_env import LiberoAsyncVectorEnv  # noqa: E402
from scripts.utils.policy_ws_client import MultiPolicyWSClient  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--server_uri", required=True)
p.add_argument("--n_chains", type=int, default=50)
p.add_argument("--chain_len", type=int, default=5)
p.add_argument("--num_parallel", type=int, default=10)
p.add_argument("--task_budget", type=int, default=300, help="max steps per command")
p.add_argument("--num_steps_wait", type=int, default=20)
p.add_argument("--hold", type=int, default=5, help="goal must hold this many consecutive steps")
p.add_argument("--seed", type=int, default=7)
p.add_argument("--chains_seed", type=int, default=0)
p.add_argument("--output_dir", required=True)
a = p.parse_args()
os.makedirs(a.output_dir, exist_ok=True)


def make_env_fn(bddl, goal_bddls, seed):
    def _fn():
        from libero.libero.envs import OffScreenRenderEnv
        import libero.libero.envs.bddl_utils as BDDLUtils
        env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256)
        env.seed(seed)
        P = env.env
        goals = [BDDLUtils.robosuite_parse_problem(g)["goal_state"] for g in goal_bddls]
        _orig = env.step

        def _step(act):
            obs, r, done, info = _orig(act)
            info = dict(info or {})
            sat = []
            for gs in goals:
                try:
                    sat.append(bool(all(P._eval_predicate(s) for s in gs)))
                except Exception:
                    sat.append(False)
            info["goal_sat"] = sat
            return obs, r, False, info  # the chain controller decides when the episode ends
        env.step = _step
        return env
    return _fn


async def main():
    ensure_libero_config()
    from libero.libero import benchmark, get_libero_path
    suite = benchmark.get_benchmark_dict()["libero_goal"]()
    langs = [suite.get_task(i).language for i in range(suite.n_tasks)]
    bddls = [str(Path(get_libero_path("bddl_files")) / suite.get_task(i).problem_folder / suite.get_task(i).bddl_file) for i in range(suite.n_tasks)]
    rng = random.Random(a.chains_seed)
    chains = [rng.sample(range(len(langs)), a.chain_len) for _ in range(a.n_chains)]
    P = a.num_parallel
    vec = await asyncio.to_thread(LiberoAsyncVectorEnv, env_fns=[make_env_fn(bddls[0], bddls, a.seed)] * P,
                                  dummy_env_fn=make_env_fn(bddls[0], bddls, a.seed))
    results = []
    t0 = time.time()
    async with MultiPolicyWSClient(a.server_uri, P, timeout_s=600) as client:
        n_exec = int(client.metadata[0].get("action_steps", 10))
        for c0 in range(0, len(chains), P):
            batch = chains[c0:c0 + P]; n = len(batch)
            inits = [suite.get_task_init_states(ch[0])[(c0 + j) % 50] for j, ch in enumerate(batch)] + [suite.get_task_init_states(0)[0]] * (P - n)
            obs = await asyncio.to_thread(vec.set_init_state_each, inits)
            gstate = [LiberoGripperCommandState.from_obs(o) for o in obs]
            active = [j < n for j in range(P)]
            for _ in range(a.num_steps_wait):
                ob, _, _, _ = await asyncio.to_thread(vec.step_selective, active, [LIBERO_DUMMY_ACTION] * P)
                obs = [x if x is not None else o for x, o in zip(ob, obs)]
            pos = [0] * P; since = [0] * P; held = [0] * P; done_steps = [[] for _ in range(P)]; total = [0] * P
            for j in range(P):
                client.clear_chunk(j)
            while any(active):
                live = [j for j in range(P) if active[j]]
                await client.infer_batch(live, [build_libero_raw_obs(obs=obs[j], task_description=langs[batch[j][pos[j]]], embodiment_type="libero")[0] for j in live])
                chunks = {j: client.action_chunks[j] for j in live}
                for s in range(n_exec):
                    acts = [LIBERO_DUMMY_ACTION] * P; mask = [False] * P
                    for j in live:
                        if active[j]:
                            single = {k: (np.asarray(v)[min(s, np.asarray(v).shape[0] - 1)] if np.asarray(v).ndim >= 1 else v) for k, v in chunks[j].items()}
                            acts[j] = single_action_to_libero_action(single, obs=obs[j], gripper_state=gstate[j]); mask[j] = True
                    if not any(mask):
                        break
                    ob, _, _, infos = await asyncio.to_thread(vec.step_selective, mask, acts)
                    advanced = False
                    for j in range(P):
                        if not mask[j]:
                            continue
                        obs[j] = ob[j]; since[j] += 1; total[j] += 1
                        sat = infos[j].get("goal_sat", [False] * len(langs))
                        held[j] = held[j] + 1 if sat[batch[j][pos[j]]] else 0
                        if held[j] >= a.hold:  # command completed -> next one
                            done_steps[j].append(since[j]); pos[j] += 1; since[j] = 0; held[j] = 0; advanced = True
                            if pos[j] >= a.chain_len:
                                active[j] = False
                        elif since[j] >= a.task_budget:
                            active[j] = False
                    if advanced:
                        break  # re-plan immediately with the new command
            for j in range(n):
                results.append({"chain": batch[j], "completed": pos[j], "steps_per_command": done_steps[j]})
            comp = [r["completed"] for r in results]
            print(json.dumps({"chains_done": len(results), "avg_len": round(float(np.mean(comp)), 3), "t": round(time.time() - t0)}), flush=True)
    await asyncio.to_thread(vec.close)
    comp = np.array([r["completed"] for r in results])
    summary = {"n_chains": len(results), "chain_len": a.chain_len, "avg_completed": float(comp.mean()),
               "success_at": {k: float((comp >= k).mean()) for k in range(1, a.chain_len + 1)}, "langs": langs, "chains": results}
    json.dump(summary, open(f"{a.output_dir}/chain_results.json", "w"), indent=2)
    print(json.dumps({k: summary[k] for k in ("n_chains", "avg_completed", "success_at")}), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
