"""Statistics for the two memory results (paired, episode-level bootstrap; episodes are the independent units).

MEM-1b: task-intent decoding, memory (offsets / snapshot / first) vs current frame, same held-out rows, seeds 0-2.
WM-MEM: world-model rollout error, memory (short / long) vs Markovian control, same held-out rows and seed.
Output: results/mem_stats.json
"""
import glob
import json
import os

import numpy as np

G = os.environ.get("G05_ROOT", "/export/home/kaziz/g05")
R = f"{G}/results"
rng = np.random.RandomState(0)
B = 2000


def boot_diff(a, b, groups):
    """mean(a - b) with a 95% CI from resampling groups (episodes) with replacement."""
    d = a.astype(np.float64) - b.astype(np.float64)
    ug, inv = np.unique(groups, return_inverse=True)
    s = np.bincount(inv, d); n = np.bincount(inv)
    stats = []
    for _ in range(B):
        idx = rng.randint(0, len(ug), len(ug))
        stats.append(s[idx].sum() / n[idx].sum())
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return {"diff": float(d.mean()), "ci95": [float(lo), float(hi)], "p_le0": float(np.mean(np.array(stats) <= 0)),
            "n_rows": int(len(d)), "n_episodes": int(len(ug))}


out = {"mem1b": {}, "wm_mem": {}}
# ---------------- MEM-1b ----------------
M = f"{R}/mem_intent"
SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
windows = {"all_>=2s": (40, 10_000), "2-4s": (40, 80), "4-8s": (80, 160), ">=4s": (80, 10_000)}
for seed in (0, 1, 2):
    tag = lambda c: c if seed == 0 else f"{c}_seed{seed}"  # noqa: E731
    f_cur = f"{M}/{tag('current')}_rows.npz"
    if not os.path.exists(f_cur):
        continue
    cur = np.load(f_cur)
    for cond in ("offsets", "snapshot", "first"):
        f = f"{M}/{tag(cond)}_rows.npz"
        if not os.path.exists(f):
            continue
        mem = np.load(f)
        assert (mem["rows"] == cur["rows"]).all()
        shared = np.isin(cur["suite"], [0, 1, 2])
        r = {}
        for wname, (lo, hi) in windows.items():
            for sname, smask in [("shared_scene", shared)] + [(s, cur["suite"] == i) for i, s in enumerate(SUITES)]:
                mk = smask & (cur["frame"] >= lo) & (cur["frame"] < hi)
                if mk.sum() == 0:
                    continue
                bd = boot_diff(mem["correct"][mk], cur["correct"][mk], cur["episode"][mk])
                bd.update({"acc_current": float(cur["correct"][mk].mean()), "acc_memory": float(mem["correct"][mk].mean()),
                           "err_current": float(1 - cur["correct"][mk].mean()), "err_memory": float(1 - mem["correct"][mk].mean())})
                r[f"{sname}|{wname}"] = bd
        out["mem1b"][f"seed{seed}|{cond}_vs_current"] = r
# seed-pooled summary for the headline numbers
for cond in ("offsets", "snapshot"):
    for key in ("shared_scene|4-8s", "shared_scene|all_>=2s", "libero_spatial|all_>=2s", "shared_scene|>=4s"):
        vals = [v[key] for k, v in out["mem1b"].items() if k.endswith(f"|{cond}_vs_current") and key in v]
        if vals:
            out["mem1b"][f"summary|{cond}|{key}"] = {
                "n_seeds": len(vals), "err_current_mean": float(np.mean([v["err_current"] for v in vals])),
                "err_memory_mean": float(np.mean([v["err_memory"] for v in vals])),
                "all_seeds_ci_above_0": bool(all(v["ci95"][0] > 0 for v in vals)),
                "per_seed_diff": [v["diff"] for v in vals]}

# ---------------- WM-MEM ----------------
meta = np.load(f"{R}/feat_cache/meta.npz")
ep = meta["episode_index"]
base = f"{R}/wmmem_markov"
if os.path.exists(f"{base}/err_step1.npy"):
    ev = np.load(f"{base}/eval_rows.npy")
    for name in ("short", "long"):
        d = f"{R}/wmmem_{name}"
        if not os.path.exists(f"{d}/err_step1.npy"):
            continue
        assert (np.load(f"{d}/eval_rows.npy") == ev).all()
        r = {"eval": json.load(open(f"{d}/wm_eval.json")), "markov_eval": json.load(open(f"{base}/wm_eval.json"))}
        for s in (1, 2, 4):
            em, eh = np.load(f"{base}/err_step{s}.npy"), np.load(f"{d}/err_step{s}.npy")
            bd = boot_diff(em, eh, ep[ev])  # > 0 means memory has lower error
            bd.update({"mse_markov": float(em.mean()), "mse_memory": float(eh.mean()),
                       "rel_reduction": float(1 - eh.mean() / em.mean()), "frac_rows_memory_better": float((eh < em).mean())})
            r[f"step{s}"] = bd
        out["wm_mem"][f"{name}_vs_markov"] = r
json.dump(out, open(f"{R}/mem_stats.json", "w"), indent=2)
for k, v in out["mem1b"].items():
    if k.startswith("summary"):
        print(k, json.dumps(v))
for k, v in out["wm_mem"].items():
    print(k, {s: {kk: round(vv, 4) if isinstance(vv, float) else vv for kk, vv in v[s].items()} for s in ("step1", "step2", "step4")})
    ab = {s: (round(v["eval"][s]["ratio"], 4), round(v["eval"].get(f"hist_ablated_{s}", {}).get("ratio", float("nan")), 4)) for s in ("step1", "step2", "step4")}
    print("  ratio vs persistence (memory, memory-ablated):", ab, " markov:", {s: round(v["markov_eval"][s]["ratio"], 4) for s in ("step1", "step2", "step4")})
