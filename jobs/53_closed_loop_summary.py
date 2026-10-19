"""Closed-loop results: memory (LIBERO-Mem memory module, MEM-2), steering (recovery fine-tune), world model (planner).
Paired comparisons share initial states and seeds; exact McNemar on discordant pairs. Output: results/closed_loop_summary.json
"""
import glob
import json
import math
import os

R = "/export/home/kaziz/g05/results"
out = {}


def mcnemar(a, b):
    """a, b: dict trial_key -> bool (paired). Returns diff and exact two-sided p."""
    keys = sorted(set(a) & set(b))
    n01 = sum(1 for k in keys if not a[k] and b[k]); n10 = sum(1 for k in keys if a[k] and not b[k])
    n = n01 + n10
    p = 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(min(n01, n10) + 1)) / 2 ** n)
    return {"n": len(keys), "rate_a": sum(a[k] for k in keys) / max(len(keys), 1), "rate_b": sum(b[k] for k in keys) / max(len(keys), 1),
            "b_wins": n01, "a_wins": n10, "p": p}


# ---------------- memory: LIBERO-Mem closed loop with the memory module ----------------
for run in sorted(glob.glob(f"{R}/lmem_cl/*")):
    name = os.path.basename(run)
    if name.startswith("smoke"):
        continue
    conds = {}
    for cdir in sorted(glob.glob(f"{run}/[A-G]")):
        c = os.path.basename(cdir); trials = {}; sub = []; per_task = {}
        for tdir in sorted(glob.glob(f"{cdir}/t*")):
            f = glob.glob(f"{tdir}/*_parallel_results.json")
            if not f:
                continue
            r = json.load(open(f[0]))["tasks"][0]; g = r.get("goal_len") or 1
            fin = r.get("trial_final", {}); s_ = r.get("trial_subgoals", {})
            t = os.path.basename(tdir)
            for k, v in fin.items():
                trials[f"{t}_{k}"] = bool(v["final_done"])
            per_task[t] = {"success": sum(v["final_done"] for v in fin.values()), "n": len(fin),
                           "subgoal_completion": sum(min(v, g) for v in s_.values()) / max(len(s_), 1) / g,
                           "halted": sum(v.get("halt_step") is not None for v in fin.values())}
            sub += [min(v, g) / g for v in s_.values()]
        if trials:
            conds[c] = {"success": sum(trials.values()) / len(trials), "n": len(trials), "subgoal_completion": sum(sub) / max(len(sub), 1),
                        "per_task": per_task, "_trials": trials}
    pairs = {}
    for a_, b_ in [("A", "C"), ("B", "C"), ("D", "C"), ("C", "E"), ("A", "E"), ("A", "F"), ("G", "F"), ("F", "C")]:
        if a_ in conds and b_ in conds:
            pairs[f"{a_}_vs_{b_}"] = mcnemar(conds[a_]["_trials"], conds[b_]["_trials"])
    for v in conds.values():
        v.pop("_trials")
    out.setdefault("memory_libero_mem", {})[name] = {"conditions": conds, "paired": pairs}

# ---------------- memory: MEM-2 (LIBERO-Goal instruction withdrawal) ----------------
for pol in ("B00t4", "RB25", "RB20"):
    m2 = f"{R}/mem2/{pol}/mem2_summary.json"
    if os.path.exists(m2):
        s2 = json.load(open(m2))
        out.setdefault("memory_mem2", {})[pol] = {c: {"rate": v["rate"], "n": v["total"], "memory_inference_acc": v.get("memory_inference_acc")} for c, v in s2.items()}


# ---------------- steering: recovery fine-tune vs matched control ----------------
def steer_trials(policy):
    f = f"{R}/steer/{policy}/steer_summary.json"
    if not os.path.exists(f):
        return None
    s = json.load(open(f))
    res = {}
    for cond, v in s.items():
        tr = {}
        for t, pr in (v.get("pairs") or {}).items():
            if pr and pr.get("trial_successes"):
                for k, ok in pr["trial_successes"].items():
                    tr[f"{t}_{k}"] = bool(ok)
        res[cond] = tr
    return res


steer = {}
for pol in ["B00", "B00t4", "F01", "F01cont_t4", "F01recov_t4"]:
    st = steer_trials(pol)
    if st:
        steer[pol] = {c: {"rate": sum(v.values()) / max(len(v), 1), "n": len(v)} for c, v in st.items()}
paired = {}
a_, b_ = steer_trials("F01cont_t4"), steer_trials("F01recov_t4")
if a_ and b_:
    for c in set(a_) & set(b_):
        paired[f"recov_vs_cont|{c}"] = mcnemar(a_[c], b_[c])
g_, b2 = steer_trials("B00"), steer_trials("F01recov_t4")
if g_ and b2:
    for c in set(g_) & set(b2):
        paired[f"recov_vs_G05|{c}"] = mcnemar(g_[c], b2[c])
out["steering"] = {"policies": steer, "paired": paired}

# ---------------- world model: planner (runs split by task are merged per mode) ----------------
def planner(prefix):
    modes = {}
    for mdir in sorted(glob.glob(f"{R}/wmplan/{prefix}*/*/")):
        f = f"{mdir}/planner_results.json"
        if not os.path.exists(f):
            continue
        r = json.load(open(f)); m = os.path.basename(mdir.rstrip("/"))
        d = modes.setdefault(m, {"_t": {}, "frac_target_chosen": {}, "per_task": {}})
        for t, v in r["tasks"].items():
            for tr in v["trials"]:
                d["_t"][f"{t}_{tr['trial']}"] = tr["success"]
            d["frac_target_chosen"][t] = v.get("frac_target_chosen")
            d["per_task"][t] = f"{v['successes']}/{v['total']}"
    for m, d in modes.items():
        d["success"] = sum(d["_t"].values()) / max(len(d["_t"]), 1); d["n"] = len(d["_t"])
    pairs = {}
    for a_, b_ in [("random", "wm"), ("random", "wm_commit"), ("blank", "wm"), ("blank", "wm_commit"), ("wm", "oracle"), ("wm_commit", "oracle"), ("oracle", "wm")]:
        if a_ in modes and b_ in modes:
            pairs[f"{a_}_vs_{b_}"] = mcnemar(modes[a_]["_t"], modes[b_]["_t"])
    for d in modes.values():
        d.pop("_t")
    return {"modes": modes, "paired": pairs}


plan = {"goal_video": planner("full"), "goal_switch_at_2s": planner("switch"), "goal_image_vlm": planner("vlm_goalimg"),
        "goal_image_wm": planner("wm_goalimg")}
out["world_model_planner"] = plan
json.dump(out, open(f"{R}/closed_loop_summary.json", "w"), indent=2)
print(json.dumps(out, indent=1)[:12000])
