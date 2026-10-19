"""Collect every result into results/report_data.json (plain python3, no third-party imports)."""
import glob
import json
import math
import os
import random
import re

R = "/export/home/kaziz/g05/results"
out = {}


def jload(p):
    try:
        return json.load(open(p))
    except Exception:
        return None


# B00 full
b = jload(f"{R}/screen/B00_full/screen_summary.json")
if b:
    suites = {}
    for t in b["tasks"]:
        if t.get("missing"):
            continue
        s = t["task"].split(":")[0]
        a = suites.setdefault(s, [0, 0])
        a[0] += t["successes"]; a[1] += t["total"]
    out["B00_full"] = {"suites": {k: {"successes": v[0], "total": v[1], "rate": v[0] / v[1]} for k, v in suites.items()},
                       "successes": b["successes"], "total": b["total"], "rate": b["rate"],
                       "tasks": [{k: t.get(k) for k in ("task", "description", "successes", "total")} for t in b["tasks"]]}
out["W00"] = jload(f"{R}/lawm_sanity/summary.json")
out["lawm_param_drift"] = jload(f"{R}/lawm_sanity/param_drift.json")
out["F00"] = jload(f"{R}/hybrid/P_s0/f00_summary.json")
out["screen_tasks"] = open(f"{R}/screen/tasks.txt").read().split() if os.path.exists(f"{R}/screen/tasks.txt") else None

# training curves (sparse)
curves = {}
for name in ("P_s0", "F01_s0", "F02_s0"):
    p = f"{R}/hybrid/{name}/train_log.jsonl"
    if not os.path.exists(p):
        continue
    rows = [json.loads(l) for l in open(p) if '"train"' in l]
    setup = [json.loads(l) for l in open(p) if '"setup"' in l]
    curves[name] = {"points": [{k: r.get(k) for k in ("step", "loss", "cos")} for r in rows if r["step"] % 250 == 0 or r["step"] == 1],
                    "n_trainable": setup[-1].get("n_trainable") if setup else None}
out["training"] = curves

# screens
scr = {}
for d in sorted(glob.glob(f"{R}/screen/*/screen_summary.json")):
    n = d.split("/")[-2]
    if n in ("B00_full",) or n.startswith("egltest"):
        continue
    s = jload(d)
    scr[n] = {"successes": s["successes"], "total": s["total"], "rate": s["rate"],
              "tasks": [{k: t.get(k) for k in ("task", "successes", "total")} for t in s["tasks"]]}
out["screens"] = scr
out["screen_compare"] = jload(f"{R}/screen/compare.json")


# steering / memory
def paired(a_pairs, b_pairs, reps=5000, seed=0):
    """a_pairs/b_pairs: {target: {trial: bool}} -> macro diff over targets with bootstrap CI (paired by trial id)."""
    targets = sorted(set(a_pairs) & set(b_pairs))
    data = {}
    for t in targets:
        common = sorted(set(a_pairs[t]) & set(b_pairs[t]))
        if common:
            data[t] = [(a_pairs[t][i], b_pairs[t][i]) for i in common]
    if not data:
        return None
    def macro(dd):
        return sum(sum(x - y for x, y in v) / len(v) for v in dd.values()) / len(dd)
    rng = random.Random(seed)
    boots = sorted(macro({t: [v[rng.randrange(len(v))] for _ in v] for t, v in data.items()}) for _ in range(reps))
    bb = sum(1 for v in data.values() for x, y in v if x and not y)
    cc = sum(1 for v in data.values() for x, y in v if y and not x)
    n = bb + cc
    p = 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(min(bb, cc) + 1)) / 2 ** n)
    return {"diff": macro(data), "ci95": [boots[int(0.025 * reps)], boots[int(0.975 * reps)]],
            "a_only": bb, "b_only": cc, "mcnemar_p": p, "n": sum(len(v) for v in data.values())}


steer = {}
for d in sorted(glob.glob(f"{R}/steer/*/steer_summary.json")):
    pol = d.split("/")[-2]
    s = jload(d)
    steer[pol] = s
out["steer"] = {pol: {c: {k: v[k] for k in ("successes", "total", "rate", "missing_pairs")} for c, v in s.items()}
                for pol, s in steer.items()}


def trials(pol, cond):
    s = steer.get(pol, {}).get(cond)
    if not s:
        return None
    return {t: {int(k): bool(v) for k, v in (p.get("trial_successes") or {}).items()}
            for t, p in s["pairs"].items() if p}


cmp = []
within = [("cf", "ref"), ("switch", "ref"), ("m_none", "ref"), ("m_stale", "ref"), ("rev_none", "switch"), ("rev_stale", "switch")]
for pol in steer:
    for a, b_ in within:
        A, B = trials(pol, a), trials(pol, b_)
        if A and B:
            cmp.append({"policy": pol, "A": a, "B": b_, **(paired(A, B) or {})})
for a_pol, b_pol in (("F02", "F01"), ("F02", "B00"), ("F01", "B00")):
    for c in ("ref", "cf", "switch", "m_none", "m_stale", "rev_none", "rev_stale"):
        A, B = trials(a_pol, c), trials(b_pol, c)
        if A and B:
            cmp.append({"policy": f"{a_pol} vs {b_pol}", "A": c, "B": c, **(paired(A, B) or {})})
out["steer_compare"] = cmp

# latency (server timing lines: infer=XXms for the batch)
lat = {}
for d in glob.glob(f"{R}/screen/*/server.log"):
    n = d.split("/")[-2]
    xs = []
    for line in open(d, errors="replace"):
        m = re.search(r"batch_size=(\d+), waited=[\d.]+ms, infer=([\d.]+)ms", line)
        if m:
            xs.append((int(m.group(1)), float(m.group(2))))
    if xs:
        per = sorted(v / bsz for bsz, v in xs)
        lat[n] = {"calls": len(xs), "median_ms_per_obs": per[len(per) // 2],
                  "median_batch": sorted(b_ for b_, _ in xs)[len(xs) // 2]}
out["latency_server"] = lat
out["world_model"] = jload(f"{R}/wm_s0/wm_eval.json")
out["memory"] = {"with_memory": jload(f"{R}/hybrid/Mmem_s0/memory_eval.json"), "masked_control": jload(f"{R}/hybrid/Mctrl_s0/memory_eval.json"),
                 "SM1_snapshot_memory": jload(f"{R}/hybrid/SM1_s0/memory_eval.json"),
                 "SM1_snapshot_stats": jload(f"{R}/feat_cache/snapshot_stats.json")}
rec = glob.glob(f"{R}/recovery/*/rec/*.npz")
out["recovery_data"] = {"episodes": len(rec)}
b01 = glob.glob(f"{R}/B01_lawam/**/*summary*.json", recursive=True) + glob.glob(f"{R}/B01_lawam/**/*result*.json", recursive=True)
out["B01_files"] = b01[:20]
json.dump(out, open(f"{R}/report_data.json", "w"), indent=2, default=float)
print("wrote", f"{R}/report_data.json", "keys:", list(out))
