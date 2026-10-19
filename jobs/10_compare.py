"""Paired comparison of screen conditions (same tasks, same init states, same env seed).

Task-macro paired difference with a paired bootstrap CI (resampling trials within task) and exact McNemar
counts on discordant pairs. Conditions: directories under results/screen/<name>/screen_summary.json.
Usage: python 10_compare.py OUT.json  A:B  A:B ...   (A minus B)
"""
import json
import math
import random
import sys

ROOT = "/export/home/kaziz/g05/results/screen"


def load(name):
    s = json.load(open(f"{ROOT}/{name}/screen_summary.json"))
    out = {}
    for t in s["tasks"]:
        if t.get("missing") or not t.get("trial_successes"):
            continue
        out[t["task"]] = {int(k): bool(v) for k, v in t["trial_successes"].items()}
    return out, s


def mcnemar_p(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def compare(a, b, reps=10000, seed=0):
    A, sa = load(a)
    B, sb = load(b)
    tasks = sorted(set(A) & set(B))
    per_task, pairs = [], {}
    for t in tasks:
        common = sorted(set(A[t]) & set(B[t]))
        pairs[t] = [(A[t][i], B[t][i]) for i in common]
        ra = sum(x for x, _ in pairs[t]) / len(common)
        rb = sum(y for _, y in pairs[t]) / len(common)
        per_task.append({"task": t, "n": len(common), "A": ra, "B": rb, "diff": ra - rb})
    macro = sum(p["diff"] for p in per_task) / max(len(per_task), 1)
    rng = random.Random(seed)
    boots = []
    for _ in range(reps):
        d = 0.0
        for t in tasks:
            pr = pairs[t]
            smp = [pr[rng.randrange(len(pr))] for _ in pr]
            d += sum(x - y for x, y in smp) / len(smp)
        boots.append(d / len(tasks))
    boots.sort()
    b_ = sum(1 for t in tasks for x, y in pairs[t] if x and not y)
    c_ = sum(1 for t in tasks for x, y in pairs[t] if y and not x)
    return {"A": a, "B": b, "tasks": len(tasks), "trials": sum(len(p) for p in pairs.values()),
            "A_rate": sum(p["A"] for p in per_task) / len(per_task), "B_rate": sum(p["B"] for p in per_task) / len(per_task),
            "macro_diff": macro, "ci95": [boots[int(0.025 * reps)], boots[int(0.975 * reps)]],
            "A_only_success": b_, "B_only_success": c_, "mcnemar_p": mcnemar_p(b_, c_), "per_task": per_task}


if __name__ == "__main__":
    out = sys.argv[1]
    res = []
    for spec in sys.argv[2:]:
        a, b = spec.split(":")
        try:
            r = compare(a, b)
        except FileNotFoundError as e:
            r = {"A": a, "B": b, "error": str(e)}
        res.append(r)
        if "error" in r:
            print(f"{a} vs {b}: missing ({r['error']})")
        else:
            print(f"{a:>22} vs {b:<22} {r['A_rate']:.3f} vs {r['B_rate']:.3f}  diff {r['macro_diff']:+.3f} "
                  f"CI [{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}]  discordant {r['A_only_success']}/{r['B_only_success']} "
                  f"McNemar p={r['mcnemar_p']:.3f}")
    json.dump(res, open(out, "w"), indent=2)
