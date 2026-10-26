"""Paired comparisons for the pi0.7-inspired probes (same statistics as 10_compare.py).

A condition is DIR:COND, where DIR is a results directory holding steer_summary.json (results/steer_gc/<name> or
results/steer/<name>) and COND a condition in it. Trials are paired by (command pair t, trial index), i.e. by the
same initial state. Task-macro paired difference over the 10 command pairs, paired bootstrap CI (resampling trials
within each pair), exact McNemar test on discordant trials.
Usage: python gc_analyze.py OUT.json  A_DIR:A_COND,B_DIR:B_COND ...   (A minus B)
"""
import json
import math
import random
import sys

R = "/export/home/kaziz/g05/results"


def load(spec):
    d, cond = spec.rsplit(":", 1)
    path = d if d.startswith("/") else f"{R}/{d}"
    s = json.load(open(f"{path}/steer_summary.json"))[cond]
    return {int(t): {int(k): bool(v) for k, v in p["trial_successes"].items()} for t, p in s["pairs"].items() if p}


def mcnemar_p(b, c):
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def compare(a, b, reps=10000, seed=0):
    A, B = load(a), load(b)
    pairs = {t: [(A[t][i], B[t][i]) for i in sorted(set(A[t]) & set(B[t]))] for t in sorted(set(A) & set(B))}
    pairs = {t: v for t, v in pairs.items() if v}
    diffs = [sum(x - y for x, y in v) / len(v) for v in pairs.values()]
    rng = random.Random(seed)
    boots = sorted(sum(sum(x - y for x, y in (v[rng.randrange(len(v))] for _ in v)) / len(v) for v in pairs.values()) / len(pairs)
                   for _ in range(reps))
    b_ = sum(1 for v in pairs.values() for x, y in v if x and not y)
    c_ = sum(1 for v in pairs.values() for x, y in v if y and not x)
    n = sum(len(v) for v in pairs.values())
    return {"A": a, "B": b, "pairs": len(pairs), "trials": n,
            "A_succ": sum(x for v in pairs.values() for x, _ in v), "B_succ": sum(y for v in pairs.values() for _, y in v),
            "macro_diff": sum(diffs) / len(diffs), "ci95": [boots[int(0.025 * reps)], boots[int(0.975 * reps)]],
            "A_only": b_, "B_only": c_, "mcnemar_p": mcnemar_p(b_, c_),
            "per_pair_A": {t: sum(x for x, _ in v) for t, v in pairs.items()}, "per_pair_B": {t: sum(y for _, y in v) for t, v in pairs.items()}}


if __name__ == "__main__":
    res = []
    for spec in sys.argv[2:]:
        a, b = spec.split(",")
        try:
            r = compare(a, b)
        except (FileNotFoundError, KeyError) as e:
            print(f"{spec}: missing ({e!r})")
            continue
        res.append(r)
        print(f"{a:>34} {r['A_succ']:3d}/{r['trials']:<3d} vs {b:<34} {r['B_succ']:3d}/{r['trials']:<3d} diff {r['macro_diff']:+.3f} "
              f"CI [{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}]  discordant {r['A_only']}/{r['B_only']}  McNemar p={r['mcnemar_p']:.2g}")
    json.dump(res, open(sys.argv[1], "w"), indent=2)
