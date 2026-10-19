"""Summary of the LIBERO-Mem cycle-count decoders (46_lmem_count.py) with paired episode-bootstrap CIs."""
import json, os
import numpy as np
os.chdir("/export/home/kaziz/g05/results")
C = ["current", "first", "snapshot", "offsets", "uniform8"]
R = {c: json.load(open(f"lmem_count/{c}.json")) for c in C if os.path.exists(f"lmem_count/{c}.json")}
names = ["bowl x1", "bottle x1", "bowl x3", "bottle x3", "bowl x5", "bowl x7", "swap 2", "rotate 3", "bskt that", "bskt empty"]
print("cond      all    after1st ", " ".join(f"{n:>10s}" for n in names))
for c, r in R.items():
    print(f"{c:9s} {r['acc_all']:.3f}  {r['acc_after_first_cycle']:.3f}   ", " ".join(f"{r['per_task'][str(t)]['acc']:10.3f}" for t in range(10)))
print("majority                  ", " ".join(f"{R['current']['per_task'][str(t)]['majority_baseline']:10.3f}" for t in range(10)))
print("max count                 ", " ".join(f"{R['current']['per_task'][str(t)]['max_count']:10d}" for t in range(10)))
for c in R:
    print("per_count", c, {k: round(v['acc'], 3) for k, v in R[c]['per_count'].items()})
rng = np.random.RandomState(0)
cur = np.load("lmem_count/current_rows.npz")
out = {}
for c in ["offsets", "uniform8", "snapshot", "first"]:
    m = np.load(f"lmem_count/{c}_rows.npz"); assert (m["rows"] == cur["rows"]).all()
    for nm, mk in [("all", np.ones(len(cur["y"]), bool)), ("after1st", cur["y"] >= 1), ("repeat_tasks_after1st", np.isin(cur["task"], [2, 3, 4, 5]) & (cur["y"] >= 1)),
                   ("swap_rotate_after1st", np.isin(cur["task"], [6, 7]) & (cur["y"] >= 1))]:
        d = (m["pred"] == m["y"]).astype(float) - (cur["pred"] == cur["y"]).astype(float)
        ep = cur["task"] * 1000 + cur["demo"]; ue, inv = np.unique(ep[mk], return_inverse=True)
        s = np.bincount(inv, d[mk]); n = np.bincount(inv)
        bs = [s[i].sum() / n[i].sum() for i in (rng.randint(0, len(ue), len(ue)) for _ in range(2000))]
        out[f"{c}|{nm}"] = {"acc_cur": float((cur["pred"] == cur["y"])[mk].mean()), "acc_mem": float((m["pred"] == m["y"])[mk].mean()),
                            "diff": float(d[mk].mean()), "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], "episodes": int(len(ue))}
        print(f"{c:9s} vs current [{nm:22s}]: {out[f'{c}|{nm}']['acc_cur']:.3f} -> {out[f'{c}|{nm}']['acc_mem']:.3f}  diff {d[mk].mean():+.3f}  CI95 [{np.percentile(bs, 2.5):+.3f}, {np.percentile(bs, 97.5):+.3f}]  episodes={len(ue)}")
json.dump({"per_cond": R, "paired": out}, open("lmem_count/summary.json", "w"), indent=2)
