"""Paired comparisons for the RoboBrain closed-loop runs (same initial states and seeds as the reference runs)."""
import glob, json, math, os
R = "/export/home/kaziz/g05/results"


def mc(a, b):
    ks = sorted(set(a) & set(b)); n01 = sum(1 for k in ks if not a[k] and b[k]); n10 = sum(1 for k in ks if a[k] and not b[k]); n = n01 + n10
    p = 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(min(n01, n10) + 1)) / 2 ** n)
    return {"n": len(ks), "a": sum(a[k] for k in ks), "b": sum(b[k] for k in ks), "b_wins": n01, "a_wins": n10, "p": p}


def mem2(pol, cond):
    out = {}
    for d in glob.glob(f"{R}/mem2/{pol}/{cond}_t*"):
        f = glob.glob(f"{d}/*_parallel_results.json")
        if f:
            t = d.rsplit("_t", 1)[1]; k = json.load(open(f[0]))["tasks"][0]
            out.update({f"{t}_{i}": bool(v) for i, v in k["trial_successes"].items()})
    return out


def plan(run, mode):
    out = {}
    for f in glob.glob(f"{R}/wmplan/{run}/{mode}/planner_results.json"):
        for t, v in json.load(open(f))["tasks"].items():
            out.update({f"{t}_{tr['trial']}": tr["success"] for tr in v["trials"]})
    return out


def lmem(cond):
    out = {}
    for tdir in glob.glob(f"{R}/lmem_cl/expert/{cond}/t*"):
        f = glob.glob(f"{tdir}/*_parallel_results.json")
        if f:
            fin = json.load(open(f[0]))["tasks"][0].get("trial_final", {})
            out.update({f"{os.path.basename(tdir)}_{k}": bool(v["final_done"]) for k, v in fin.items()})
    return out


res = {}
res["mem2_rb_multi_vs_no_memory"] = mc(mem2("B00t4", "m_none"), mem2("RB25", "vlm_multi"))
res["mem2_rb_current_vs_no_memory"] = mc(mem2("B00t4", "m_none"), mem2("RB25", "vlm_current"))
res["mem2_trained_vs_rb_multi"] = mc(mem2("RB25", "vlm_multi"), mem2("B00t4", "mem_current"))
wmimg = plan("wm_goalimg", "wm"); vlm = plan("vlm_goalimg", "vlm")
rnd = {**plan("full_base", "random")}; wmv = {**plan("full_wm_a", "wm"), **plan("full_wm_b", "wm")}
res["plan_wm_goalimage_vs_vlm_goalimage"] = mc(wmimg, vlm)
res["plan_random_vs_vlm_goalimage"] = mc(rnd, vlm)
res["plan_vlm_goalimage_vs_wm_goalvideo"] = mc(vlm, wmv)
res["lmem_alone_vs_rb_counter"] = mc(lmem("A"), lmem("F"))
res["lmem_rb_counter_vs_trained_counter"] = mc(lmem("F"), lmem("C"))
res["lmem_rb_current_vs_rb_multi"] = mc(lmem("G"), lmem("F"))
for k, v in res.items():
    print(f"{k:40s} {v['a']}/{v['n']} -> {v['b']}/{v['n']}  b_wins={v['b_wins']} a_wins={v['a_wins']} p={v['p']:.1e}")
json.dump(res, open(f"{R}/robobrain_stats.json", "w"), indent=2)
