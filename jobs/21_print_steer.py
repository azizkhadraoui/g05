import json
d = json.load(open("/export/home/kaziz/g05/results/report_data.json"))
for c in d["steer_compare"]:
    print("%-12s %9s vs %-7s diff %+.2f CI [%+.2f, %+.2f] p=%.2g n=%d" % (c["policy"], c["A"], c["B"], c["diff"], c["ci95"][0], c["ci95"][1], c["mcnemar_p"], c["n"]))
for pol in d["steer"]:
    s = json.load(open("/export/home/kaziz/g05/results/steer/%s/steer_summary.json" % pol))
    for cond in ["switch", "m_none", "m_stale", "rev_none", "rev_stale"]:
        if cond in s:
            print(pol, cond, [p["successes"] if p else None for t, p in sorted(s[cond]["pairs"].items(), key=lambda x: int(x[0]))])
