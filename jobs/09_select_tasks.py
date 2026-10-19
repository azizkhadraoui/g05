"""Pre-registered screen-task selection (protocol section 5: pick tasks before examining hybrid results).

Rule (fixed 2026-10-17, before any hybrid evaluation): the 6 LIBERO tasks with the lowest B00 success rate
(released G0.5 FM, full 50-trial protocol); ties broken by suite order libero_10 > libero_goal > libero_spatial >
libero_object, then task id. Reads the B00 screen_summary.json written by 08_eval_screen.sh.
"""
import json
import sys

src = sys.argv[1] if len(sys.argv) > 1 else "/export/home/kaziz/g05/results/screen/B00_full/screen_summary.json"
out = sys.argv[2] if len(sys.argv) > 2 else "/export/home/kaziz/g05/results/screen/tasks.txt"
n = int(sys.argv[3]) if len(sys.argv) > 3 else 6
order = {"libero_10": 0, "libero_goal": 1, "libero_spatial": 2, "libero_object": 3}
s = json.load(open(src))
rows = []
for t in s["tasks"]:
    if t.get("missing"):
        continue
    suite, tid = t["task"].split(":")
    rows.append((t["successes"] / t["total"], order.get(suite, 9), int(tid), suite, t.get("description", "")))
rows.sort()
sel = rows[:n]
for r in sel:
    print(f"{r[3]}:{r[2]}  B00={r[0]:.2f}  {r[4]}")
with open(out, "w") as f:
    f.write(" ".join(f"{r[3]}:{r[2]}" for r in sel) + "\n")
print("wrote", out)
