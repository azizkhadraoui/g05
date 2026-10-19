"""RoboBrain offline check on held-out LIBERO-Goal demos (episode % 5 == 0), before closed-loop use.

  task_multi    frames at 0, 0.5, 1, 1.5, 2 s -> "which of the 10 tasks?"   (memory)
  task_current  frame at 2 s only            -> same question               (no memory)
  goal_choice   [first frame, final frame of ANOTHER held-out demo of the same task] -> "which instruction reaches the goal?"
Output: results/robobrain/offline_<model>.json
"""
import argparse
import glob
import importlib
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import t4_fp16  # noqa: F401,E402
from robobrain import RoboBrain, first_int, goal_choice_prompt, task_choice_prompt  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--model", required=True)
p.add_argument("--batch", type=int, default=8)
a = p.parse_args()
G = "/export/home/kaziz/g05"; C = f"{G}/results/feat_cache"; DS = f"{G}/data/libero_merged_g05grip"
os.makedirs(f"{G}/results/robobrain", exist_ok=True)
t0 = time.time()
m = np.load(f"{C}/meta.npz"); ep, fr, L = m["episode_index"], m["frame_index"], m["ep_length"]
d = pd.concat([pd.read_parquet(f, columns=["index", "task_index"]) for f in sorted(glob.glob(f"{DS}/data/*/*.parquet"))]).sort_values("index")
tidx = d["task_index"].values
tp = pd.read_parquet(f"{DS}/meta/tasks.parquet").reset_index()
tcol = [c for c in tp.columns if tp[c].dtype == object][0]
norm = lambda s: " ".join(str(s).lower().replace("_", " ").split())  # noqa: E731
from libero.libero import benchmark  # noqa: E402

langs = [t.language for t in benchmark.get_benchmark_dict()["libero_goal"]().tasks]
name2idx = {norm(r[tcol]): int(r["task_index"]) for _, r in tp.iterrows()}
goal_ids = [name2idx[norm(x)] for x in langs]
T = importlib.import_module("07_train_hybrid")
reader = T.FrameReader({"episode_index": torch.from_numpy(ep), "frame_index": torch.from_numpy(fr)})
starts = np.nonzero(fr == 0)[0]
items = []  # (start_row, length, goal-task position)
for s in starts:
    if ep[s] % 5 == 0 and tidx[s] in goal_ids and L[s] > 45:
        items.append((s, L[s], goal_ids.index(tidx[s])))
print(f"{len(items)} held-out LIBERO-Goal demos ({time.time() - t0:.0f}s)", flush=True)
rb = RoboBrain(f"{G}/models/{a.model}", device="cuda:0")
print(f"loaded {a.model} dtype={rb.dtype} ({time.time() - t0:.0f}s)", flush=True)


def frames(rows):
    return [x for x in reader.frames(torch.from_numpy(np.asarray(rows))).permute(0, 2, 3, 1).numpy()]


res = {"model": a.model, "n": len(items)}
for cond in ("task_multi", "task_current", "goal_choice"):
    correct, unparsed, lat = [], 0, []
    for b in range(0, len(items), a.batch):
        chunk = items[b:b + a.batch]; batch = []
        for s, Ln, k in chunk:
            if cond == "task_multi":
                ims = frames([s + f for f in (0, 10, 20, 30, 40)]); q = task_choice_prompt(langs, len(ims), "multi")
            elif cond == "task_current":
                ims = frames([s + 40]); q = task_choice_prompt(langs, 1, "current")
            else:
                other = [x for x in items if x[2] == k and x[0] != s][0]
                ims = frames([s, other[0] + other[1] - 1]); q = goal_choice_prompt(langs)
            batch.append((ims, q))
        t1 = time.time(); ans = rb.ask(batch); lat.append((time.time() - t1) / len(batch))
        for (s, Ln, k), txt in zip(chunk, ans):
            v = first_int(txt, 1, len(langs))
            unparsed += v is None
            correct.append(v is not None and v - 1 == k)
        if b == 0:
            print(cond, "example answers:", [x[:40] for x in ans[:3]], flush=True)
    res[cond] = {"acc": float(np.mean(correct)), "unparsed": unparsed, "sec_per_query": float(np.mean(lat))}
    print(json.dumps({cond: res[cond]}), flush=True)
json.dump(res, open(f"{G}/results/robobrain/offline_{a.model}.json", "w"), indent=2)
print("done", f"{time.time() - t0:.0f}s")
