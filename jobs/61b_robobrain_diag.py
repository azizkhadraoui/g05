"""Why is RoboBrain at chance on LIBERO-Goal task choice? Prompt-format / resolution / precision diagnostics (30 demos).

  describe        free description of one frame (does it perceive the scene?)
  numbers         numbered options (as in 61)
  letters         lettered options A-J
  open            "which task?" free text -> mapped to the nearest command by word overlap
each at 256 px and 448 px; precision = model dtype argument (fp16 on T4, or bf16 emulated / fp32 if requested)
"""
import argparse, glob, importlib, json, os, re, sys, time
import numpy as np, pandas as pd, torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import t4_fp16  # noqa: F401,E402
from robobrain import RoboBrain, first_int  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--model", required=True)
p.add_argument("--dtype", default="auto", choices=["auto", "fp32", "bf16"])
p.add_argument("--n", type=int, default=30)
a = p.parse_args()
G = "/export/home/kaziz/g05"; C = f"{G}/results/feat_cache"; DS = f"{G}/data/libero_merged_g05grip"
m = np.load(f"{C}/meta.npz"); ep, fr, L = m["episode_index"], m["frame_index"], m["ep_length"]
d = pd.concat([pd.read_parquet(f, columns=["index", "task_index"]) for f in sorted(glob.glob(f"{DS}/data/*/*.parquet"))]).sort_values("index")
tidx = d["task_index"].values
tp = pd.read_parquet(f"{DS}/meta/tasks.parquet").reset_index(); tcol = [c for c in tp.columns if tp[c].dtype == object][0]
norm = lambda s: " ".join(str(s).lower().replace("_", " ").split())  # noqa: E731
from libero.libero import benchmark  # noqa: E402
langs = [t.language for t in benchmark.get_benchmark_dict()["libero_goal"]().tasks]
name2idx = {norm(r[tcol]): int(r["task_index"]) for _, r in tp.iterrows()}; goal_ids = [name2idx[norm(x)] for x in langs]
T = importlib.import_module("07_train_hybrid")
reader = T.FrameReader({"episode_index": torch.from_numpy(ep), "frame_index": torch.from_numpy(fr)})
starts = [s for s in np.nonzero(fr == 0)[0] if ep[s] % 5 == 0 and tidx[s] in goal_ids and L[s] > 80]
rng = np.random.RandomState(0); starts = list(rng.choice(starts, min(a.n, len(starts)), replace=False))
rb = RoboBrain(f"{G}/models/{a.model}", device="cuda:0")
if a.dtype != "auto":
    rb.dtype = torch.float32 if a.dtype == "fp32" else torch.bfloat16
    rb.model.to(rb.dtype)
print(f"model {a.model} dtype {rb.dtype}", flush=True)
frames = lambda rows: [x for x in reader.frames(torch.tensor(rows)).permute(0, 2, 3, 1).numpy()]  # noqa: E731
L_ = "ABCDEFGHIJ"
words = lambda s: set(re.findall(r"[a-z]+", s.lower())) - {"the", "a", "of", "on", "in", "to", "and", "it", "put"}  # noqa: E731
res = {}
ims0 = frames([starts[0] + 60])
for side in (256, 448):
    rb.max_side = side
    print(f"[{side}px] describe:", rb.ask([(ims0, "Describe what you see in this image in one sentence.")], max_new_tokens=60)[0][:200], flush=True)
    for fmt in ("numbers", "letters", "open"):
        correct = []; answers = []
        for s in starts:
            k = langs.index([x for x in langs if name2idx[norm(x)] == tidx[s]][0])
            ims = frames([s + 60])  # 3 s in: the arm is clearly heading to the target
            if fmt == "numbers":
                q = "Which task is the robot arm performing?\n" + "\n".join(f"{j + 1}. {c}" for j, c in enumerate(langs)) + "\nAnswer with the number only."
            elif fmt == "letters":
                q = "Which task is the robot arm performing?\n" + "\n".join(f"{L_[j]}. {c}" for j, c in enumerate(langs)) + "\nAnswer with the letter only."
            else:
                q = "What task is the robot arm performing? Answer with one short instruction."
            ans = rb.ask([(ims, q)], max_new_tokens=24)[0]; answers.append(ans[:40])
            if fmt == "numbers":
                v = first_int(ans, 1, 10); pred = v - 1 if v else None
            elif fmt == "letters":
                mm = re.search(r"\b([A-J])\b", ans); pred = L_.index(mm.group(1)) if mm else None
            else:
                ov = [len(words(ans) & words(c)) for c in langs]; pred = int(np.argmax(ov)) if max(ov) > 0 else None
            correct.append(pred == k)
        res[f"{side}px_{fmt}"] = float(np.mean(correct))
        print(f"[{side}px] {fmt}: acc {np.mean(correct):.2f}  answers {answers[:5]}", flush=True)
json.dump(res, open(f"{G}/results/robobrain/diag_{a.model}_{a.dtype}.json", "w"), indent=2)
print(json.dumps(res))
