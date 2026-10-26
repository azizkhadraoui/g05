"""Command-level world model for picture goals: predict the scene at the END of a command from the scene now,
f_end = g(f_now, command), so a goal picture can be read by imagining each command's outcome and keeping the closest.

Why: the chunk-level world model (23_train_wm.py) imagines 1.6 s, too short to tell "turn on the stove" from "put the
bowl on the stove"; in missions it picked the picture's command only 12-15% of the replans (chance 10%).
Data: cached DINOv3 features of the LIBERO-Goal training demos (results/feat_cache, the frames G0.5 was trained on),
pooled to a 2x2 grid + mean (5 x 768). Pairs: an early frame (first 30% of a demo) -> that demo's final frame.
The first demo of each task is held out entirely: its last frame is the goal picture the mission benchmark shows
(125_e2e_missions.py goal_picture), so the model never saw the pictures it is tested with.
Offline check: held-out demos' first frame + the mission picture -> argmin over the 10 commands of the predicted
end-state distance (accuracy, chance 10%); also a "nearest end frame" baseline with no model (picture vs current
scene differences only through the identity map).
Output: results/endstate_wm/model.pt (read by 125_e2e_missions.py --picture wm_end)
"""
import glob
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

G = os.environ.get("G05_ROOT", "/export/home/kaziz/g05")
C = f"{G}/results/feat_cache"; DS = f"{G}/data/libero_merged_g05grip"; OUT = f"{G}/results/endstate_wm"
TASKS = ["open the middle drawer of the cabinet", "put the bowl on the stove", "put the wine bottle on top of the cabinet",
         "open the top drawer and put the bowl inside", "put the bowl on top of the cabinet", "push the plate to the front of the stove",
         "put the cream cheese in the bowl", "turn on the stove", "put the bowl on the plate", "put the wine bottle on the rack"]
nrm = lambda s: " ".join(str(s).lower().replace("_", " ").split())  # noqa: E731


def pool(f):  # [..., 64, 768] (8x8 grid) -> [..., 5 * 768]: 2x2 grid + mean
    f = torch.as_tensor(np.asarray(f, np.float32))
    g = f.reshape(*f.shape[:-2], 8, 8, 768)
    q = g.reshape(*f.shape[:-2], 2, 4, 2, 4, 768).mean((-4, -2)).reshape(*f.shape[:-2], 4 * 768)
    return torch.cat([q, f.mean(-2)], -1)


class EndState(nn.Module):
    def __init__(self, n_cmd=10, d=5 * 768, w=1024):
        super().__init__()
        self.e = nn.Embedding(n_cmd, 128)
        self.net = nn.Sequential(nn.Linear(d + 128, w), nn.GELU(), nn.Linear(w, w), nn.GELU(), nn.Linear(w, d))

    def forward(self, x, c):  # normalised pooled features now, command index -> normalised end-state features
        return x + self.net(torch.cat([x, self.e(c)], -1))


def score(model, x_now, x_goal):  # [d], [d] (normalised) -> distance per command [n_cmd]
    with torch.no_grad():
        c = torch.arange(model.e.num_embeddings, device=x_now.device)
        return ((model(x_now[None].expand(len(c), -1), c) - x_goal[None]) ** 2).mean(-1)


def main():
    os.makedirs(OUT, exist_ok=True)
    m = np.load(f"{C}/meta.npz"); fc = np.load(f"{C}/feats.npy", mmap_mode="r"); L = m["ep_length"]; fr0 = m["frame_index"]
    tidx = pd.concat([pd.read_parquet(f, columns=["index", "task_index"]) for f in sorted(glob.glob(f"{DS}/data/*/*.parquet"))]) \
        .sort_values("index")["task_index"].values
    tp = pd.read_parquet(f"{DS}/meta/tasks.parquet").reset_index(); tcol = [c for c in tp.columns if tp[c].dtype == object][0]
    name2idx = {nrm(r[tcol]): int(r["task_index"]) for _, r in tp.iterrows()}
    eps = {}  # command -> list of episode start rows (first = the mission's picture demo)
    for c, t in enumerate(TASKS):
        rows = np.nonzero(tidx == name2idx[nrm(t)])[0]
        eps[c] = list(rows[fr0[rows] == 0])
    rng = np.random.default_rng(0)
    split = {c: (s[0], [s[i] for i in rng.permutation(np.arange(1, len(s)))]) for c, s in eps.items()}  # (picture demo, rest)
    tr, va = [], []
    for c, (_, rest) in split.items():
        k = max(2, len(rest) // 10)
        va += [(c, s) for s in rest[:k]]; tr += [(c, s) for s in rest[k:]]
    print({c: len(v[1]) for c, v in split.items()}, "train demos", len(tr), "val demos", len(va), flush=True)

    def frames(c_s, early=0.3, step=4):
        X, Y, Cc = [], [], []
        for c, s in c_s:
            n = int(L[s]); ts = list(range(0, max(1, int(early * n)), step))
            X.append(pool(fc[s + np.array(ts)])); Y.append(pool(fc[s + n - 1])[None].expand(len(ts), -1)); Cc += [c] * len(ts)
        return torch.cat(X), torch.cat(Y), torch.tensor(Cc)
    Xtr, Ytr, Ctr = frames(tr)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-3
    N = lambda x: (x - mu) / sd  # noqa: E731
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    Xtr, Ytr, Ctr = N(Xtr).to(dev), N(Ytr).to(dev), Ctr.to(dev)
    model = EndState().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)
    E = 200; sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, E)
    pic = {c: N(pool(fc[split[c][0] + int(L[split[c][0]]) - 1])).to(dev) for c in eps}  # mission pictures

    def evaluate():
        model.eval(); hit = hit0 = hit_id = n = 0
        for c, s in va:
            x0 = N(pool(fc[s])).to(dev)
            d = score(model, x0, pic[c]); hit += int(d.argmin()) == c
            others = [pic[k] for k in eps]  # no model: the picture closest to the current scene (identity "world model")
            hit_id += int(torch.stack([((x0 - p) ** 2).mean() for p in others]).argmin()) == c
            xe = N(pool(fc[s + int(L[s]) - 1])).to(dev)  # sanity: own end frame as the goal (same layout)
            hit0 += int(score(model, x0, xe).argmin()) == c; n += 1
        model.train()
        return hit / n, hit0 / n, hit_id / n
    for ep in range(E):
        for b in torch.randperm(len(Xtr), device=dev).split(256):
            loss = F.mse_loss(model(Xtr[b], Ctr[b]), Ytr[b])
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        if ep % 25 == 0 or ep == E - 1:
            a, a0, aid = evaluate()
            print(f"epoch {ep}: train mse {loss.item():.4f} | held-out demos: picture-goal accuracy {a:.3f} "
                  f"(own end frame {a0:.3f}; nearest picture, no model {aid:.3f}; chance 0.100)", flush=True)
    torch.save({"model": model.state_dict(), "mu": mu, "sd": sd, "tasks": TASKS}, f"{OUT}/model.pt")
    print("saved", f"{OUT}/model.pt", flush=True)


if __name__ == "__main__":
    main()
