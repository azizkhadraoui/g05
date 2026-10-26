"""Instruction-conditioned latent subgoal generator: the cheap stand-in for pi0.7's 14B subgoal-image model.

  g_t = u_t + f(u_t, instruction)       u = pooled 8x8 DINOv3 tokens (64 x 768), g_t ~ u_{t+k}, k = 32 frames (1.6 s)
The instruction enters as a learned embedding per LIBERO task (4 tokens); the target is clamped to the episode's
last frame, like the goal videos. Trained with MSE on cached features (22_cache_feats.py); episodes with
episode_index % 20 == 0 are held out.

Held-out evaluation (normalised MSE to the true u_{t+k}, lower is better):
  persistence u_t | demo goal video (mean of the task's training demos at the same time) | generator
  | generator with another LIBERO-Goal instruction (how much the instruction matters in the shared scene)
"""
import argparse
import glob
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

G = os.environ.get("G05_ROOT", "/export/home/kaziz/g05")
C = f"{G}/results/feat_cache"
DS = f"{G}/data/libero_merged_g05grip"


class SubgoalGen(nn.Module):
    def __init__(self, n_tasks, width=512, depth=8, n_task_tok=4):
        super().__init__()
        self.inp = nn.Linear(768, width)
        self.pos = nn.Parameter(torch.randn(64, width) * 0.02)
        self.task = nn.Embedding(n_tasks, n_task_tok * width)
        self.task_pos = nn.Parameter(torch.randn(n_task_tok, width) * 0.02)
        self.n_task_tok = n_task_tok
        layer = nn.TransformerEncoderLayer(width, max(4, width // 64), width * 4, dropout=0.0, batch_first=True, norm_first=True)
        self.tf = nn.TransformerEncoder(layer, depth)
        self.norm = nn.LayerNorm(width)
        self.out = nn.Linear(width, 768)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)

    def forward(self, u, task_ids):  # u normalised [B,64,768], task_ids [B] -> predicted normalised future
        x = self.inp(u) + self.pos
        tt = self.task(task_ids).view(u.shape[0], self.n_task_tok, -1) + self.task_pos
        h = self.tf(torch.cat([x, tt], 1))[:, :64]
        return u + self.out(self.norm(h))


class Generator:
    """Inference wrapper: raw pooled features in, raw pooled features out."""

    def __init__(self, ck, device):
        a = ck["args"]
        self.model = SubgoalGen(ck["n_tasks"], a["width"], a["depth"]).to(device).eval()
        self.model.load_state_dict(ck["model"])
        self.f_mean, self.f_std = ck["f_mean"].to(device), ck["f_std"].to(device)
        self.device, self.ahead = device, a["ahead"]

    @torch.no_grad()
    def predict(self, u_raw, task_ids):
        u = (u_raw.to(self.device).float() - self.f_mean) / self.f_std
        t = torch.as_tensor(task_ids, device=self.device, dtype=torch.long)
        with torch.autocast("cuda", dtype=torch.float16):
            g = self.model(u, t)
        return g.float() * self.f_std + self.f_mean


def load_generator(path, device="cuda"):
    return Generator(torch.load(path, map_location="cpu", weights_only=False), device)


def main():
    import pandas as pd

    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=10000)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--depth", type=int, default=8)
    p.add_argument("--ahead", type=int, default=32)
    p.add_argument("--eval_n", type=int, default=4096)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=f"{G}/results/subgoal_gen")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = torch.device("cuda")
    bf16 = torch.cuda.get_device_capability()[0] >= 8
    amp = (lambda: torch.autocast("cuda", dtype=torch.bfloat16)) if bf16 else (lambda: torch.autocast("cuda", dtype=torch.float16))
    scaler = torch.cuda.amp.GradScaler(enabled=not bf16)
    t0 = time.time()
    logf = open(f"{a.out}/log.jsonl", "a")

    def log(d):
        d["t"] = round(time.time() - t0, 1)
        print(json.dumps(d), flush=True); logf.write(json.dumps(d) + "\n"); logf.flush()

    meta = np.load(f"{C}/meta.npz")
    ep, fr, L = meta["episode_index"], meta["frame_index"], meta["ep_length"]
    d = pd.concat([pd.read_parquet(f, columns=["index", "task_index"]) for f in sorted(glob.glob(f"{DS}/data/*/*.parquet"))]).sort_values("index")
    tidx = d["task_index"].values.astype(np.int64)
    assert len(tidx) == len(ep)
    tp = pd.read_parquet(f"{DS}/meta/tasks.parquet").reset_index()
    tcol = [c for c in tp.columns if tp[c].dtype == object][0]
    task_names = {int(r["task_index"]): str(r[tcol]) for _, r in tp.iterrows()}
    n_tasks = int(tidx.max()) + 1
    t_load = time.time()
    feats = np.ascontiguousarray(np.load(f"{C}/feats.npy", mmap_mode="r"))  # 27 GB into RAM (NFS random reads are slow)
    log({"event": "loaded", "rows": len(ep), "n_tasks": n_tasks, "load_s": round(time.time() - t_load, 1)})
    held = ep % 20 == 0
    remaining = L - 1 - fr
    target = np.arange(len(ep)) + np.minimum(a.ahead, remaining)
    train_rows = np.nonzero(~held)[0]
    f_mean = torch.tensor(np.asarray(feats[np.random.choice(train_rows, 4096)], dtype=np.float32).mean((0, 1)), device=dev)
    f_std = torch.tensor(np.asarray(feats[np.random.choice(train_rows, 4096)], dtype=np.float32).std((0, 1)) + 1e-4, device=dev)
    norm = lambda x: (x - f_mean) / f_std  # noqa: E731

    def get(rows):
        return torch.from_numpy(np.asarray(feats[rows], dtype=np.float32)).to(dev)

    model = SubgoalGen(n_tasks, a.width, a.depth).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01, betas=(0.9, 0.95))
    log({"event": "setup", "n_params": sum(p_.numel() for p_ in model.parameters()), "train_rows": int(len(train_rows)),
         "gpu": torch.cuda.get_device_name(0), "args": vars(a)})
    for step in range(1, a.steps + 1):
        warm = 500
        lr = a.lr * min(1, step / warm) * 0.5 * (1 + math.cos(math.pi * min(1, max(0, step - warm) / (a.steps - warm))))
        for g in opt.param_groups:
            g["lr"] = lr
        rows = np.random.choice(train_rows, a.batch)
        u0, u1 = norm(get(rows)), norm(get(target[rows]))
        with amp():
            pred = model(u0, torch.from_numpy(tidx[rows]).to(dev))
        loss = F.mse_loss(pred.float(), u1)
        opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); scaler.step(opt); scaler.update()
        if step % 250 == 0 or step == 1:
            log({"event": "train", "step": step, "loss": round(loss.item(), 5), "lr": lr})
    torch.save({"model": model.state_dict(), "args": vars(a), "n_tasks": n_tasks, "task_names": task_names,
                "f_mean": f_mean.cpu(), "f_std": f_std.cpu()}, f"{a.out}/gen.pt")

    # ---------------- held-out evaluation ----------------
    model.eval()
    starts_by_task = {t: np.nonzero((tidx == t) & (fr == 0) & ~held)[0] for t in range(n_tasks)}
    ref_cache = {}

    def demo_ref(t, f_target):  # mean over the task's training demos of the frame at f_target (clamped per demo)
        key = (t, f_target)
        if key not in ref_cache:
            s = starts_by_task[t]
            ref_cache[key] = np.asarray(feats[s + np.minimum(f_target, L[s] - 1)], dtype=np.float32).mean(0)
        return ref_cache[key]

    goal_task_ids = sorted({t for t, n in task_names.items() if " ".join(n.lower().replace("_", " ").split()) in GOAL_SUITE})
    rng = np.random.RandomState(1)
    res = {"libero_goal_task_ids": goal_task_ids}
    for name, pool in (("all", np.nonzero(held)[0]), ("libero_goal", np.nonzero(held & np.isin(tidx, goal_task_ids))[0])):
        if len(pool) == 0:
            continue
        ev = rng.choice(pool, min(a.eval_n, len(pool)), replace=False)
        errs = {"persist": [], "demo_video": [], "gen": [], "gen_wrong_instr": []}
        for s in range(0, len(ev), 256):
            r = ev[s:s + 256]
            u0, u1 = norm(get(r)), norm(get(target[r]))
            vid = norm(torch.from_numpy(np.stack([demo_ref(int(tidx[i]), 8 * ((int(fr[i]) + a.ahead) // 8)) for i in r])).to(dev))
            if goal_task_ids:
                wrong = np.array([goal_task_ids[(goal_task_ids.index(int(tidx[i])) + 3) % len(goal_task_ids)]
                                  if int(tidx[i]) in goal_task_ids else (int(tidx[i]) + 1) % n_tasks for i in r])
            else:
                wrong = (tidx[r] + 1) % n_tasks
            with torch.no_grad(), amp():
                g = model(u0, torch.from_numpy(tidx[r]).to(dev)).float()
                gw = model(u0, torch.from_numpy(wrong).to(dev)).float()
            for k, x in (("persist", u0), ("demo_video", vid), ("gen", g), ("gen_wrong_instr", gw)):
                errs[k].append(((x - u1) ** 2).mean((1, 2)).cpu())
        e = {k: torch.cat(v) for k, v in errs.items()}
        res[name] = {"n": int(len(ev)), **{f"mse_{k}": v.mean().item() for k, v in e.items()},
                     "gen_over_persist": (e["gen"].mean() / e["persist"].mean()).item(),
                     "demo_video_over_persist": (e["demo_video"].mean() / e["persist"].mean()).item(),
                     "gen_over_demo_video": (e["gen"].mean() / e["demo_video"].mean()).item(),
                     "wrong_instr_over_gen": (e["gen_wrong_instr"].mean() / e["gen"].mean()).item(),
                     "frac_gen_beats_demo_video": (e["gen"] < e["demo_video"]).float().mean().item()}
    json.dump(res, open(f"{a.out}/gen_eval.json", "w"), indent=2)
    log({"event": "eval", **res})


GOAL_SUITE = {  # LIBERO-Goal task languages (shared scene, 10 commands), as stored in the dataset's tasks.parquet
    "open the middle drawer of the cabinet", "put the bowl on the stove", "put the wine bottle on top of the cabinet",
    "open the top drawer and put the bowl inside", "put the bowl on top of the cabinet", "push the plate to the front of the stove",
    "put the cream cheese in the bowl", "turn on the stove", "put the bowl on the plate", "put the wine bottle on the rack"}

if __name__ == "__main__":
    main()
