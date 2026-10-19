"""WM-MEM: does episodic memory help the small latent world model? (non-Markovian vs Markovian dynamics)

  Markovian (23_train_wm.py):  u_{t+k} = u_t + f(u_t, a_{t:t+k})
  Memory (this script):        u_{t+k} = u_t + f(u_t, u_{t-o_1}, ..., u_{t-o_H}, a_{t:t+k})
History frames are pooled to 4x4 tokens (16 per frame) with a learned per-offset embedding; offsets are multiples of k
so multi-step rollouts can take history from earlier predictions. Frames before the episode start are clamped to the
first frame (what an online memory would hold). Same data, held-out split, eval rows, steps and seed as 23_train_wm.py,
so wm_eval.json is directly comparable with the Markovian run (wm_base8k).

Extra evaluation: `hist_ablated` re-runs the rollout with every history frame replaced by the current frame, which
shows whether the model actually reads its memory.
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
G = os.environ.get("G05_ROOT", "/export/home/kaziz/g05")
C = f"{G}/results/feat_cache"

p = argparse.ArgumentParser()
p.add_argument("--steps", type=int, default=8000)
p.add_argument("--batch", type=int, default=128)
p.add_argument("--lr", type=float, default=3e-4)
p.add_argument("--width", type=int, default=384)
p.add_argument("--depth", type=int, default=8)
p.add_argument("--k", type=int, default=8)
p.add_argument("--hist", default="8,16", help="history offsets in frames (multiples of k); '' = Markovian")
p.add_argument("--eval_n", type=int, default=2048)
p.add_argument("--out", default=f"{G}/results/wm_mem")
p.add_argument("--seed", type=int, default=0)
a = p.parse_args()
os.makedirs(a.out, exist_ok=True)
HIST = [] if a.hist in ("", "none") else [int(x) for x in a.hist.split(",") if x.strip()]
assert all(o % a.k == 0 and o > 0 for o in HIST), "history offsets must be positive multiples of k"
torch.manual_seed(a.seed); np.random.seed(a.seed)
dev = torch.device("cuda")
BF16 = torch.cuda.get_device_capability()[0] >= 8
AMP = (lambda: torch.autocast("cuda", dtype=torch.bfloat16)) if BF16 else (lambda: torch.autocast("cuda", dtype=torch.float16))
scaler = torch.cuda.amp.GradScaler(enabled=not BF16)
t0 = time.time()
logf = open(f"{a.out}/log.jsonl", "a")


def log(d):
    d["t"] = round(time.time() - t0, 1)
    print(json.dumps(d), flush=True); logf.write(json.dumps(d) + "\n"); logf.flush()


meta = np.load(f"{C}/meta.npz")
feats = np.ascontiguousarray(np.load(f"{C}/feats.npy", mmap_mode="r"))
ep, fr, L = meta["episode_index"], meta["frame_index"], meta["ep_length"]
act = meta["action"].astype(np.float32)
a_mean, a_std = act.mean(0), act.std(0) + 1e-6
act_n = (act - a_mean) / a_std
held = ep % 20 == 0
remaining = L - 1 - fr


def valid_rows(steps, heldout):
    return np.nonzero((remaining >= steps * a.k) & (held == heldout))[0]


train_rows = {s: valid_rows(s, False) for s in (1, 2)}
f_mean = torch.tensor(np.asarray(feats[np.random.choice(train_rows[1], 4096)], dtype=np.float32).mean((0, 1)), device=dev)
f_std = torch.tensor(np.asarray(feats[np.random.choice(train_rows[1], 4096)], dtype=np.float32).std((0, 1)) + 1e-4, device=dev)
norm = lambda x: (x - f_mean) / f_std  # noqa: E731


def get_feats(rows):
    return (torch.from_numpy(np.asarray(feats[np.sort(rows)], dtype=np.float32))[np.argsort(np.argsort(rows))]).to(dev)


def real_at(rows, off):  # row index of frame t+off, clamped to the episode start
    return rows + np.maximum(off, -fr[rows])


def get_chunk(rows, offset=0):
    idx = rows[:, None] + offset + np.arange(a.k)[None]
    return torch.from_numpy(act_n[idx].reshape(len(rows), -1)).to(dev)


class WM(nn.Module):
    def __init__(self, width, depth, k, n_hist):
        super().__init__()
        self.inp = nn.Linear(768, width)
        self.pos = nn.Parameter(torch.randn(64, width) * 0.02)
        self.act = nn.Sequential(nn.Linear(7 * k, width * 4), nn.GELU(), nn.Linear(width * 4, width * 4))
        self.act_pos = nn.Parameter(torch.randn(4, width) * 0.02)
        self.n_hist = n_hist
        if n_hist:
            self.hist_inp = nn.Linear(768, width)
            self.hist_pos = nn.Parameter(torch.randn(n_hist, 16, width) * 0.02)
        layer = nn.TransformerEncoderLayer(width, max(4, width // 64), width * 4, dropout=0.0, batch_first=True, norm_first=True)
        self.tf = nn.TransformerEncoder(layer, depth)
        self.norm = nn.LayerNorm(width)
        self.out = nn.Linear(width, 768)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)

    def forward(self, u, chunk, hist=()):  # u [B,64,768] normalized; hist: n_hist x [B,64,768] normalized
        b = u.shape[0]
        toks = [self.inp(u) + self.pos]
        for j, h in enumerate(hist):
            hp = F.avg_pool2d(h.view(b, 8, 8, 768).permute(0, 3, 1, 2), 2).permute(0, 2, 3, 1).reshape(b, 16, 768)
            toks.append(self.hist_inp(hp) + self.hist_pos[j])
        toks.append(self.act(chunk).view(b, 4, -1) + self.act_pos)
        h = self.tf(torch.cat(toks, 1))[:, :64]
        return u + self.out(self.norm(h))


def rollout(model, rows, chunk_rows, n_steps, ablate=False, start=None):
    """Roll out n_steps chunks from frame t (rows). History of step s comes from real frames (time <= t) or earlier
    predictions. chunk_rows: rows whose action chunks are executed. Returns the list of predictions."""
    cur = norm(get_feats(rows)) if start is None else start
    timeline = {0: cur}
    real_cache = {}
    preds = []
    for s in range(n_steps):
        hist = []
        for o in HIST:
            j = s - o // a.k
            if ablate:
                hist.append(timeline[s])
            elif j > 0:
                hist.append(timeline[j])
            else:
                if j not in real_cache:
                    real_cache[j] = norm(get_feats(real_at(rows, j * a.k))) if j < 0 else timeline[0]
                hist.append(real_cache[j])
        nxt = model(timeline[s], get_chunk(chunk_rows, s * a.k), hist).float()
        timeline[s + 1] = nxt
        preds.append(nxt)
    return preds


model = WM(a.width, a.depth, a.k, len(HIST)).to(dev)
n_params = sum(p_.numel() for p_ in model.parameters())
opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01, betas=(0.9, 0.95))
log({"event": "setup", "gpu": torch.cuda.get_device_name(0), "bf16": BF16, "n_params": n_params, "hist": HIST,
     "train_rows": int(len(train_rows[1])), "args": vars(a)})
for step in range(1, a.steps + 1):
    warm = 500
    lr = a.lr * min(1, step / warm) * 0.5 * (1 + math.cos(math.pi * min(1, max(0, step - warm) / (a.steps - warm))))
    for g in opt.param_groups:
        g["lr"] = lr
    rows = np.random.choice(train_rows[2], a.batch)
    u1, u2 = (norm(get_feats(rows + s * a.k)) for s in (1, 2))
    with AMP():
        p1, p2 = rollout(model, rows, rows, 2)
    loss1 = F.mse_loss(p1.float(), u1)
    loss2 = F.mse_loss(p2.float(), u2)
    loss = loss1 + 0.5 * loss2
    opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.unscale_(opt)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); scaler.step(opt); scaler.update()
    if step % 250 == 0 or step == 1:
        log({"event": "train", "step": step, "loss1": round(loss1.item(), 5), "loss2": round(loss2.item(), 5), "lr": lr})
torch.save({"model": model.state_dict(), "args": vars(a), "f_mean": f_mean.cpu(), "f_std": f_std.cpu(),
            "a_mean": a_mean, "a_std": a_std}, f"{a.out}/wm.pt")

# ---------------- evaluation (same rows as 23_train_wm.py: RandomState(1), valid_rows(4, held-out)) ----------------
model.eval()
res = {"n_params_wm": n_params, "hist": HIST}
rng = np.random.RandomState(1)
ev = rng.choice(valid_rows(4, True), a.eval_n, replace=False)
with torch.no_grad(), AMP():
    persist = norm(get_feats(ev))
    for tag, abl in (("", False), ("hist_ablated_", True)):
        if abl and not HIST:
            continue
        preds = rollout(model, ev, ev, 4, ablate=abl)
        for s in (1, 2, 4):
            tgt = norm(get_feats(ev + s * a.k))
            e_pred = ((preds[s - 1] - tgt) ** 2).mean((1, 2)).cpu().numpy()
            e_pers = ((persist - tgt) ** 2).mean((1, 2)).cpu().numpy()
            moving = e_pers >= np.median(e_pers)
            res[f"{tag}step{s}"] = {"horizon_s": s * a.k / 20, "mse_pred": float(e_pred.mean()), "mse_persist": float(e_pers.mean()),
                                    "ratio": float(e_pred.mean() / e_pers.mean()),
                                    "ratio_moving": float(e_pred[moving].mean() / e_pers[moving].mean()),
                                    "frac_beats_persist": float((e_pred < e_pers).mean())}
            if not abl:
                np.save(f"{a.out}/err_step{s}.npy", e_pred)  # per-row errors for paired comparison with the Markovian run
    # action sensitivity (1 step)
    tgt = norm(get_feats(ev + a.k))
    perm = np.roll(np.arange(len(ev)), 1)
    e_ok = ((rollout(model, ev, ev, 1)[0] - tgt) ** 2).mean((1, 2))
    e_bad = ((rollout(model, ev, ev[perm], 1)[0] - tgt) ** 2).mean((1, 2))
    res["action_sensitivity"] = {"mse_correct": e_ok.mean().item(), "mse_permuted": e_bad.mean().item(),
                                 "ratio_permuted_over_correct": (e_bad.mean() / e_ok.mean()).item(),
                                 "frac_correct_better": (e_ok < e_bad).float().mean().item()}
    # planning signal, random-episode negatives (as 23_train_wm.py)
    K = 4
    g_feat = norm(get_feats(ev + (L[ev] - 1 - fr[ev])))
    cand_rows = [ev] + [rng.choice(valid_rows(4, True), len(ev)) for _ in range(K - 1)]
    scores = torch.stack([((rollout(model, ev, cr, 4)[-1] - g_feat) ** 2).mean((1, 2)) for cr in cand_rows], 1)
    res["planning_rank"] = {"K": K, "horizon_s": 4 * a.k / 20, "expert_top1": (scores.argmin(1) == 0).float().mean().item(), "chance": 0.25}
    # hard negatives: same episode, chunks at -1.2 / +1.2 / +2.4 s
    offs = np.array([-24, 24, 48])
    ok = (fr[ev] + offs.min() >= 0) & (remaining[ev] >= offs.max() + 4 * a.k)
    evh = ev[ok]
    for gname, grow in (("goal_final_frame", evh + (L[evh] - 1 - fr[evh])), ("goal_subgoal_1.6s", evh + 4 * a.k)):
        g_feat = norm(get_feats(grow))
        sc = torch.stack([((rollout(model, evh, cr, 4)[-1] - g_feat) ** 2).mean((1, 2)) for cr in [evh] + [evh + o for o in offs]], 1)
        res[f"planning_rank_hard_{gname}"] = {"K": 4, "n": int(len(evh)), "expert_top1": (sc.argmin(1) == 0).float().mean().item(), "chance": 0.25}
res["held_out_rows"] = int(len(ev))
np.save(f"{a.out}/eval_rows.npy", ev)
json.dump(res, open(f"{a.out}/wm_eval.json", "w"), indent=2)
log({"event": "eval", **res})
