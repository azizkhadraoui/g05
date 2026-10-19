"""Small JEPA-style action-conditioned latent world model on LIBERO (DINOv3 feature space).

  u_{t+k} = u_t + f(u_t, a_{t:t+k})        k = 8 frames (0.4 s), u = pooled 8x8 DINOv3 tokens (64 x 768)
Trained with a 1-step + 2-step (rollout) MSE loss on cached features (22_cache_feats.py); episodes with
episode_index % 20 == 0 are held out.

Evaluation (held-out):
  * 1/2/4-step rollout MSE vs persistence (u_t)
  * LaWAM's 230M LaWM with the privileged posterior latent (W00 oracle), pooled to the same 8x8 grid
  * action sensitivity: correct vs permuted action chunks
  * offline planning signal: among K=4 candidate action chunks (expert + 3 from other episodes of the same task
    suite), pick the one whose 4-step predicted future is closest to the episode's final frame; chance = 25%
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
p.add_argument("--steps", type=int, default=20000)
p.add_argument("--batch", type=int, default=256)
p.add_argument("--lr", type=float, default=3e-4)
p.add_argument("--width", type=int, default=384)
p.add_argument("--depth", type=int, default=8)
p.add_argument("--k", type=int, default=8)
p.add_argument("--eval_n", type=int, default=2048)
p.add_argument("--out", default=f"{G}/results/wm_s0")
p.add_argument("--seed", type=int, default=0)
p.add_argument("--frac", type=float, default=1.0, help="fraction of training episodes (data-efficiency curve)")
p.add_argument("--no_multistep", action="store_true", help="1-step loss only (ablation of the 2-step rollout loss)")
p.add_argument("--skip_lawam", action="store_true")
p.add_argument("--eval_only", default=None, help="evaluate an existing wm.pt (no training)")
a = p.parse_args()
os.makedirs(a.out, exist_ok=True)
torch.manual_seed(a.seed); np.random.seed(a.seed)
dev = torch.device("cuda")
BF16 = torch.cuda.get_device_capability()[0] >= 8  # T4: bf16 is emulated (slow) -> fp16 autocast + GradScaler
AMP = (lambda: torch.autocast("cuda", dtype=torch.bfloat16)) if BF16 else (lambda: torch.autocast("cuda", dtype=torch.float16))
scaler = torch.cuda.amp.GradScaler(enabled=not BF16)
t0 = time.time()
logf = open(f"{a.out}/log.jsonl", "a")


def log(d):
    d["t"] = round(time.time() - t0, 1)
    print(json.dumps(d), flush=True); logf.write(json.dumps(d) + "\n"); logf.flush()


meta = np.load(f"{C}/meta.npz")
feats = np.load(f"{C}/feats.npy", mmap_mode="r")
feats = np.ascontiguousarray(feats)  # 27 GB into RAM: random memmap reads over NFS were the bottleneck
ep, fr, L = meta["episode_index"], meta["frame_index"], meta["ep_length"]
act = meta["action"].astype(np.float32)
a_mean, a_std = act.mean(0), act.std(0) + 1e-6
act_n = (act - a_mean) / a_std
N = len(ep)
held = ep % 20 == 0
remaining = L - 1 - fr  # frames left in the episode after t


def valid_rows(steps, heldout):
    return np.nonzero((remaining >= steps * a.k) & (held == heldout))[0]


train_rows = {s: valid_rows(s, False) for s in (1, 2)}
if a.frac < 1.0:
    tr_eps = np.unique(ep[train_rows[1]])
    keep = np.random.RandomState(123).choice(tr_eps, max(1, int(round(a.frac * len(tr_eps)))), replace=False)
    train_rows = {k: v[np.isin(ep[v], keep)] for k, v in train_rows.items()}
f_mean = torch.tensor(np.asarray(feats[np.random.choice(train_rows[1], 4096)], dtype=np.float32).mean((0, 1)), device=dev)
f_std = torch.tensor(np.asarray(feats[np.random.choice(train_rows[1], 4096)], dtype=np.float32).std((0, 1)) + 1e-4, device=dev)


def get_feats(rows):
    return (torch.from_numpy(np.asarray(feats[np.sort(rows)], dtype=np.float32))[np.argsort(np.argsort(rows))]).to(dev)


def get_chunk(rows, offset=0):
    idx = rows[:, None] + offset + np.arange(a.k)[None]
    return torch.from_numpy(act_n[idx].reshape(len(rows), -1)).to(dev)


class WM(nn.Module):
    def __init__(self, width, depth, k):
        super().__init__()
        self.inp = nn.Linear(768, width)
        self.pos = nn.Parameter(torch.randn(64, width) * 0.02)
        self.act = nn.Sequential(nn.Linear(7 * k, width * 4), nn.GELU(), nn.Linear(width * 4, width * 4))
        self.act_pos = nn.Parameter(torch.randn(4, width) * 0.02)
        layer = nn.TransformerEncoderLayer(width, max(4, width // 64), width * 4, dropout=0.0, batch_first=True, norm_first=True)
        self.tf = nn.TransformerEncoder(layer, depth)
        self.norm = nn.LayerNorm(width)
        self.out = nn.Linear(width, 768)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)

    def forward(self, u, chunk):  # u normalized [B,64,768], chunk [B,7k] -> next u normalized
        x = self.inp(u) + self.pos
        at = self.act(chunk).view(u.shape[0], 4, -1) + self.act_pos
        h = self.tf(torch.cat([x, at], 1))[:, :64]
        return u + self.out(self.norm(h))


if a.eval_only:
    ck = torch.load(a.eval_only, map_location="cpu", weights_only=False)
    a.width, a.depth, a.k = ck["args"]["width"], ck["args"]["depth"], ck["args"]["k"]
model = WM(a.width, a.depth, a.k).to(dev)
if a.eval_only:
    model.load_state_dict(ck["model"])
    f_mean, f_std = ck["f_mean"].to(dev), ck["f_std"].to(dev)
    a.steps = 0
n_params = sum(p_.numel() for p_ in model.parameters())
opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01, betas=(0.9, 0.95))
log({"event": "setup", "gpu": torch.cuda.get_device_name(0), "bf16": BF16, "arch_list": torch.cuda.get_arch_list(), "n_params": n_params, "train_rows": int(len(train_rows[1])), "args": vars(a)})
norm = lambda x: (x - f_mean) / f_std  # noqa: E731
denorm = lambda x: x * f_std + f_mean  # noqa: E731
for step in range(1, a.steps + 1):
    warm = 500
    lr = a.lr * min(1, step / warm) * 0.5 * (1 + math.cos(math.pi * min(1, max(0, step - warm) / (a.steps - warm))))
    for g in opt.param_groups:
        g["lr"] = lr
    rows = np.random.choice(train_rows[2], a.batch)
    u0, u1, u2 = (norm(get_feats(rows + s * a.k)) for s in (0, 1, 2))
    with AMP():
        p1 = model(u0, get_chunk(rows, 0))
        p2 = model(p1, get_chunk(rows, a.k))
    loss1 = F.mse_loss(p1.float(), u1)
    loss2 = F.mse_loss(p2.float(), u2)
    loss = loss1 + (0.0 if a.no_multistep else 0.5) * loss2
    opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.unscale_(opt)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); scaler.step(opt); scaler.update()
    if step % 250 == 0 or step == 1:
        log({"event": "train", "step": step, "loss1": round(loss1.item(), 5), "loss2": round(loss2.item(), 5), "lr": lr})
if not a.eval_only:
    torch.save({"model": model.state_dict(), "args": vars(a), "f_mean": f_mean.cpu(), "f_std": f_std.cpu(),
                "a_mean": a_mean, "a_std": a_std}, f"{a.out}/wm.pt")

# ---------------- evaluation ----------------
model.eval()
res = {"n_params_wm": n_params}
rng = np.random.RandomState(1)
ev = rng.choice(valid_rows(4, True), a.eval_n, replace=False)
with torch.no_grad(), AMP():
    u = norm(get_feats(ev))
    persist = u.clone()
    for s in (1, 2, 3, 4):
        u = model(u, get_chunk(ev, (s - 1) * a.k)).float()
        if s in (1, 2, 4):
            tgt = norm(get_feats(ev + s * a.k))
            e_pred = ((u - tgt) ** 2).mean((1, 2)).cpu().numpy()
            e_pers = ((persist - tgt) ** 2).mean((1, 2)).cpu().numpy()
            moving = e_pers >= np.median(e_pers)
            res[f"step{s}"] = {"horizon_s": s * a.k / 20, "mse_pred": float(e_pred.mean()), "mse_persist": float(e_pers.mean()),
                               "ratio": float(e_pred.mean() / e_pers.mean()),
                               "ratio_moving": float(e_pred[moving].mean() / e_pers[moving].mean()),
                               "frac_beats_persist": float((e_pred < e_pers).mean())}
    # action sensitivity (1 step)
    u0 = norm(get_feats(ev)); tgt = norm(get_feats(ev + a.k))
    perm = np.roll(np.arange(len(ev)), 1)
    e_ok = ((model(u0, get_chunk(ev)).float() - tgt) ** 2).mean((1, 2))
    e_bad = ((model(u0, get_chunk(ev)[perm]).float() - tgt) ** 2).mean((1, 2))
    res["action_sensitivity"] = {"mse_correct": e_ok.mean().item(), "mse_permuted": e_bad.mean().item(),
                                 "ratio_permuted_over_correct": (e_bad.mean() / e_ok.mean()).item(),
                                 "frac_correct_better": (e_ok < e_bad).float().mean().item()}
    # offline planning signal: K=4 candidates, goal = final frame of the episode (pooled features)
    K = 4
    goal_rows = ev + (L[ev] - 1 - fr[ev])
    g_feat = norm(get_feats(goal_rows))
    scores = []
    cand_rows = [ev] + [rng.choice(valid_rows(4, True), len(ev)) for _ in range(K - 1)]
    for cr in cand_rows:
        u = norm(get_feats(ev))
        for s in range(4):
            u = model(u, get_chunk(cr, s * a.k)).float()
        scores.append(((u - g_feat) ** 2).mean((1, 2)))
    scores = torch.stack(scores, 1)  # [n, K]
    res["planning_rank"] = {"K": K, "horizon_s": 4 * a.k / 20, "expert_top1": (scores.argmin(1) == 0).float().mean().item(),
                            "chance": 1 / K, "negatives": "chunks from random held-out episodes"}
    # HARD negatives: chunks from the SAME episode, 1.2 s before / 1.2 s after / 2.4 s after the current time
    offs = np.array([-24, 24, 48])
    ok = (fr[ev] + offs.min() >= 0) & (remaining[ev] >= offs.max() + 4 * a.k)
    evh = ev[ok]
    u_start = norm(get_feats(evh))
    for gname, grow in (("goal_final_frame", evh + (L[evh] - 1 - fr[evh])), ("goal_subgoal_1.6s", evh + 4 * a.k)):
        g_feat = norm(get_feats(grow))
        sc = []
        for cr in [evh] + [evh + o for o in offs]:
            u = u_start
            for s in range(4):
                u = model(u, get_chunk(cr, s * a.k)).float()
            sc.append(((u - g_feat) ** 2).mean((1, 2)))
        sc = torch.stack(sc, 1)
        res[f"planning_rank_hard_{gname}"] = {"K": 4, "n": int(len(evh)), "expert_top1": (sc.argmin(1) == 0).float().mean().item(),
                                              "chance": 0.25, "negatives": "same episode, chunks at -1.2 s / +1.2 s / +2.4 s"}
res["held_out_rows"] = int(len(ev))
json.dump(res, open(f"{a.out}/wm_eval.json", "w"), indent=2)
log({"event": "eval", **{k: v for k, v in res.items()}})


# ---------------- LaWAM comparison (pooled 8x8, privileged posterior latent) ----------------
try:
    if a.skip_lawam:
        raise RuntimeError("skipped (--skip_lawam)")
    import lawam_hybrid as H
    sys.path.insert(0, f"{G}/jobs")
    T = __import__("importlib").import_module("07_train_hybrid")
    lam = H.load_lawm()
    rowmap = {"episode_index": torch.from_numpy(ep), "frame_index": torch.from_numpy(fr)}
    reader = T.FrameReader(rowmap)
    sub = ev[:512]
    e_lawam, e_ours, e_pers = [], [], []
    for s in range(0, len(sub), 32):
        r = torch.from_numpy(sub[s:s + 32])
        f0 = reader.frames(r).cuda(); f1 = reader.frames(r + a.k).cuda()
        x = torch.stack([f0, f1], 1).float().div(255)
        x = H.imagenet_norm(x.flatten(0, 1)).view(len(r), 2, 3, 256, 256)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            out = lam.get_latent_action(videos=x, states=None, dec_videos=x, predict_future_frame=True)
            rec = out["recon"][0] if isinstance(out["recon"], tuple) else out["recon"]
        rec = rec.float().reshape(len(r), 16, 16, 768).permute(0, 3, 1, 2)
        rec = F.avg_pool2d(rec, 2).permute(0, 2, 3, 1).reshape(len(r), 64, 768)
        tgt = get_feats(sub[s:s + 32] + a.k); cur = get_feats(sub[s:s + 32])
        with torch.no_grad(), AMP():
            ours = denorm(model(norm(cur), get_chunk(sub[s:s + 32])).float())
        e_lawam.append(((norm(rec) - norm(tgt)) ** 2).mean((1, 2)).cpu())
        e_ours.append(((norm(ours) - norm(tgt)) ** 2).mean((1, 2)).cpu())
        e_pers.append(((norm(cur) - norm(tgt)) ** 2).mean((1, 2)).cpu())
    e_lawam, e_ours, e_pers = (torch.cat(x) for x in (e_lawam, e_ours, e_pers))
    res["vs_lawam"] = {"n": len(sub), "lawam_params": sum(p_.numel() for p_ in lam.decoder.parameters()) + sum(p_.numel() for p_ in lam.encoder.parameters()),
                       "ratio_ours": (e_ours.mean() / e_pers.mean()).item(), "ratio_lawam_oracle": (e_lawam.mean() / e_pers.mean()).item(),
                       "note": "LaWAM uses the privileged posterior latent (sees the future frame); ours uses the executed actions"}
    json.dump(res, open(f"{a.out}/wm_eval.json", "w"), indent=2)
    log({"event": "vs_lawam", **res["vs_lawam"]})
except Exception as e:  # keep the main results even if the comparison fails
    log({"event": "vs_lawam_failed", "error": repr(e)})
