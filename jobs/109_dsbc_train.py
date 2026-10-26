"""DSBC-style noise policy for frozen G0.5 (the Flow Reversal Steering paper's extension: a small model predicts noise
initialisations that make the frozen generalist succeed; the generalist's weights are untouched).

Data: requests recorded by 94_frs_serve.py (REC_DIR) during the 2 s after the command switch in the nudge condition
(frs_ref_rec): each holds the camera image, proprio state, command and the EXACT noise from which G0.5's own 10-step
sampler reproduces the executed chunk (fixed-point inversion). Labels: episode success from the run's result files.
Model: pooled DINOv3 features of the agent view (2x2 grid + mean, frozen) + proprio state + commanded task (one-hot)
-> noise [H, D] (padded action dims masked), trained on successful episodes only (success-filtered behaviour cloning in
noise space). At test time the server (frs_mode "dsbc") feeds this noise to the frozen sampler; no reference is used.
Usage: python 109_dsbc_train.py --rec_dir ... --results ... --out results/steer_frs/dsbc/policy.pt
"""
import argparse
import glob
import json
import os
import re
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

G = os.environ.get("G05_ROOT", "/export/home/kaziz/g05")
TASKS = ["open the middle drawer of the cabinet", "put the bowl on the stove", "put the wine bottle on top of the cabinet",
         "open the top drawer and put the bowl inside", "put the bowl on top of the cabinet", "push the plate to the front of the stove",
         "put the cream cheese in the bowl", "turn on the stove", "put the bowl on the plate", "put the wine bottle on the rack"]
FEAT = 5 * 768


class NoisePolicy(nn.Module):
    def __init__(self, H, D, sdim, width=1024, n_tasks=len(TASKS)):
        super().__init__()
        self.H, self.D = H, D
        self.f = nn.Sequential(nn.LayerNorm(FEAT), nn.Linear(FEAT, 512), nn.GELU())
        self.s = nn.Sequential(nn.Linear(sdim, 64), nn.GELU())
        self.t = nn.Embedding(n_tasks + 1, 64)  # last row: a command outside the vocabulary
        self.mlp = nn.Sequential(nn.Linear(512 + 64 + 64, width), nn.GELU(), nn.Dropout(0.1), nn.Linear(width, width), nn.GELU(),
                                 nn.Linear(width, H * D))

    def forward(self, f, s, t):
        return self.mlp(torch.cat([self.f(f), self.s(s), self.t(t)], -1)).view(-1, self.H, self.D)


def dino_fn(device):
    sys.path.insert(0, f"{G}/jobs"); sys.path.insert(0, f"{G}/GalaxeaVLA")
    import lawam_hybrid as Hy
    lam = Hy.load_lawm(device="cpu"); enc = lam.vision_encoder.to(device).eval()
    lam.vision_encoder = enc

    @torch.no_grad()
    def feats(img_chw_u8):  # [3, H, W] uint8 -> [5 * 768]: 2x2 pooled grid + global mean of DINOv3 patch features
        x = torch.as_tensor(np.asarray(img_chw_u8))[None].to(device)
        f = Hy.dino_from_uint8(lam, x).float()  # [1, 256, 768]
        g = f.view(1, 16, 16, 768).permute(0, 3, 1, 2)
        return torch.cat([F.adaptive_avg_pool2d(g, 2).flatten(2).transpose(1, 2).reshape(-1), f.mean(1)[0]]).cpu()
    return feats


def task_index(task, vocab):
    """Command -> embedding row (normalised text; len(vocab) = outside the vocabulary). Ints pass through (-1 = outside)."""
    if isinstance(task, (int, np.integer)):
        return int(task) if task >= 0 else len(vocab)
    task = " ".join(str(task).lower().split())
    return vocab.index(task) if task in vocab else len(vocab)


def load_policy(ckpt, device="cuda"):
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    vocab = ck.get("tasks", TASKS)  # LIBERO-Goal tasks, or the checkpoint's own commands (e.g. LIBERO-Mem steps, 140)
    m = NoisePolicy(ck["H"], ck["D"], ck["sdim"], n_tasks=len(vocab)).to(device).eval(); m.load_state_dict(ck["model"])
    smean, sstd = torch.tensor(ck["s_mean"]), torch.tensor(ck["s_std"])
    feats = dino_fn(device)

    @torch.no_grad()
    def pol(f, state, task):  # task: command text or vocabulary index
        s = (torch.as_tensor(state, dtype=torch.float32) - smean) / sstd
        t = torch.tensor([task_index(task, vocab)])
        z = m(f[None].to(device), s[None].to(device), t.to(device))[0].float().cpu()
        if ck.get("pad") is not None:
            z[:, torch.as_tensor(ck["pad"]).bool()] = 0.0
        return z
    return pol, feats


def fit_noise_policy(X, S, T, Z, E, pad, epochs, n_tasks, dev="cuda"):
    """Regress the recorded noise on (DINOv3 features, state, command) with an episode-level 10% validation split.
    X, Z: lists of tensors; S: list of state vectors; T: command indices; E: episode keys. Returns the checkpoint dict."""
    X, Z = torch.stack(X), torch.stack(Z)
    S = torch.tensor(np.stack(S), dtype=torch.float32); T = torch.tensor(T)
    smean, sstd = S.mean(0), S.std(0) + 1e-3
    S = (S - smean) / sstd
    H, D = Z.shape[1:]
    mask = torch.ones(D, dtype=torch.bool) if pad is None else ~torch.as_tensor(pad)
    eps = sorted(set(E)); rng = np.random.default_rng(0)
    val_eps = {eps[i] for i in rng.permutation(len(eps))[: max(1, len(eps) // 10)]}  # by index: keys mix str and int
    vi = torch.tensor([e in val_eps for e in E])
    print(f"samples {len(E)} from {len(eps)} episodes (val {int(vi.sum())}); z rms {Z[..., mask].pow(2).mean().sqrt():.3f}", flush=True)
    model = NoisePolicy(H, D, S.shape[1], n_tasks=n_tasks).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    Xd, Sd, Td, Zd = X.to(dev), S.to(dev), T.to(dev), Z.to(dev)
    tr = (~vi).nonzero().view(-1).to(dev); va = vi.nonzero().view(-1).to(dev)
    for ep in range(epochs):
        model.train()
        for b in tr[torch.randperm(len(tr), device=dev)].split(64):
            loss = (model(Xd[b], Sd[b], Td[b]) - Zd[b])[..., mask.to(dev)].pow(2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        if ep % 50 == 0 or ep == epochs - 1:
            model.eval()
            with torch.no_grad():
                vl = (model(Xd[va], Sd[va], Td[va]) - Zd[va])[..., mask.to(dev)].pow(2).mean().item() if len(va) else float("nan")
                base = Zd[va][..., mask.to(dev)].pow(2).mean().item() if len(va) else float("nan")
            print(f"epoch {ep}: train {loss.item():.4f}  val {vl:.4f}  (predicting zeros: {base:.4f})", flush=True)
    return {"model": model.state_dict(), "H": H, "D": D, "sdim": S.shape[1], "s_mean": smean.tolist(), "s_std": sstd.tolist(),
            "pad": None if pad is None else pad.tolist(), "n": len(E), "episodes": len(eps)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rec_dir", default=f"{G}/results/steer_frs/dsbc_records")
    ap.add_argument("--prefix", default="frs_ref_rec")
    ap.add_argument("--results", default=f"{G}/results/steer_frs/frs_screen")
    ap.add_argument("--out", default=f"{G}/results/steer_frs/dsbc/policy.pt")
    ap.add_argument("--sets", default="", help="several recording sets 'prefix:results_dir,...' (default: --prefix:--results)")
    ap.add_argument("--all", action="store_true", help="train on all episodes, not only successful ones (ablation)")
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--max_inv_err", type=float, default=0.1)
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    sets = [s.split(":", 1) for s in a.sets.split(",")] if a.sets else [(a.prefix, a.results)]
    succ, files = {}, []  # episodes keyed (prefix, target, trial): trial indices repeat across pair sets
    for prefix, results in sets:
        for f in glob.glob(f"{results}/{prefix}_t*/*_parallel_results.json"):
            t = int(re.search(r"_t(\d+)/", f).group(1))
            task = json.load(open(f))["tasks"][0]
            off = (task.get("frs") or {}).get("init_offset", 0)
            for k, v in (task.get("trial_successes") or {}).items():
                succ[(prefix, t, int(k) + off)] = bool(v)
        files += [(prefix, f) for f in sorted(glob.glob(f"{a.rec_dir}/{prefix}_t*_tr*_s*.npz"))]
    print(f"{len(files)} recorded requests from {len(sets)} set(s); {sum(succ.values())}/{len(succ)} successful episodes", flush=True)
    dev = "cuda"
    feats = dino_fn(dev)
    X, S, T, Z, E, pad, n_bad = [], [], [], [], [], None, 0
    for prefix, f in files:
        m = re.search(r"_t(\d+)_tr(\d+)_s(\d+)\.npz$", f); t, tr = int(m.group(1)), int(m.group(2))
        if (prefix, t, tr) not in succ or (not a.all and not succ[(prefix, t, tr)]):
            continue
        d = np.load(f, allow_pickle=True)
        if "inv_err" in d.files and float(d["inv_err"]) > a.max_inv_err:  # the noise must reproduce the executed chunk
            n_bad += 1
            continue
        X.append(feats(d["image"])); Z.append(torch.from_numpy(d["z"]).float())
        S.append(np.concatenate([d[k].reshape(-1) for k in sorted(k for k in d.files if k.startswith("state_"))]))
        T.append(task_index(str(d["task"]), TASKS))
        E.append((prefix, t, tr))
        if pad is None and d["pad"].size:
            pad = d["pad"].astype(bool)
    print(f"dropped {n_bad} records whose exact-inversion error exceeds {a.max_inv_err}", flush=True)
    ck = fit_noise_policy(X, S, T, Z, E, pad, a.epochs, len(TASKS), dev)
    torch.save({**ck, "all": a.all}, a.out)
    print("saved", a.out, flush=True)


if __name__ == "__main__":
    main()
