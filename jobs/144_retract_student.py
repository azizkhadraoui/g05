"""Retraction as the teacher: a vision-based student that decides how far to back off after a change of mind.

Teacher data (125_e2e_missions.py --steer home --grip place --retract frac --retract_frac f --save_change_obs, one run per
depth f in FRACS, same missions and start states): at the change of mind the robot sets down anything it holds, retracts
a fraction f of the scripted path to its rest pose (0 = release only, 1 = full retraction), then G0.5 gets the new
command. Each run saves what the robot sees and feels at that moment (agent + wrist cameras, end-effector pose, finger
state, old and new command) and whether the changed step was then completed, and how long it took.
Student: DINOv3 features of both cameras (2x2 grid + mean, frozen encoder) + proprio + commands -> one success logit per
depth. At run time (125 --retract learned) the robot takes the SHORTEST depth whose predicted success is within
--retract_margin of the best depth: full retraction's reliability where it is needed, less detour where it is not.
Offline check: 5-fold cross-validation grouped by situation (mission, start state); the chosen depth's real outcome is
read from that depth's run of the same situation.
Usage: python 144_retract_student.py --runs DIR0:0.0,DIR1:0.25,... --out results/retract_student/model.pt
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

G = os.environ.get("G05_ROOT", "/export/home/kaziz/g05")
FRACS = [0.0, 0.25, 0.5, 0.75, 1.0]
N_CMD = 10
VDIM = 2 * 5 * 768  # agent + wrist camera, 2x2 grid + mean of DINOv3 patch features
SDIM = 13


def dino_pooled(lam, H, img_hwc_u8, dev):
    x = torch.as_tensor(np.asarray(img_hwc_u8, np.uint8)).permute(2, 0, 1)[None].to(dev)
    with torch.no_grad():
        f = H.dino_from_uint8(lam, x).float()  # [1, 256, 768]
    g = f.view(1, 16, 16, 768).permute(0, 3, 1, 2)
    return torch.cat([F.adaptive_avg_pool2d(g, 2).flatten(2).transpose(1, 2).reshape(-1), f.mean(1)[0]]).cpu()


def proprio(o):
    pos, rest = np.asarray(o["eef_pos"], np.float32), np.asarray(o["rest_pos"], np.float32)
    return np.concatenate([pos, pos - rest, np.asarray(o["eef_quat"], np.float32), np.asarray(o["gripper_qpos"], np.float32)[:2],
                           [float(o["held"])]]).astype(np.float32)


class Student(nn.Module):
    def __init__(self, w=256):
        super().__init__()
        self.v = nn.Sequential(nn.LayerNorm(VDIM), nn.Linear(VDIM, w), nn.GELU(), nn.Dropout(0.2))
        self.s = nn.Sequential(nn.Linear(SDIM, 32), nn.GELU())
        self.head = nn.Sequential(nn.Linear(w + 32 + N_CMD + N_CMD + 1, w), nn.GELU(), nn.Dropout(0.2), nn.Linear(w, len(FRACS)))

    def forward(self, v, s, old, new):  # -> success logit per depth [B, len(FRACS)]
        oh = lambda i, n: F.one_hot(i.clamp(min=0), n).float() * (i >= 0).float()[:, None]  # noqa: E731
        return self.head(torch.cat([self.v(v), self.s(s), oh(old, N_CMD), oh(new, N_CMD), (new < 0).float()[:, None]], -1))


def load(path, dev, lam, H):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = Student().to(dev).eval(); m.load_state_dict(ck["model"])
    return dict(m=m, s_mean=torch.tensor(ck["s_mean"]), s_std=torch.tensor(ck["s_std"]), lam=lam, H=H, dev=dev)


def choose(rs, o, margin):
    """Shortest depth whose predicted success is within `margin` of the best depth's."""
    dev = rs["dev"]
    v = torch.cat([dino_pooled(rs["lam"], rs["H"], o["image"], dev), dino_pooled(rs["lam"], rs["H"], o["wrist"], dev)])[None].to(dev)
    s = ((torch.from_numpy(proprio(o)) - rs["s_mean"]) / rs["s_std"])[None].to(dev)
    with torch.no_grad():
        p = torch.sigmoid(rs["m"](v, s, torch.tensor([int(o["old_cmd"])], device=dev), torch.tensor([int(o["new_cmd"])], device=dev)))[0].cpu().numpy()
    i = next(i for i in range(len(FRACS)) if p[i] >= p.max() - margin)
    return FRACS[i], p


def load_runs(spec):
    """(situation, depth) -> outcome, time, change observation, for every run directory."""
    rows = []
    for item in spec.split(","):
        d, f = item.rsplit(":", 1); f = float(f)
        for e in map(json.loads, open(f"{d}/episodes.jsonl")):
            if e.get("total_at_change") is None:  # step 1 never finished: no change of mind happened
                continue
            npz = f"{d}/change_obs/m{e['mission']}_i{e['init']}.npz"
            if not os.path.exists(npz):
                continue
            done2 = next((x for x in e["steps_done"] if x["stage"] == 1), None)
            rows.append(dict(sit=(e["mission"], e["init"]), frac=f, y=int(e["stage_reached"] >= 2), npz=npz,
                             time=(done2["total"] - e["total_at_change"]) if done2 and "total" in done2 else None))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True, help="'run_dir:depth,...' (125 outputs with --save_change_obs)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=150)
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    sys.path.insert(0, f"{G}/jobs"); sys.path.insert(0, f"{G}/GalaxeaVLA")
    import lawam_hybrid as H
    dev = "cuda"
    lam = H.load_lawm(device="cpu"); lam.vision_encoder.to(dev).eval()
    rows = load_runs(a.runs)
    print(f"{len(rows)} (situation, depth) samples from {len({r['sit'] for r in rows})} situations", flush=True)
    for f in FRACS:
        rr = [r for r in rows if r["frac"] == f]; ts = [r["time"] for r in rr if r["y"] and r["time"] is not None]
        print(f"  depth {f:.2f}: changed step done {sum(r['y'] for r in rr)}/{len(rr)}, median steps to done {np.median(ts) if ts else float('nan'):.0f}", flush=True)
    V, S, OLD, NEW, FI, Y = [], [], [], [], [], []
    for r in rows:
        o = np.load(r["npz"], allow_pickle=True)
        V.append(torch.cat([dino_pooled(lam, H, o["image"], dev), dino_pooled(lam, H, o["wrist"], dev)]))
        S.append(torch.from_numpy(proprio(o))); OLD.append(int(o["old_cmd"])); NEW.append(int(o["new_cmd"]))
        FI.append(FRACS.index(r["frac"])); Y.append(r["y"])
    V, S = torch.stack(V), torch.stack(S); OLD, NEW, FI, Y = map(torch.tensor, (OLD, NEW, FI, Y))
    s_mean, s_std = S.mean(0), S.std(0) + 1e-3
    S = (S - s_mean) / s_std
    sits = sorted({r["sit"] for r in rows}); rng = np.random.default_rng(0); perm = rng.permutation(len(sits))
    fold = {sits[i]: k % 5 for k, i in enumerate(perm)}
    outcome = {(r["sit"], r["frac"]): (r["y"], r["time"]) for r in rows}

    def fit(idx):
        m = Student().to(dev); opt = torch.optim.AdamW(m.parameters(), lr=3e-4, weight_decay=1e-2)
        idx = torch.as_tensor(idx)
        for _ in range(a.epochs):
            m.train()
            for b in idx[torch.randperm(len(idx))].split(64):
                lg = m(V[b].to(dev), S[b].to(dev), OLD[b].to(dev), NEW[b].to(dev))
                loss = F.binary_cross_entropy_with_logits(lg.gather(1, FI[b].to(dev)[:, None])[:, 0], Y[b].float().to(dev))
                opt.zero_grad(); loss.backward(); opt.step()
        return m.eval()

    # cross-validated predictions per situation (using that situation's full-retraction observation as the input)
    preds = {}
    for k in range(5):
        tr = [i for i, r in enumerate(rows) if fold[r["sit"]] != k]
        m = fit(tr)
        with torch.no_grad():
            for i, r in enumerate(rows):
                if fold[r["sit"]] == k and r["frac"] == 1.0:
                    preds[r["sit"]] = torch.sigmoid(m(V[i:i + 1].to(dev), S[i:i + 1].to(dev), OLD[i:i + 1].to(dev), NEW[i:i + 1].to(dev)))[0].cpu().numpy()
    full = [s for s in preds if all((s, f) in outcome for f in FRACS)]
    print(f"\noffline policy evaluation on {len(full)} situations with all depths recorded (cross-validated):", flush=True)

    def evaluate(pick):
        ys, ts = [], []
        for s in full:
            y, t = outcome[(s, pick(s))]; ys.append(y)
            if y and t is not None:
                ts.append(t)
        return sum(ys), len(ys), (np.mean(ts) if ts else float("nan"))
    for name, pick in [("always full retraction", lambda s: 1.0), ("always release only (depth 0)", lambda s: 0.0)] + \
            [(f"student, margin {mg}", (lambda mg: lambda s: FRACS[next(i for i in range(len(FRACS)) if preds[s][i] >= preds[s].max() - mg)])(mg))
             for mg in (0.0, 0.05, 0.1, 0.2)] + \
            [("oracle: shortest depth that worked", lambda s: next((f for f in FRACS if outcome[(s, f)][0]), 1.0))]:
        k, n, t = evaluate(pick)
        print(f"  {name:36s} changed step done {k}/{n}  mean steps from change to done {t:.0f}", flush=True)
    m = fit(list(range(len(rows))))
    torch.save({"model": m.state_dict(), "s_mean": s_mean.tolist(), "s_std": s_std.tolist(), "fracs": FRACS, "n": len(rows)}, a.out)
    print("saved", a.out, flush=True)


if __name__ == "__main__":
    main()
