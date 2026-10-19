"""Phase 0 step 4: does LaWAM's pretrained latent world model (LaWM) predict future DINOv3 features of LIBERO frames
better than copying the current frame?

Uses the oracle latent action from the LAM inverse-dynamics encoder (which sees both frames), so this measures the
decoder's best case; the policy later has to predict that latent itself.

Sources: lawam_lam (LAM release) and lawam_pretrain (policy pretrain; decoder fine-tuned during policy pretraining).
Horizons: 0.4 s (LaWAM LIBERO SFT) and 1.2 s (LaWM pretraining).
"""
import copy
import glob
import json
import os
import random
import sys
import time

import pandas as pd
import torch
import torch.nn.functional as F
import yaml

G, L = os.environ["G05_ROOT"], os.environ["LAWAM_ROOT"]
sys.path.insert(0, L)
from latent_action_model.core.lam_model import load_latent_action_model  # noqa: E402
from torchcodec.decoders import VideoDecoder  # noqa: E402

DS = f"{G}/data/libero_merged_no_noops_20hz"
CAM = "observation.images.image"
LAM_DIR = f"{L}/latent_action_model/logs/dino_large_vae/lam_release"
PRETRAIN = f"{L}/results/Checkpoints/pretrain/lawam_pretrain/final_model/pytorch_model.pt"
DINO = f"{L}/weights/dinov3-vitb16-pretrain-lvd1689m"
OUT = f"{G}/results/lawm_sanity"
N_PAIRS = int(os.environ.get("N_PAIRS", 512))
BATCH, SEED = 32, 0
HORIZONS_S = [0.4, 1.2]
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
os.makedirs(OUT, exist_ok=True)
dev = torch.device("cuda")

cfg = yaml.safe_load(open(f"{LAM_DIR}/dino_large_vae.yaml"))
cfg["model"]["vision_model_id"] = DINO  # local copy instead of the HF id
YAML = f"{OUT}/lam_local.yaml"
yaml.safe_dump(cfg, open(YAML, "w"))


def load_lam_release():
    return load_latent_action_model(f"{LAM_DIR}/checkpoints/pytorch_model.pt", YAML)


def load_from_pretrain(base):
    sd = torch.load(PRETRAIN, map_location="cpu", weights_only=False)
    for k in ("state_dict", "model", "module"):
        if isinstance(sd, dict) and k in sd and isinstance(sd[k], dict):
            sd = sd[k]
    lam_keys = [k for k in sd if k.startswith("lam.") or ".lam." in k]
    if not lam_keys:
        raise RuntimeError(f"no lam.* keys in pretrain checkpoint; sample keys: {list(sd)[:10]}")
    k0 = lam_keys[0]
    prefix = k0[: k0.index("lam.") + len("lam.")]
    sub = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
    m = copy.deepcopy(base)
    ms = m.state_dict()
    missing = sorted(set(ms) - set(sub))
    unexpected = sorted(set(sub) - set(ms))
    mismatch = [k for k in set(ms) & set(sub) if ms[k].shape != sub[k].shape]
    report = {
        "prefix": prefix,
        "n_ckpt_lam_keys": len(sub),
        "n_model_keys": len(ms),
        "missing": len(missing),
        "missing_sample": missing[:8],
        "missing_all_vision_encoder": all(k.startswith("vision_encoder.") for k in missing),
        "unexpected": len(unexpected),
        "unexpected_sample": unexpected[:8],
        "shape_mismatch": mismatch[:8],
    }
    if mismatch:
        raise RuntimeError(f"shape mismatches loading pretrain LaWM: {report}")
    m.load_state_dict(sub, strict=False)
    # How far did policy pretraining move each part relative to the LAM release?
    drift = {}
    for part in ("encoder.", "decoder.", "vision_encoder."):
        num = den = 0.0
        for k, v in sub.items():
            if k.startswith(part) and k in ms and v.dtype.is_floating_point:
                num += (v.float() - ms[k].float()).pow(2).sum().item()
                den += ms[k].float().pow(2).sum().item()
        drift[part.rstrip(".")] = (num / den) ** 0.5 if den else None
    report["relative_param_drift_vs_lam_release"] = drift
    return m, report


def sample_pairs(k_frames, n, seed):
    info = json.load(open(f"{DS}/meta/info.json"))
    fps = info["fps"]
    eps = pd.concat(
        [pd.read_parquet(f) for f in sorted(glob.glob(f"{DS}/meta/episodes/*/*.parquet"))], ignore_index=True
    )
    eps = eps[eps["length"] > k_frames + 1].reset_index(drop=True)
    rng = random.Random(seed)
    pairs = []
    for _ in range(n):
        e = eps.iloc[rng.randrange(len(eps))]
        t = rng.randrange(0, int(e["length"]) - k_frames)
        path = f"{DS}/" + info["video_path"].format(
            video_key=CAM,
            chunk_index=int(e[f"videos/{CAM}/chunk_index"]),
            file_index=int(e[f"videos/{CAM}/file_index"]),
        )
        t0 = float(e[f"videos/{CAM}/from_timestamp"]) + t / fps
        pairs.append((path, t0, t0 + k_frames / fps, int(e["episode_index"]), t))
    return pairs


def decode_pairs(pairs):
    by_file = {}
    for i, (p, _, _, _, _) in enumerate(pairs):
        by_file.setdefault(p, []).append(i)
    f0 = [None] * len(pairs)
    f1 = [None] * len(pairs)
    for p, idx in by_file.items():
        d = VideoDecoder(p)
        ts = [pairs[i][1] for i in idx] + [pairs[i][2] for i in idx]
        fr = d.get_frames_played_at(ts).data  # [2n, 3, H, W] uint8
        for j, i in enumerate(idx):
            f0[i], f1[i] = fr[j], fr[len(idx) + j]
    return torch.stack(f0), torch.stack(f1)


def to_input(f0, f1, flip=False):
    v = torch.stack([f0, f1], dim=1).float().div_(255.0)  # [B, 2, 3, H, W]
    if flip:
        v = v.flip(-1, -2)
    b = v.shape[0]
    v = v.reshape(-1, *v.shape[2:])
    if v.shape[-1] != 256 or v.shape[-2] != 256:
        v = F.interpolate(v, size=(256, 256), mode="bilinear", align_corners=False)
    v = (v - IMAGENET_MEAN) / IMAGENET_STD
    return v.reshape(b, 2, *v.shape[1:])


def _first(x):
    return x[0] if isinstance(x, tuple) else x


@torch.no_grad()
def evaluate(model, f0, f1, flip=False):
    model.eval().to(dev)
    rows = []
    g = torch.Generator().manual_seed(SEED)
    for s in range(0, len(f0), BATCH):
        v = to_input(f0[s:s + BATCH], f1[s:s + BATCH], flip).to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model.get_latent_action(videos=v, states=None, dec_videos=v, predict_future_frame=True)
            recon, tgt, cur, z = _first(out["recon"]), out["tgt"], out["dec_in"], out["quantized"]
            perm = torch.randperm(z.shape[0], generator=g).to(dev)
            recon_shuf = _first(model.decoder(features=cur, actions=z[perm]))
        r, t, c, rs = (x.float().reshape(x.shape[0], -1, x.shape[-1]) for x in (recon, tgt, cur, recon_shuf))
        for i in range(r.shape[0]):
            rows.append({
                "mse_pred": F.mse_loss(r[i], t[i]).item(),
                "mse_copy": F.mse_loss(c[i], t[i]).item(),
                "mse_shuffled_z": F.mse_loss(rs[i], t[i]).item(),
                "cos_pred": F.cosine_similarity(r[i], t[i], dim=-1).mean().item(),
                "cos_copy": F.cosine_similarity(c[i], t[i], dim=-1).mean().item(),
            })
        if s == 0:
            print(f"    shapes: recon {tuple(recon.shape)} tgt {tuple(tgt.shape)} cur {tuple(cur.shape)} z {tuple(z.shape)}")
    return pd.DataFrame(rows)


def _stats(d):
    return {
        "n": len(d),
        "mse_pred": d["mse_pred"].mean(),
        "mse_copy": d["mse_copy"].mean(),
        "mse_shuffled_z": d["mse_shuffled_z"].mean(),
        "ratio_pred_over_copy": d["mse_pred"].mean() / d["mse_copy"].mean(),
        "ratio_shuffled_over_pred": d["mse_shuffled_z"].mean() / d["mse_pred"].mean(),
        "frac_pred_beats_copy": float((d["mse_pred"] < d["mse_copy"]).mean()),
        "cos_pred": d["cos_pred"].mean(),
        "cos_copy": d["cos_copy"].mean(),
    }


def summarize(df):
    moving = df[df["mse_copy"] >= df["mse_copy"].median()]
    return {"all": _stats(df), "moving_half": _stats(moving)}


t_start = time.time()
results = {"n_pairs": N_PAIRS, "seed": SEED, "camera": CAM, "sources": {}}
print("loading lawam_lam (strict load inside load_latent_action_model) ...")
lam = load_lam_release()
print(f"  code_dim={lam.code_dim} input_dim={lam.input_dim} grid={lam.grid_height}x{lam.grid_width}")
models = {"lawam_lam": lam}
try:
    print("loading lawam_pretrain lam.* ...")
    models["lawam_pretrain"], rep = load_from_pretrain(lam)
    results["pretrain_load_report"] = rep
    print("  ", json.dumps(rep, default=str))
except Exception as e:  # keep the lam_release numbers even if this fails
    results["pretrain_load_error"] = repr(e)
    print("  pretrain load FAILED:", repr(e))

for h in HORIZONS_S:
    pairs = sample_pairs(int(round(h * 20)), N_PAIRS, SEED)
    f0, f1 = decode_pairs(pairs)
    print(f"horizon {h}s: {len(pairs)} pairs decoded, frames {tuple(f0.shape)}")
    for name, m in models.items():
        for flip in (False, True):
            key = f"{name}/tau{h}s" + ("/flip180" if flip else "")
            df = evaluate(m, f0, f1, flip)
            res = summarize(df)
            results["sources"][key] = res
            a, mv = res["all"], res["moving_half"]
            print(
                f"  {key:32s} pred/copy={a['ratio_pred_over_copy']:.3f} (moving {mv['ratio_pred_over_copy']:.3f})  "
                f"beats-copy={a['frac_pred_beats_copy']:.2f}  shuffled/pred={a['ratio_shuffled_over_pred']:.3f}  "
                f"cos pred {a['cos_pred']:.3f} vs copy {a['cos_copy']:.3f}"
            )
        m.to("cpu")
results["wall_s"] = time.time() - t_start
json.dump(results, open(f"{OUT}/summary.json", "w"), indent=2, default=float)
print(f"saved {OUT}/summary.json  ({results['wall_s']:.0f}s)")
