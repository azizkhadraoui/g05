"""Cosmos 3 Edge-Policy-DROID on the DROID household windows FLUX 3 Action was scored on (112_flux_action_eval.py):
predicted 32-step joint chunks vs what the robot did next, against the same no-motion and constant-velocity baselines.

Cosmos 3's policy mode takes one conditioning image (wrist view on top, the two exterior views below, 640x540) and the
instruction, and generates the next actions together with the video of executing them. The request follows the
checkpoint's own policy block (checkpoint.json: action_chunk_size 32, conditioning_fps 15), so the horizon matches
FLUX's (32 actions at 15 Hz, 2.1 s). Output rows: 7 absolute joint positions (rad), then gripper and two more channels.
Unlike FLUX, Cosmos receives no robot state, so its chunk may start offset from the true joint pose; two scores:
  abs    joint L1 of the predicted absolute positions against the real ones
  delta  joint L1 of the predicted motion (pred_k - pred_0) against the real motion (obs_k - state_t): is the movement
         right even when the starting pose read off the image is not (also computed for FLUX in --merge)
Shards: --shard k/n, then --merge <tag>. Run with the Cosmos diffusers venv. Output: results/wm_cosmos/<tag>.json
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps

G = "/export/home/kaziz/g05"
D = f"{G}/data/droid_household"
CK = f"{G}/wm/checkpoints"
WINDOWS = f"{G}/results/wm_flux/base.json"
OUT = f"{G}/results/wm_cosmos"
CAMS = {"wrist": "wrist_left", "exterior_1": "exterior_1_left", "exterior_2": "exterior_2_left"}
CATS = ("fold_towel", "fold_cloth", "washer")
W, HH, H = 640, 540, 32


def frame_at(path, t):
    import imageio.v2 as imageio
    return Image.fromarray(imageio.get_reader(path).get_data(t))


def concat(frames):
    """Wrist on top (full width), exterior 1 / 2 below: the policy's conditioning canvas (as 107_cosmos_policy_viz.py)."""
    can = Image.new("RGB", (W, HH)); top = HH // 2
    can.paste(ImageOps.fit(frames["wrist"], (W, top), method=Image.Resampling.BICUBIC), (0, 0))
    can.paste(ImageOps.fit(frames["exterior_1"], (W // 2, HH - top), method=Image.Resampling.BICUBIC), (0, top))
    can.paste(ImageOps.fit(frames["exterior_2"], (W // 2, HH - top), method=Image.Resampling.BICUBIC), (W // 2, top))
    return can


def load_states(eps):
    files = sorted(glob.glob(f"{D}/data/**/*.parquet", recursive=True))
    cols = ["episode_index", "frame_index", "observation.state.joint_position", "observation.state.gripper_position"]
    df = pd.concat([pd.read_parquet(f, columns=cols) for f in files])
    df = df[df["episode_index"].isin(eps)].sort_values(["episode_index", "frame_index"])
    return {int(e): np.concatenate([np.stack(r["observation.state.joint_position"].values),
                                    np.stack(r["observation.state.gripper_position"].values).reshape(-1, 1)], 1).astype(np.float32)
            for e, r in df.groupby("episode_index")}


def scores(pred, st, t):
    """pred [H, >=7], states [T, 8] -> abs and delta joint L1 (mean and at 0.5 / 1 / 2.1 s) for pred, hold, const-vel."""
    obs = st[t + 1: t + 1 + H, :7]
    hold = np.repeat(st[t:t + 1, :7], H, 0)
    cv = st[t, :7][None] + np.arange(1, H + 1)[:, None] * ((st[t, :7] - st[t - 3, :7]) / 3.0)[None]
    r = {}
    for name, m in (("model", pred[:H, :7]), ("hold", hold), ("constvel", cv)):
        a = np.abs(m - obs).mean(1)
        d = np.abs((m - m[0:1]) - (obs - st[t:t + 1, :7])).mean(1) if name == "model" else a  # baselines start at the true pose
        r.update({f"abs_{name}": float(a.mean()), f"abs_{name}_h": [float(a[k]) for k in (7, 15, 31)],
                  f"delta_{name}": float(d.mean()), f"delta_{name}_h": [float(d[k]) for k in (7, 15, 31)]})
    r["start_offset"] = float(np.abs(pred[0, :7] - st[t, :7]).mean())
    return r


def agg(rs):
    out = {"n": len(rs)}
    if not rs:
        return out
    for k in rs[0]:
        if isinstance(rs[0][k], float):
            out[k] = float(np.mean([r[k] for r in rs]))
        elif isinstance(rs[0][k], list):
            out[k] = np.mean([r[k] for r in rs], 0).tolist()
    for kind in ("abs", "delta"):
        out[f"{kind}_vs_hold"] = out[f"{kind}_model"] / out[f"{kind}_hold"]
        out[f"{kind}_beats_hold_frac"] = float(np.mean([r[f"{kind}_model"] < r[f"{kind}_hold"] for r in rs]))
    return out


def summarize(recs):
    return {"all": agg(recs), **{c: agg([r for r in recs if r["category"] == c]) for c in CATS}}


def report(summary, name="Cosmos"):
    for c, s in summary.items():
        if s["n"]:
            print(f"{c:11s} n={s['n']:3d}  joint L1 (2.1 s) abs: {name} {s['abs_model']:.3f} hold {s['abs_hold']:.3f} "
                  f"const-vel {s['abs_constvel']:.3f} ({name}/hold {s['abs_vs_hold']:.2f}) | motion only: {name} "
                  f"{s['delta_model']:.3f} ({s['delta_vs_hold']:.2f}x hold, better in {100 * s['delta_beats_hold_frac']:.0f}%) "
                  f"| start offset {s['start_offset']:.3f}", flush=True)


def merge(tag):
    parts = [json.load(open(f)) for f in sorted(glob.glob(f"{OUT}/{tag}_s*of*.json"))]
    res = dict(parts[0]); res["windows"] = [w for p in parts for w in p["windows"]]
    res["summary"] = summarize(res["windows"]); res["shards"] = len(parts)
    json.dump(res, open(f"{OUT}/{tag}.json", "w"), indent=1)
    report(res["summary"])
    # FLUX on the same windows, same two scores (its saved chunks + the same states)
    fl = json.load(open(WINDOWS)); recs = []
    states = load_states(sorted({int(w["episode"][2:]) for w in fl["windows"]}))
    for f in sorted(glob.glob(f"{G}/results/wm_flux/base_s*of*_actions.npz")):
        z = np.load(f)
        for p, a in zip(z["pred"], z["anchors"]):
            n, t = str(a).split(":"); t = int(t)
            w = next(x for x in fl["windows"] if x["episode"] == n and x["t"] == t)
            recs.append({"category": w["category"], **scores(p, states[int(n[2:])], t)})
    fs = summarize(recs); res["flux_same_scores"] = fs
    json.dump(res, open(f"{OUT}/{tag}.json", "w"), indent=1)
    print("--- FLUX 3 Action, same windows and scores:"); report(fs, "FLUX")
    print(f"merged {len(parts)} shards, {len(res['windows'])} windows -> {OUT}/{tag}.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Cosmos3-Edge-Policy-DROID")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shard", default="")
    ap.add_argument("--tag", default="edge")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    from diffusers import Cosmos3OmniPipeline, CosmosActionCondition
    from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
    t0 = time.time()
    wins = json.load(open(WINDOWS))["windows"]
    if a.shard:
        k, n_sh = map(int, a.shard.split("/")); wins = [w for i, w in enumerate(wins) if i % n_sh == k]
    states = load_states(sorted({int(w["episode"][2:]) for w in wins}))
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}[os.environ.get("COSMOS_DTYPE", "fp16")]
    pipe = Cosmos3OmniPipeline.from_pretrained(f"{CK}/{a.model}", torch_dtype=dtype, safety_checker=None, enable_safety_checker=False)
    pipe.enable_model_cpu_offload()
    for m in ("enable_tiling", "enable_slicing"):
        if hasattr(pipe.vae, m):
            getattr(pipe.vae, m)()
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=5.0, use_karras_sigmas=False)
    print(f"loaded {a.model} in {dtype} ({time.time() - t0:.0f}s); {len(wins)} windows", flush=True)
    recs, preds = [], []
    for i, w in enumerate(wins):
        t1 = time.time(); ep = int(w["episode"][2:]); t = w["t"]
        cond = concat({k: frame_at(f"{D}/clips/{ep:06d}/{v}.mp4", t) for k, v in CAMS.items()})
        g = torch.Generator(device="cuda").manual_seed(a.seed)
        res = pipe(prompt=w["task"].split(" | ")[0].strip(),
                   action=CosmosActionCondition(mode="policy", chunk_size=H, domain_name="droid_lerobot", resolution_tier=480,
                                                image=cond, view_point="concat_view"),
                   fps=15, num_inference_steps=a.steps, guidance_scale=1.0, use_system_prompt=False, generator=g)
        act = res.action[0]
        p = np.asarray(act.float().cpu() if torch.is_tensor(act) else act, dtype=np.float32)
        r = {"episode": w["episode"], "category": w["category"], "t": t, "sec": time.time() - t1,
             "finite": bool(np.isfinite(p).all()), "shape": list(p.shape), **scores(p, states[ep], t)}
        recs.append(r); preds.append(p)
        print(f"[{i + 1}/{len(wins)}] {w['episode']} t={t} {w['category']}: abs Cosmos {r['abs_model']:.3f} hold "
              f"{r['abs_hold']:.3f} | motion {r['delta_model']:.3f} | start offset {r['start_offset']:.3f} ({r['sec']:.0f}s)", flush=True)
    tag = a.tag + (f"_s{a.shard.replace('/', 'of')}" if a.shard else "")
    json.dump({"model": a.model, "dtype": str(dtype), "steps": a.steps, "chunk": H, "fps": 15,
               "summary": summarize(recs), "windows": recs}, open(f"{OUT}/{tag}.json", "w"), indent=1)
    np.savez(f"{OUT}/{tag}_actions.npz", pred=np.stack(preds), anchors=np.array([f"{w['episode']}:{w['t']}" for w in wins]))
    report(summarize(recs)); print("wrote", f"{OUT}/{tag}.json", flush=True)


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--merge":
        merge(sys.argv[2])
    else:
        main()
