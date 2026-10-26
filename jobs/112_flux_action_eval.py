"""FLUX 3 Action (Black Forest Labs, 7B world action model, DROID policy package) on the DROID household clips.

FLUX 3 Action denoises the next 32 actions (absolute joint targets + gripper closed fraction, 15 Hz, 2.1 s) jointly
with the next video latents; only the actions are decoded. Offline evaluation: from a real frame of a household
episode (wrist + both exterior views, 7 joints + gripper, the instruction), predict the 32-step chunk and compare it
with what the robot actually did next, against two motion-free baselines (hold the current joints; extrapolate the
current joint velocity). Same anchors for every model and seed (inference_seed of the package).

Memory: the DiT is 14.2 GB in bf16 and a T4 has 15 GB, so transformer blocks stay resident on the GPU up to
RESIDENT_GB and the rest stream from pinned CPU memory at every forward (a few GB per pass over PCIe). Captions are
encoded once, with the 4B Qwen3-VL text encoder on the GPU before the DiT is placed, then the encoder goes to the CPU;
the video VAE (fp32, see below) visits the GPU only to encode each observed frame.
The DiT runs in fp16 by default (T4 has no bf16 tensor cores); the first window is also run in bf16 and the
difference is logged, so fp16 can be checked against the package's own precision.

Inputs: data/droid_household clips (320x180 H.264, upsampled 2x to the 360x640 the package expects) + parquet rows,
episodes.json (99_droid_to_robojepa.py). Run with the flux-action venv. Output: results/wm_flux/<tag>.json (+ .npz)
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
import torch.nn as nn
import torch.nn.functional as F

G = "/export/home/kaziz/g05"
D = f"{G}/data/droid_household"
META = f"{G}/data/droid_household_robojepa/episodes.json"
CK = f"{G}/wm/checkpoints/flux-3-action-droid"
BASE = f"{G}/wm/checkpoints/flux-3-action-base"
OUT = f"{G}/results/wm_flux"
CAMS = {"images.wrist": "wrist_left", "images.left": "exterior_1_left", "images.right": "exterior_2_left"}
CATS = ("fold_towel", "fold_cloth", "washer")
H = 32  # chunk (15 Hz)


def frames_at(path, idxs):
    import av
    want, out = set(idxs), {}
    with av.open(path) as c:
        for i, f in enumerate(c.decode(video=0)):
            if i in want:
                out[i] = f.to_ndarray(format="rgb24")
            if i >= max(idxs):
                break
    return [out[i] for i in idxs]


def to_cam(img, dev):
    """(180, 320, 3) uint8 -> (1, 3, 360, 640) float in [0, 1] (bilinear 2x, done on the CPU as the package asks)."""
    x = torch.from_numpy(img).permute(2, 0, 1)[None].float() / 255.0
    return F.interpolate(x, size=(360, 640), mode="bilinear", align_corners=False).clamp_(0, 1).to(dev)


def place(policy, dev, resident_gb):
    """DiT blocks on the GPU up to resident_gb (in module order, so the DiT's first parameter is on the GPU and
    policy.device reports cuda); the remaining blocks keep pinned CPU weights copied in for each forward."""
    dit = policy.dit
    blocks = [m for m in dit.modules() if type(m).__name__ in ("ModeBlock", "SingleStreamBlock")]
    in_blocks = {id(p) for b in blocks for p in b.parameters()}
    for name, p in dit.named_parameters():
        if id(p) not in in_blocks:
            p.data = p.data.to(dev)
    for name, b in dit.named_buffers():
        b.data = b.data.to(dev)
    used, off = 0, []
    for b in blocks:
        size = sum(p.numel() * p.element_size() for p in b.parameters())
        if used + size <= resident_gb * 2**30:
            b.to(dev); used += size
        else:
            b._cpu = [p.data.pin_memory() for p in b.parameters()]
            for p, c in zip(b.parameters(), b._cpu):
                p.data = c

            def pre(mod, args):
                for p, c in zip(mod.parameters(), mod._cpu):
                    p.data = c.to(dev, non_blocking=True)

            def post(mod, args, out):
                for p, c in zip(mod.parameters(), mod._cpu):
                    p.data = c
                return out
            b.register_forward_pre_hook(pre); b.register_forward_hook(post)
            off.append(b)
    print(f"DiT blocks: {len(blocks) - len(off)} resident ({used / 2**30:.1f} GiB), {len(off)} streamed "
          f"({sum(sum(c.numel() * c.element_size() for c in b._cpu) for b in off) / 2**30:.1f} GiB)", flush=True)
    return off


def cast_dit(policy, off, dtype):
    """Cast the DiT's floating-point weights (resident and streamed) and the cached text contexts to dtype."""
    for p in policy.dit.parameters():
        if p.data.is_floating_point() and p.data.device.type == "cuda":
            p.data = p.data.to(dtype)
    for b in off:
        b._cpu = [c.to(dtype).pin_memory() if c.is_floating_point() else c for c in b._cpu]
        for p, c in zip(b.parameters(), b._cpu):
            p.data = c
    saved = dict(policy._ctx_cache)
    policy.set_compute_dtype(None)  # dtype_ <- the DiT's parameter dtype (clears the caption cache)
    policy._ctx_cache.update({k: (v[0].to(dtype), v[1]) for k, v in saved.items()})


def episodes(a):
    meta = json.load(open(META))
    rng = np.random.default_rng(a.seed)
    names = sorted(meta); rng.shuffle(names)
    picked = {c: [] for c in CATS}
    for n in names:
        c = meta[n]["category"]
        if c in picked and len(picked[c]) < a.per_category and meta[n]["T"] >= 4 * H:
            picked[c].append(n)
    return meta, [n for c in CATS for n in picked[c]]


def load_rows(eps_idx):
    files = sorted(glob.glob(f"{D}/data/**/*.parquet", recursive=True))
    import pyarrow.parquet as pq
    cols = pq.ParquetFile(files[0]).schema_arrow.names
    want = ["episode_index", "frame_index", "observation.state.joint_position", "observation.state.gripper_position"]
    want += [c for c in ("action.joint_position", "action.gripper_position") if c in cols]
    print("parquet action columns:", [c for c in cols if c.startswith("action")], flush=True)
    df = pd.concat([pd.read_parquet(f, columns=want) for f in files])
    df = df[df["episode_index"].isin(eps_idx)].sort_values(["episode_index", "frame_index"])
    out = {}
    for ep, r in df.groupby("episode_index"):
        st = np.concatenate([np.stack(r["observation.state.joint_position"].values),
                             np.stack(r["observation.state.gripper_position"].values).reshape(-1, 1)], 1).astype(np.float32)
        act = None
        if "action.joint_position" in r:
            act = np.concatenate([np.stack(r["action.joint_position"].values),
                                  np.stack(r["action.gripper_position"].values).reshape(-1, 1)], 1).astype(np.float32)
        out[int(ep)] = (st, act)
    return out


def errors(pred, ref):
    """pred / ref [H, 8] -> joint L1 (rad, mean over 7 joints) at every step, and gripper |err| mean."""
    j = np.abs(pred[:, :7] - ref[:, :7]).mean(1)
    return j, float(np.abs(pred[:, 7] - ref[:, 7]).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subfolder", default="", help="'' = base recipe (4 steps, CFG 4), variants/gd, variants/sd")
    ap.add_argument("--per_category", type=int, default=8)
    ap.add_argument("--anchors", default="0.2,0.5,0.8", help="fractions of each episode's valid start range")
    ap.add_argument("--dtype", default="fp16", choices=["fp16", "bf16"])
    ap.add_argument("--check_bf16", type=int, default=1, help="also run the first window in bf16 and log the difference")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--shard", default="", help="k/n: evaluate windows i with i %% n == k (merge: --merge <tag>)")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    sys.path.insert(0, f"{G}/wm/flux-action/src")
    from flux_action.models.text_encoder import load_text_encoder
    from flux_action.models.video_vae import load_video_vae
    from flux_action.policy import FluxActionPolicy
    dev = torch.device("cuda")
    t0 = time.time()
    meta, names = episodes(a)
    rows = load_rows([meta[n]["episode_index"] for n in names])
    anchors = []
    for n in names:
        st, act = rows[meta[n]["episode_index"]]
        T = min(meta[n]["T"], len(st))
        lo, hi = 5, T - H - 2
        for f in map(float, a.anchors.split(",")):
            anchors.append((n, int(lo + f * (hi - lo))))
    captions = sorted({meta[n]["task"].split(" | ")[0].strip() for n in names})
    print(f"{len(names)} episodes, {len(anchors)} windows, {len(captions)} captions ({time.time() - t0:.0f}s)", flush=True)

    vae = load_video_vae(f"{BASE}/video_vae.safetensors")
    # NATTEN's fused kernels take fp16/fp32 only below sm80 (T4 = sm75), and the package feeds the VAE bf16:
    # run the frame encoder in VAE_DTYPE (fp32 default: 5 GB, numerically the safest) and hand bf16 latents back.
    # The VAE visits the GPU only for its one frame per prediction, so the DiT's attention has the memory meanwhile.
    vdt = {"fp32": torch.float32, "fp16": torch.float16}[os.environ.get("VAE_DTYPE", "fp32")]
    vae.module.model.to(vdt)
    _encode_frame = vae.encode_frame

    def encode_frame(frame):
        vae.module.model.to(frame.device)
        try:
            return _encode_frame(frame.to(vdt)).to(torch.bfloat16)
        finally:
            vae.module.model.to("cpu"); torch.cuda.empty_cache()
    vae.encode_frame = encode_frame
    te = load_text_encoder(f"{BASE}/text_encoder")
    policy = FluxActionPolicy.from_pretrained(os.path.join(CK, a.subfolder) if a.subfolder else CK, video_vae=vae,
                                              text_encoder=te, device="cpu")
    policy.config.single_frame_encode = True  # serving path: encode the observed frame alone (vae.encode_frame)
    cfg = policy.config
    print(f"loaded ({time.time() - t0:.0f}s): sampler {cfg.sampler} steps {cfg.num_inference_steps} guidance "
          f"{cfg.guidance_scale}/{cfg.guidance_scale_action} chunk {cfg.chunk_size} fps {cfg.fps}", flush=True)
    # captions first, with the text encoder alone on the GPU
    te.to(dev)
    with torch.no_grad():
        policy._contexts(captions + [""], dev)
    te.to("cpu"); torch.cuda.empty_cache()
    off = place(policy, dev, float(os.environ.get("RESIDENT_GB", "6.5")))
    print(f"cuda {torch.cuda.memory_allocated() / 2**30:.1f} GiB after placement ({time.time() - t0:.0f}s)", flush=True)

    def predict(n, t):
        ep = meta[n]["episode_index"]
        st, _ = rows[ep]
        imgs = {k: frames_at(f"{D}/clips/{ep:06d}/{v}.mp4", [t])[0] for k, v in CAMS.items()}
        batch = {k: to_cam(im, dev) for k, im in imgs.items()}
        batch["state"] = torch.from_numpy(st[t][None]).to(dev)
        batch["task"] = [meta[n]["task"].split(" | ")[0].strip()]
        with torch.no_grad():
            return policy.predict_action_chunk(batch)[0].float().cpu().numpy()

    check = None
    if a.check_bf16:
        t1 = time.time(); p_bf = predict(*anchors[0]); tb = time.time() - t1
        cast_dit(policy, off, torch.float16 if a.dtype == "fp16" else torch.bfloat16)
        t1 = time.time(); p_lo = predict(*anchors[0]); tl = time.time() - t1
        check = {"max_abs_joint_diff": float(np.abs(p_bf[:, :7] - p_lo[:, :7]).max()),
                 "mean_abs_joint_diff": float(np.abs(p_bf[:, :7] - p_lo[:, :7]).mean()), "sec_bf16": tb, "sec_run": tl,
                 "finite": bool(np.isfinite(p_lo).all())}
        print(f"precision check on window 0: bf16 vs {a.dtype} joint |diff| max {check['max_abs_joint_diff']:.4f} "
              f"mean {check['mean_abs_joint_diff']:.4f} rad; {tb:.0f}s vs {tl:.0f}s per window", flush=True)
        if not check["finite"] or check["max_abs_joint_diff"] > 0.05:
            print("fp16 disagrees with bf16: continuing in bf16", flush=True)
            cast_dit(policy, off, torch.bfloat16); a.dtype = "bf16"
    elif a.dtype == "fp16":
        cast_dit(policy, off, torch.float16)

    recs, preds = [], []
    if a.shard:  # windows i with i % n == k (several GPUs share one evaluation; --merge combines the shards)
        k, n_sh = map(int, a.shard.split("/"))
        anchors = [w for i, w in enumerate(anchors) if i % n_sh == k]
    for i, (n, t) in enumerate(anchors):
        t1 = time.time()
        p = predict(n, t)
        ep = meta[n]["episode_index"]; st, act = rows[ep]
        obs = st[t + 1: t + 1 + H]
        hold = np.repeat(st[t][None], H, 0)
        vel = (st[t] - st[t - 3]) / 3.0
        cv = st[t][None] + np.arange(1, H + 1)[:, None] * vel[None]; cv[:, 7] = st[t, 7]
        r = {"episode": n, "category": meta[n]["category"], "t": t, "task": meta[n]["task"], "sec": time.time() - t1,
             "finite": bool(np.isfinite(p).all()), "motion": float(np.abs(obs[-1, :7] - st[t, :7]).mean())}
        for ref_name, ref in (("obs", obs), ("cmd", None if act is None else act[t: t + H])):
            if ref is None:
                continue
            for m_name, m in (("flux", p), ("hold", hold), ("constvel", cv)):
                j, g = errors(m, ref)
                r[f"{ref_name}_{m_name}_joint"] = float(j.mean())
                r[f"{ref_name}_{m_name}_joint_h"] = [float(j[k]) for k in (7, 15, 31)]
                r[f"{ref_name}_{m_name}_grip"] = g
        recs.append(r); preds.append(p)
        print(f"[{i + 1}/{len(anchors)}] {n} t={t} {r['category']}: joint L1 vs real  FLUX {r['obs_flux_joint']:.3f}  "
              f"hold {r['obs_hold_joint']:.3f}  const-vel {r['obs_constvel_joint']:.3f} rad  ({r['sec']:.0f}s)", flush=True)

    tag = (a.tag or ("base" if not a.subfolder else a.subfolder.split("/")[-1])) + \
        (f"_s{a.shard.replace('/', 'of')}" if a.shard else "")
    res = {"model": "flux-3-action-droid", "package": a.subfolder or "root (base recipe)", "dtype": a.dtype,
           "precision_check": check, "inference": {k: getattr(cfg, k) for k in ("sampler", "num_inference_steps",
           "guidance_scale", "guidance_scale_action", "sampler_shift", "inference_seed", "chunk_size", "fps")},
           "summary": summarize(recs), "windows": recs, "seconds": time.time() - t0}
    json.dump(res, open(f"{OUT}/{tag}.json", "w"), indent=1)
    np.savez(f"{OUT}/{tag}_actions.npz", pred=np.stack(preds), anchors=np.array([f"{n}:{t}" for n, t in anchors]))
    report(res["summary"])
    print("wrote", f"{OUT}/{tag}.json", flush=True)


def agg(rs):
    out = {"n": len(rs)}
    for ref in ("obs", "cmd"):
        for m in ("flux", "hold", "constvel"):
            k = f"{ref}_{m}_joint"
            if rs and k in rs[0]:
                out[k] = float(np.mean([r[k] for r in rs]))
                out[k + "_h"] = np.mean([r[k + "_h"] for r in rs], 0).tolist()
                out[f"{ref}_{m}_grip"] = float(np.mean([r[f"{ref}_{m}_grip"] for r in rs]))
        if f"{ref}_flux_joint" in out:
            out[f"{ref}_flux_vs_hold"] = out[f"{ref}_flux_joint"] / out[f"{ref}_hold_joint"]
            out[f"{ref}_flux_beats_hold_frac"] = float(np.mean([r[f"{ref}_flux_joint"] < r[f"{ref}_hold_joint"] for r in rs]))
    return out


def summarize(recs):
    return {"all": agg(recs), **{c: agg([r for r in recs if r["category"] == c]) for c in CATS}}


def report(summary):
    for c, s in summary.items():
        print(f"{c:11s} n={s['n']:3d}  joint L1 vs real (2.1 s): FLUX {s.get('obs_flux_joint', float('nan')):.3f}  "
              f"hold {s.get('obs_hold_joint', float('nan')):.3f}  const-vel {s.get('obs_constvel_joint', float('nan')):.3f}  "
              f"FLUX/hold {s.get('obs_flux_vs_hold', float('nan')):.2f}  FLUX better than hold in "
              f"{100 * s.get('obs_flux_beats_hold_frac', float('nan')):.0f}% of windows", flush=True)


def merge(tag):
    """OUT/<tag>_s<k>of<n>.json shards -> OUT/<tag>.json (windows concatenated, summary recomputed)."""
    parts = [json.load(open(f)) for f in sorted(glob.glob(f"{OUT}/{tag}_s*of*.json"))]
    res = dict(parts[0]); res["windows"] = [w for p in parts for w in p["windows"]]
    res["summary"] = summarize(res["windows"]); res["shards"] = len(parts)
    res["seconds"] = max(p["seconds"] for p in parts)
    json.dump(res, open(f"{OUT}/{tag}.json", "w"), indent=1)
    report(res["summary"])
    print(f"merged {len(parts)} shards, {len(res['windows'])} windows -> {OUT}/{tag}.json")


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--merge":
        merge(sys.argv[2])
    else:
        main()
