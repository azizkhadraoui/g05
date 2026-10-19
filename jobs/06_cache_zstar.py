"""(1) Extract lawam_pretrain's LaWM (policy_backend.lam.*) into a small state file.
(2) Precompute the frozen teacher target z* = Q(frame_t, frame_t+tau) for every frame of the LIBERO training set.

z* follows LaWAM's own teacher call (get_latent_action, predict_future_frame=False, quantized output).
Pairs at the end of an episode use the last frame as the future (clamped); `clamped` marks them.
"""
import glob
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lawam_hybrid as H  # noqa: E402
from torchcodec.decoders import VideoDecoder  # noqa: E402

G = H.G05_ROOT
DS = f"{G}/data/libero_merged_no_noops_20hz"
CAM = "observation.images.image"
PRETRAIN = f"{H.LAWAM_ROOT}/results/Checkpoints/pretrain/lawam_pretrain/final_model/pytorch_model.pt"
OUT = f"{G}/results/lawm_cache"
TAU_S = float(os.environ.get("TAU_S", 0.4))
os.makedirs(OUT, exist_ok=True)
t0 = time.time()

# (1) LaWM state from lawam_pretrain
if not os.path.exists(H.LAWM_STATE):
    sd = torch.load(PRETRAIN, map_location="cpu", weights_only=False)
    for k in ("state_dict", "model", "module"):
        if isinstance(sd, dict) and k in sd and isinstance(sd[k], dict):
            sd = sd[k]
    P = "policy_backend.lam."
    lam_sd = {k[len(P):]: v for k, v in sd.items() if k.startswith(P)}
    assert len(lam_sd) == 660, len(lam_sd)
    torch.save(lam_sd, H.LAWM_STATE)
    del sd
    print(f"saved {H.LAWM_STATE} ({len(lam_sd)} tensors)")
lam = H.load_lawm()

# (2) frame table
info = json.load(open(f"{DS}/meta/info.json"))
fps = info["fps"]
k_fut = int(round(TAU_S * fps))
data = pd.concat([pd.read_parquet(f, columns=["index", "episode_index", "frame_index"])
                  for f in sorted(glob.glob(f"{DS}/data/*/*.parquet"))], ignore_index=True)
data = data.sort_values("index").reset_index(drop=True)
N = len(data)
assert (data["index"].values == np.arange(N)).all(), "dataset 'index' is not a dense 0..N-1 range"
eps = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"{DS}/meta/episodes/*/*.parquet"))], ignore_index=True)
print(f"{N} frames, {len(eps)} episodes, fps={fps}, tau={TAU_S}s -> {k_fut} frames")

zstar = torch.zeros(N, 32, dtype=torch.float32)
clamped = torch.zeros(N, dtype=torch.bool)
ep_index = torch.from_numpy(data["episode_index"].values.astype(np.int64))
fr_index = torch.from_numpy(data["frame_index"].values.astype(np.int64))
first_row = data.groupby("episode_index")["index"].min()

decoders = {}
done = 0
for _, e in eps.iterrows():
    ei, L = int(e["episode_index"]), int(e["length"])
    path = f"{DS}/" + info["video_path"].format(video_key=CAM, chunk_index=int(e[f"videos/{CAM}/chunk_index"]),
                                               file_index=int(e[f"videos/{CAM}/file_index"]))
    if path not in decoders:
        decoders[path] = VideoDecoder(path)
    t_start = float(e[f"videos/{CAM}/from_timestamp"])
    frames = decoders[path].get_frames_played_at([t_start + k / fps for k in range(L)]).data  # [L,3,H,W] uint8
    x = frames.float().div(255.0)
    if x.shape[-1] != 256:
        x = torch.nn.functional.interpolate(x, size=(256, 256), mode="bilinear", align_corners=False)
    x = H.imagenet_norm(x)
    fut = torch.clamp(torch.arange(L) + k_fut, max=L - 1)
    row0 = int(first_row[ei])
    for s in range(0, L, 128):
        idx = torch.arange(s, min(s + 128, L))
        v = torch.stack([x[idx], x[fut[idx]]], dim=1).cuda()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            out = lam.get_latent_action(videos=v, states=None, dec_videos=v, predict_future_frame=False)
        zstar[row0 + idx] = out["quantized"].float().reshape(len(idx), -1).cpu()
    clamped[row0 + torch.arange(L)] = (torch.arange(L) + k_fut) > (L - 1)
    done += 1
    if done % 100 == 0:
        print(f"  {done}/{len(eps)} episodes  ({time.time() - t0:.0f}s)", flush=True)

out_path = f"{OUT}/zstar_tau{TAU_S}.pt"
torch.save({"zstar": zstar, "clamped": clamped, "episode_index": ep_index, "frame_index": fr_index,
            "tau_s": TAU_S, "fps": fps}, out_path)
print(f"saved {out_path}: zstar {tuple(zstar.shape)} norm mean {zstar.norm(dim=-1).mean():.3f} "
      f"std-per-dim {zstar.std(0).mean():.3f}  clamped {clamped.float().mean():.3f}  ({time.time() - t0:.0f}s)")
