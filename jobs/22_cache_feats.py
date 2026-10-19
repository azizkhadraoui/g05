"""Shared feature cache for the world-model and memory experiments.

For every frame of the (gripper-converted) LIBERO set:
  feats.npy    [N, 64, 768] fp16  DINOv3 ViT-B/16 layer -2 features of the main camera (LaWM convention: 256x256,
                                  ImageNet norm), average-pooled 16x16 -> 8x8 patches
  meta.npz     episode_index, frame_index, action [N,7] (G0.5 gripper convention), state [N,8], ep_length
"""
import glob
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lawam_hybrid as H  # noqa: E402
from torchcodec.decoders import VideoDecoder  # noqa: E402

G = H.G05_ROOT
DS = f"{G}/data/libero_merged_g05grip"
CAM = "observation.images.image"
OUT = f"{G}/results/feat_cache"
os.makedirs(OUT, exist_ok=True)
t0 = time.time()

info = json.load(open(f"{DS}/meta/info.json"))
fps = info["fps"]
data = pd.concat([pd.read_parquet(f, columns=["index", "episode_index", "frame_index", "action", "observation.state"])
                  for f in sorted(glob.glob(f"{DS}/data/*/*.parquet"))], ignore_index=True)
data = data.sort_values("index").reset_index(drop=True)
N = len(data)
assert (data["index"].values == np.arange(N)).all()
eps = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"{DS}/meta/episodes/*/*.parquet"))], ignore_index=True)
action = np.stack(data["action"].values).astype(np.float32)
state = np.stack(data["observation.state"].values).astype(np.float32)
ep_index = data["episode_index"].values.astype(np.int64)
fr_index = data["frame_index"].values.astype(np.int64)
ep_len = eps.set_index("episode_index")["length"].reindex(ep_index).values.astype(np.int64)
np.savez(f"{OUT}/meta.npz", episode_index=ep_index, frame_index=fr_index, action=action, state=state, ep_length=ep_len, fps=fps)
print(f"meta: N={N} action {action.shape} gripper values {np.unique(action[:, 6])[:5]} state {state.shape}", flush=True)

lam = H.load_lawm()
feats = np.lib.format.open_memmap(f"{OUT}/feats.npy", mode="w+", dtype=np.float16, shape=(N, 64, 768))
first_row = data.groupby("episode_index")["index"].min()
decoders = {}
for n, (_, e) in enumerate(eps.iterrows()):
    ei, L = int(e["episode_index"]), int(e["length"])
    path = f"{DS}/" + info["video_path"].format(video_key=CAM, chunk_index=int(e[f"videos/{CAM}/chunk_index"]),
                                               file_index=int(e[f"videos/{CAM}/file_index"]))
    if path not in decoders:
        decoders[path] = VideoDecoder(path)
    t_start = float(e[f"videos/{CAM}/from_timestamp"])
    frames = decoders[path].get_frames_played_at([t_start + k / fps for k in range(L)]).data
    row0 = int(first_row[ei])
    for s in range(0, L, 256):
        f = H.dino_from_uint8(lam, frames[s:s + 256].cuda())  # [b,256,768]
        b = f.shape[0]
        f = f.view(b, 16, 16, 768).permute(0, 3, 1, 2)
        f = F.avg_pool2d(f, 2).permute(0, 2, 3, 1).reshape(b, 64, 768)
        feats[row0 + s: row0 + s + b] = f.half().cpu().numpy()
    if (n + 1) % 100 == 0:
        print(f"  {n + 1}/{len(eps)} episodes ({time.time() - t0:.0f}s)", flush=True)
feats.flush()
print(f"saved {OUT}/feats.npy {feats.shape} ({time.time() - t0:.0f}s)")
