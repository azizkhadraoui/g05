"""Cache the mission benchmark's goal pictures (125_e2e_missions.py goal_picture: the last frame of the first demonstration
of each task in the merged LIBERO dataset, AV1 video) as .npy files, so missions never decode video at run time
(torchcodec cannot load the FFmpeg libraries on some V100 nodes).
Output: results/feat_cache/goal_pic_ds{dataset task index}.npy  [H, W, 3] uint8
"""
import glob
import os

import numpy as np
import pandas as pd
from torchcodec.decoders import VideoDecoder

G = os.environ.get("G05_ROOT", "/export/home/kaziz/g05")
DS = f"{G}/data/libero_merged_g05grip"; C = f"{G}/results/feat_cache"
dd = pd.concat([pd.read_parquet(f, columns=["index", "task_index", "episode_index", "frame_index"])
                for f in sorted(glob.glob(f"{DS}/data/*/*.parquet"))]).sort_values("index").set_index("index")
me = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"{DS}/meta/episodes/*/*.parquet"))]).set_index("episode_index")
key = "videos/observation.images.image/"
for t in sorted(dd["task_index"].unique()):
    st0 = int(dd.index[(dd["task_index"] == t).values & (dd["frame_index"] == 0).values][0])  # first demo of the task
    row = me.loc[int(dd.loc[st0, "episode_index"])]
    path = f"{DS}/videos/observation.images.image/chunk-{int(row[key + 'chunk_index']):03d}/file-{int(row[key + 'file_index']):03d}.mp4"
    img = VideoDecoder(path).get_frame_played_at(float(row[key + "to_timestamp"]) - 0.1).data.permute(1, 2, 0).numpy()
    np.save(f"{C}/goal_pic_ds{int(t)}.npy", img)
    print(t, img.shape, flush=True)
