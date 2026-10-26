"""DROID household clips (90_dl_droid_household.sh: LeRobot v3.0 rows + per-episode mp4 clips at 15 fps) -> the episode
layout RoboJEPA's RobotVideoDataset reads (wm/robo_jepa/app/robo_jepa/datasets/robot_video_dataset.py):
    <out>/dataset_index.csv                       column "path"
    <out>/<episode>/data.h5
        cameras/{left,right,wrist}                relative mp4 path without ".mp4" (the loader appends it)
        trajectory/observation/robot_state/cartesian_position [T, 6]  end-effector XYZRPY
        trajectory/observation/robot_state/gripper_position   [T, 1]  0 = open, 1 = closed (DROID convention)
    <out>/episodes.json                           episode -> instruction, category, building, success
States are trimmed to the shortest camera clip so every state row has a frame. Runs in the RoboJEPA env (h5py, decord).
"""
import glob
import json
import os
import re

import h5py
import numpy as np
import pandas as pd
from decord import VideoReader, cpu

D = "/export/home/kaziz/g05/data/droid_household"
OUT = "/export/home/kaziz/g05/data/droid_household_robojepa"
CAMS = {"left": "exterior_1_left", "right": "exterior_2_left", "wrist": "wrist_left"}


def category(t):
    t = t.lower()
    if re.search(r"\bfold", t) and re.search(r"\btowels?\b", t):
        return "fold_towel"
    if re.search(r"\bfold", t):
        return "fold_cloth"
    if re.search(r"washer|washing machine", t):
        return "washer"
    return "laundry_other"


def main():
    os.makedirs(OUT, exist_ok=True)
    eps = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"{D}/meta/episodes/**/*.parquet", recursive=True))])
    sel = json.load(open(f"{D}/selection.json"))["episodes"]
    want = {int(e["episode_index"]): e["task"] for e in sel}
    eps = eps[eps["episode_index"].isin(want)].set_index("episode_index")
    data = pd.concat([pd.read_parquet(f, columns=["episode_index", "frame_index", "observation.state.cartesian_position",
                                                  "observation.state.gripper_position", "building", "is_episode_successful"])
                      for f in sorted(glob.glob(f"{D}/data/**/*.parquet", recursive=True))])
    data = data[data["episode_index"].isin(want)].sort_values(["episode_index", "frame_index"])
    index, meta, skipped = [], {}, 0
    for ep, rows in data.groupby("episode_index"):
        clip_dir = f"{D}/clips/{ep:06d}"
        paths = {k: f"{clip_dir}/{v}.mp4" for k, v in CAMS.items()}
        if not all(os.path.exists(p) and os.path.getsize(p) > 0 for p in paths.values()):
            skipped += 1
            continue
        n_frames = min(len(VideoReader(p, ctx=cpu(0))) for p in paths.values())
        T = min(n_frames, len(rows))
        if T < 32:
            skipped += 1
            continue
        name = f"ep{ep:06d}"
        os.makedirs(f"{OUT}/{name}", exist_ok=True)
        for k, p in paths.items():  # the loader resolves camera paths relative to the episode dir
            link = f"{OUT}/{name}/{k}.mp4"
            if not os.path.exists(link):
                os.symlink(p, link)
        with h5py.File(f"{OUT}/{name}/data.h5", "w") as h:
            for k in CAMS:
                h[f"cameras/{k}"] = k
            rs = h.create_group("trajectory/observation/robot_state")
            rs["cartesian_position"] = np.stack(rows["observation.state.cartesian_position"].values)[:T].astype(np.float32)
            rs["gripper_position"] = np.stack(rows["observation.state.gripper_position"].values)[:T].reshape(T, 1).astype(np.float32)
        index.append(name)
        meta[name] = {"episode_index": int(ep), "task": want[ep], "category": category(want[ep]), "T": int(T),
                      "building": str(rows["building"].iloc[0]), "success": bool(rows["is_episode_successful"].iloc[0])}
    pd.DataFrame({"path": index}).to_csv(f"{OUT}/dataset_index.csv", index=False)
    json.dump(meta, open(f"{OUT}/episodes.json", "w"), indent=1)
    cats = pd.Series([m["category"] for m in meta.values()]).value_counts().to_dict()
    print(f"{len(index)} episodes converted, {skipped} skipped; categories {cats}; "
          f"setup-room (RoboJEPA holdout building) {sum(m['building'] == 'setup room' for m in meta.values())}")


if __name__ == "__main__":
    main()
