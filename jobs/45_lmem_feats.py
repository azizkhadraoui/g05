"""LIBERO-Mem offline memory test, step 1: render every demo (exact state replay) at 5 Hz, agent view, 256 px, the same
image transform G0.5 uses (extract_libero_images), pooled DINOv3 features (8x8x768, as 22_cache_feats.py).

Label = number of pick-and-place cycles completed so far (gripper close -> open transitions in the demo's actions up to
that frame). That is the quantity the tasks require remembering ("3 times", swap, rotate) and is not visible in a single
frame once the object is back on a plate.
Output: results/lmem_feats/{feats.npy [N,64,768] fp16, meta.npz}
"""
import glob
import json
import os
import sys
import time
from multiprocessing import get_context

import numpy as np

G = "/export/home/kaziz/g05"
RAW = f"{G}/libero_mem/raw"
OUT = f"{G}/results/lmem_feats"
STRIDE = 4  # 20 Hz control -> 5 Hz frames
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, f"{G}/GalaxeaVLA")
sys.path.insert(0, f"{G}/jobs")


def render_task(tid):
    import h5py
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from experiments.libero.libero_eval_utils import extract_libero_images

    suite = benchmark.get_benchmark_dict()["libero_mem"]()
    task = suite.get_task(tid)
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256)
    env.seed(0); env.reset()
    imgs, demo, frame, count, length = [], [], [], [], []
    with h5py.File(glob.glob(f"{RAW}/{task.name}_demo.hdf5")[0], "r") as f:
        keys = sorted(f["data"].keys(), key=lambda s: int(s.split("_")[-1]))
        for di, k in enumerate(keys):
            states, acts = f["data"][k]["states"][()], f["data"][k]["actions"][()]
            g = acts[:, -1] > 0  # closed
            releases = np.cumsum(np.r_[False, g[:-1] & ~g[1:]])  # close -> open transitions up to t
            for t in range(0, len(states), STRIDE):
                obs = env.set_init_state(states[t])
                imgs.append(extract_libero_images(obs)["image"])
                demo.append(di); frame.append(t // STRIDE); count.append(int(releases[t])); length.append(len(states) // STRIDE)
    env.close()
    np.save(f"{OUT}/tmp_imgs_{tid}.npy", np.stack(imgs))
    return tid, np.array(demo), np.array(frame), np.array(count), np.array(length)


if __name__ == "__main__":
    t0 = time.time()
    with get_context("spawn").Pool(10) as pool:
        parts = sorted(pool.map(render_task, range(10)))
    print(f"rendered ({time.time() - t0:.0f}s)", flush=True)
    import torch
    import torch.nn.functional as F
    import t4_fp16  # noqa: F401
    import lawam_hybrid as H

    lam = H.load_lawm()
    feats, meta = [], {k: [] for k in ("task", "demo", "frame", "count", "length")}
    for tid, demo, frame, count, length in parts:
        imgs = np.load(f"{OUT}/tmp_imgs_{tid}.npy")
        for s in range(0, len(imgs), 256):
            x = torch.from_numpy(imgs[s:s + 256]).permute(0, 3, 1, 2).cuda()
            with torch.no_grad():
                f = H.dino_from_uint8(lam, x)
            b = f.shape[0]
            f = F.avg_pool2d(f.view(b, 16, 16, 768).permute(0, 3, 1, 2), 2).permute(0, 2, 3, 1).reshape(b, 64, 768)
            feats.append(f.half().cpu().numpy())
        for k, v in zip(("task", "demo", "frame", "count", "length"), (np.full(len(demo), tid), demo, frame, count, length)):
            meta[k].append(v)
        os.remove(f"{OUT}/tmp_imgs_{tid}.npy")
        print(f"task {tid}: {len(imgs)} frames, max count {count.max()} ({time.time() - t0:.0f}s)", flush=True)
    np.save(f"{OUT}/feats.npy", np.concatenate(feats))
    np.savez(f"{OUT}/meta.npz", **{k: np.concatenate(v) for k, v in meta.items()})
    print(json.dumps({"frames": int(sum(len(v) for v in meta["task"])), "t": round(time.time() - t0)}))
