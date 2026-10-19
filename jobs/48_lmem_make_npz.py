"""LIBERO-Mem training data for G0.5, in the recorded-episode format 07_train_hybrid.py reads (_RecoveryFrames).

Repetition tasks only (bowl x1/x3/x5/x7, bottle x1/x3), demos 0-99 (100-119 stay held out for closed-loop eval).
Exact state replay, 256 px agent + wrist views through G0.5's own image transform, G0.5 state/action conventions,
no-op steps removed (as in G0.5's LIBERO data). Images kept every IMG_STRIDE steps (actions at full rate for chunks).
Each episode carries two instructions ("||"-separated, one is sampled per training example):
  the original task instruction, and the single-cycle instruction used under the memory module.
Output: results/lmem_train/t{tid}/rec/demo{k}.npz
"""
import glob
import os
import re
import sys
import time
from multiprocessing import get_context

import numpy as np

G = "/export/home/kaziz/g05"
RAW = f"{G}/libero_mem/raw"
OUT = f"{G}/results/lmem_train"
IMG_STRIDE = 2
TASKS = [0, 1, 2, 3, 4, 5]
SINGLE = {"bowl": "pick up the bowl and place it back on the plate", "bottle": "lift the bottle and put it down on the plate"}
sys.path.insert(0, f"{G}/GalaxeaVLA")
sys.path.insert(0, f"{G}/jobs")


def is_noop(a, prev):
    if prev is None:
        return np.linalg.norm(a[:-1]) < 1e-4
    return np.linalg.norm(a[:-1]) < 1e-4 and a[-1] == prev[-1]


def render_task(tid):
    import h5py
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from experiments.libero.libero_eval_utils import extract_libero_images, quat2axisangle

    suite = benchmark.get_benchmark_dict()["libero_mem"]()
    task = suite.get_task(tid)
    lang = re.sub(r"^\d+\s+", "", task.language)
    single = SINGLE["bottle" if "bottle" in lang else "bowl"]
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256)
    env.seed(0); env.reset()
    d = f"{OUT}/t{tid}/rec"; os.makedirs(d, exist_ok=True)
    n_frames = 0
    with h5py.File(glob.glob(f"{RAW}/{task.name}_demo.hdf5")[0], "r") as f:
        keys = sorted(f["data"].keys(), key=lambda s: int(s.split("_")[-1]))[:100]
        for k in keys:
            states, acts = f["data"][k]["states"][()], f["data"][k]["actions"][()]
            keep, prev = [], None
            for t, a in enumerate(acts):
                if not is_noop(a, prev):
                    keep.append(t); prev = a
            imgs, wrists, st, ac = [], [], [], []
            for j, t in enumerate(keep):
                obs = env.set_init_state(states[t])
                s = np.concatenate([obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]),
                                    np.asarray(obs["robot0_gripper_qpos"]).reshape(-1)[:2]]).astype(np.float32)
                a = acts[t].astype(np.float32).copy()
                a[-1] = (1.0 - a[-1]) / 2.0  # env gripper (-1 open / +1 close) -> G0.5 dataset convention (1 open / 0 close)
                st.append(s); ac.append(a)
                if j % IMG_STRIDE == 0:
                    im = extract_libero_images(obs)
                    imgs.append(im["image"]); wrists.append(im["wrist_image"])
            np.savez(f"{d}/{k}.npz", image=np.stack(imgs), wrist_image=np.stack(wrists), state=np.stack(st), action=np.stack(ac),
                     img_stride=IMG_STRIDE, task_description=f"{lang}||{single}")
            n_frames += len(keep)
    env.close()
    return tid, lang, single, n_frames


if __name__ == "__main__":
    t0 = time.time()
    with get_context("spawn").Pool(len(TASKS)) as pool:
        for r in pool.imap_unordered(render_task, TASKS):
            print(r, f"{time.time() - t0:.0f}s", flush=True)
