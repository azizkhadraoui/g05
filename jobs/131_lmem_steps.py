"""LIBERO-Mem object-identity tasks (suite ids 6-9: swap, rotate, cheese -> that / empty basket): step labels and step
episodes from the demos (lmem_steps.py defines the steps, names and instruction classes).

For every demo (0-119): replay the simulator states, detect the completed moves from object positions, and label each
5 Hz frame of results/lmem_feats (45_lmem_feats.py rows) with the instruction of the step in progress (the demo's own next
move; "" once all moves are done). These labels train the memory decoders (46_lmem_count.py --labels).
For training demos (0-99): G0.5 training episodes in the recorded format of 48_lmem_make_npz.py:
  <demo>_full.npz  the whole demo, original task instruction (condition A: fine-tuned G0.5 alone)
  <demo>_s<k>.npz  one step (from the previous move's completion to 6 steps after this one), explicit step instruction
Output: results/lmem_steps/t{tid}/rec/*.npz, results/lmem_feats/labels_instr.npy (+ _classes.json, _report.json)
"""
import glob
import json
import os
import re
import sys
import time
from multiprocessing import get_context

import numpy as np

G = "/export/home/kaziz/g05"
RAW = f"{G}/libero_mem/raw"
OUT = f"{G}/results/lmem_steps"
IMG_STRIDE = 2
sys.path.insert(0, f"{G}/GalaxeaVLA")
sys.path.insert(0, f"{G}/jobs")
import lmem_steps as S  # noqa: E402


def is_noop(a, prev):
    if prev is None:
        return np.linalg.norm(a[:-1]) < 1e-4
    return np.linalg.norm(a[:-1]) < 1e-4 and a[-1] == prev[-1]


def process(tid):
    import h5py
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from experiments.libero.libero_eval_utils import extract_libero_images, quat2axisangle

    suite = benchmark.get_benchmark_dict()["libero_mem"]()
    task = suite.get_task(tid)
    lang = re.sub(r"^\d+\s+", "", task.language)
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256)
    env.seed(0); env.reset(); P = env.env
    names = S.objects(tid)
    pos = lambda: {n: P.sim.data.get_joint_qpos(n + "_joint0")[:3].copy() for n in names}  # noqa: E731
    m = np.load(f"{G}/results/lmem_feats/meta.npz")
    rows_all, labs_all, rep, steps = [], [], {"task": lang, "n_events": {}, "sequences": {}, "segments": 0, "frames": 0}, {}
    d = f"{OUT}/t{tid}/rec"; os.makedirs(d, exist_ok=True)
    with h5py.File(glob.glob(f"{RAW}/{task.name}_demo.hdf5")[0], "r") as f:
        keys = sorted(f["data"].keys(), key=lambda s: int(s.split("_")[-1]))
        for di, k in enumerate(keys):
            states, acts = f["data"][k]["states"][()], f["data"][k]["actions"][()]
            P.sim.set_state_from_flattened(states[0]); tr = S.StepTracker(tid, pos())
            for t in range(1, len(states)):
                P.sim.set_state_from_flattened(states[t]); tr.update(pos())
            tr.finish()
            if di < 3:
                print(json.dumps({"tid": tid, "demo": k, "len": len(states), "events": [list(map(str, e)) for e in tr.events]}), flush=True)
            comp = [e[-1] for e in tr.events]; ins = [tr.instruction(e) for e in tr.events]
            lab = np.full(len(states), S.CLASS_ID[S.DONE], np.int64)
            for t in range(len(states)):
                kk = next((i for i, c in enumerate(comp) if c > t), None)
                if kk is not None:
                    lab[t] = S.CLASS_ID[ins[kk]]
            rows = np.nonzero((m["task"] == tid) & (m["demo"] == di))[0]
            rows_all.append(rows); labs_all.append(lab[np.minimum(m["frame"][rows] * 4, len(lab) - 1)])
            rep["n_events"][len(tr.events)] = rep["n_events"].get(len(tr.events), 0) + 1
            seq = " | ".join(ins); rep["sequences"][seq] = rep["sequences"].get(seq, 0) + 1
            if di < 100:
                steps[k] = (comp, ins)
        frac = rep["n_events"].get(S.EXPECTED_EVENTS[tid], 0) / len(keys)
        if frac < 0.8:  # broken move detection: report without rendering (the parent exits non-zero)
            env.close(); rep["aborted"] = True
            return tid, np.concatenate(rows_all), np.concatenate(labs_all), rep
        for k, (comp, ins) in steps.items():
            states, acts = f["data"][k]["states"][()], f["data"][k]["actions"][()]
            keep, prev = [], None
            for t, a in enumerate(acts):
                if not is_noop(a, prev):
                    keep.append(t); prev = a
            imgs, wrists, st, ac = [], [], [], []
            for t in keep:
                obs = env.set_init_state(states[t])
                st.append(np.concatenate([obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]),
                                          np.asarray(obs["robot0_gripper_qpos"]).reshape(-1)[:2]]).astype(np.float32))
                a = acts[t].astype(np.float32).copy(); a[-1] = (1.0 - a[-1]) / 2.0  # env gripper -> G0.5 dataset convention
                ac.append(a); im = extract_libero_images(obs); imgs.append(im["image"]); wrists.append(im["wrist_image"])
            imgs, wrists, st, ac = np.stack(imgs), np.stack(wrists), np.stack(st), np.stack(ac)
            np.savez(f"{d}/{k}_full.npz", image=imgs[::IMG_STRIDE], wrist_image=wrists[::IMG_STRIDE], state=st, action=ac,
                     img_stride=IMG_STRIDE, task_description=lang)
            keep = np.asarray(keep); start = 0
            for j, (c, text) in enumerate(zip(comp, ins)):
                j0, j1 = int(np.searchsorted(keep, start)), int(np.searchsorted(keep, c + 6))
                if j1 - j0 >= 10:
                    np.savez(f"{d}/{k}_s{j}.npz", image=imgs[j0:j1:IMG_STRIDE], wrist_image=wrists[j0:j1:IMG_STRIDE], state=st[j0:j1],
                             action=ac[j0:j1], img_stride=IMG_STRIDE, task_description=text)
                    rep["segments"] += 1
                start = c + 1
            rep["frames"] += len(keep)
    env.close()
    return tid, np.concatenate(rows_all), np.concatenate(labs_all), rep


if __name__ == "__main__":
    t0 = time.time()
    m = np.load(f"{G}/results/lmem_feats/meta.npz")
    labels = np.full(len(m["task"]), -1, np.int64); report = {}
    with get_context("spawn").Pool(len(S.TASKS)) as pool:
        for tid, rows, labs, rep in pool.imap_unordered(process, S.TASKS):
            labels[rows] = labs
            rep["class_hist"] = {S.CLASSES[c] or "done": int((labs == c).sum()) for c in np.unique(labs)}
            report[tid] = rep
            print(json.dumps({"tid": tid, "t": round(time.time() - t0), "n_events": rep["n_events"], "segments": rep["segments"],
                              "top_sequences": sorted(rep["sequences"].items(), key=lambda kv: -kv[1])[:4]}), flush=True)
    np.save(f"{G}/results/lmem_feats/labels_instr.npy", labels)
    json.dump(S.CLASSES, open(f"{G}/results/lmem_feats/labels_instr_classes.json", "w"), indent=1)
    json.dump(report, open(f"{G}/results/lmem_feats/labels_instr_report.json", "w"), indent=1)
    print("labelled rows", int((labels >= 0).sum()), "of", len(labels), f"{time.time() - t0:.0f}s", flush=True)
    ok = {tid: report[tid]["n_events"].get(S.EXPECTED_EVENTS[tid], 0) / max(sum(report[tid]["n_events"].values()), 1) for tid in report}
    print("fraction of demos with the expected number of moves:", json.dumps(ok), flush=True)
    if min(ok.values()) < 0.8:  # stop the chain (decoders, fine-tune) rather than train on broken step labels
        sys.exit(3)
