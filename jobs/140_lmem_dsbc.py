"""Noise policy for a weak skill: DSBC (109_dsbc_train.py) on the LIBERO-Mem basket tasks (8, 9), with step-level labels.

Data: G0.5 fine-tuned on LIBERO-Mem (LMEMSTEP_ex_v100, served by 94_frs_serve.py with HYBRID_CKPT / EXPERT_CKPT) runs the
tasks with step instructions (51_lmem_memclient.py --halt_mode instr_oracle --rec_prefix P) on training start states;
every request is saved with the exact noise the sampler started from. Episodes are long (two steps over ~400 policy
steps), so episode success is a coarse label; instead a request is a positive when the step it was issued in ("put the
cream cheese in the nearest basket", then "move the X basket to the center") is completed within --window policy steps
(the client logs the benchmark's satisfied-subgoal count at every request: trial_final[k]["req_k"]; the oracle step
tracker misses basket moves the robot is still holding at the end). In deployment this label is
a step-success detector or an operator's mark; at test time the noise policy sees only the camera, the robot state and
the current step instruction. Model and fit: 109's NoisePolicy, with the 22 LIBERO-Mem step instructions as commands.
Usage: python 140_lmem_dsbc.py --sets P:results/lmem_dsbc/rec,... --out results/lmem_dsbc/policy.pt [--label episode]
"""
import argparse
import glob
import importlib.util
import json
import os
import re
import sys

import numpy as np
import torch

G = os.environ.get("G05_ROOT", "/export/home/kaziz/g05")
JOBS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, JOBS)
import lmem_steps as LS  # noqa: E402

_spec = importlib.util.spec_from_file_location("dsbc", os.path.join(JOBS, "109_dsbc_train.py"))
DS = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(DS)


def labels(results_root, prefix, window, mode, col=2):
    """(prefix, task, init state, policy step) -> 0/1 for every request logged in the run's result files.
    col: which progress count marks a completed step, 2 = the benchmark's satisfied subgoals, 1 = the oracle step tracker."""
    lab, n_ep, n_done = {}, 0, 0
    for f in glob.glob(f"{results_root}/**/*_parallel_results.json", recursive=True):
        for task in json.load(open(f))["tasks"]:
            t, off, glen = task["task_id"], task.get("init_offset", 0), task.get("goal_len") or 2
            for k, fin in (task.get("trial_final") or {}).items():
                n_ep += 1; n_done += bool(fin["final_done"])
                rk = fin.get("req_k") or []
                end = [fin["k_end"]] if fin.get("k_end") else []  # count at the episode's end (after the last request)
                for j, e in enumerate(rk):
                    s, kk = e[0], e[col]
                    if mode == "episode":
                        y = bool(fin["final_done"])
                    else:  # the step this request belongs to is completed within the window (the last one: and kept)
                        y = any(e2[col] > kk and e2[0] - s <= window for e2 in rk[j + 1:] + end)
                        y = y and (kk + 1 < glen or col != 2 or bool(fin["final_done"]))
                    lab[(prefix, t, int(k) + off, int(s))] = int(y)
    return lab, n_ep, n_done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rec_dir", default=f"{G}/results/lmem_dsbc/records")
    ap.add_argument("--sets", required=True, help="'prefix:results_root,...' (results_root holds the 51 result files)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", default="step", choices=["step", "episode"])
    ap.add_argument("--window", type=int, default=250, help="policy steps from the request to the step's completion")
    ap.add_argument("--count", default="benchmark", choices=["benchmark", "tracker"],
                    help="progress signal: the benchmark's satisfied subgoals, or the oracle step tracker (lmem_steps.py)")
    ap.add_argument("--epochs", type=int, default=300)
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    lab, files = {}, []
    for prefix, root in (s.split(":", 1) for s in a.sets.split(",")):
        l, n_ep, n_done = labels(root, prefix, a.window, a.label, 2 if a.count == "benchmark" else 1)
        lab.update(l)
        files += [(prefix, f) for f in sorted(glob.glob(f"{a.rec_dir}/{prefix}_t*_tr*_s*.npz"))]
        print(f"{prefix}: {n_ep} episodes ({n_done} successful), {len(l)} logged requests, {sum(l.values())} positive", flush=True)
    dev = "cuda"
    feats = DS.dino_fn(dev)
    X, S, T, Z, E, pad, n_unl = [], [], [], [], [], None, 0
    for prefix, f in files:
        m = re.search(r"_t(\d+)_tr(\d+)_s(\d+)\.npz$", f); key = (prefix, int(m.group(1)), int(m.group(2)), int(m.group(3)))
        if key not in lab:
            n_unl += 1
            continue
        if not lab[key]:
            continue
        d = np.load(f, allow_pickle=True)
        X.append(feats(d["image"])); Z.append(torch.from_numpy(d["z"]).float())
        S.append(np.concatenate([d[k].reshape(-1) for k in sorted(k for k in d.files if k.startswith("state_"))]))
        T.append(DS.task_index(str(d["task"]), LS.CLASSES))
        E.append(key[:3])
        if pad is None and d["pad"].size:
            pad = d["pad"].astype(bool)
    cmds = np.bincount(T, minlength=len(LS.CLASSES) + 1)
    print(f"{len(files)} recorded requests, {n_unl} without a logged step count, {len(E)} positives; "
          f"by command: {[(LS.CLASSES[i] if i < len(LS.CLASSES) else 'other', int(c)) for i, c in enumerate(cmds) if c]}", flush=True)
    ck = DS.fit_noise_policy(X, S, T, Z, E, pad, a.epochs, len(LS.CLASSES), dev)
    torch.save({**ck, "tasks": list(LS.CLASSES), "label": a.label, "window": a.window, "count": a.count}, a.out)
    print("saved", a.out, flush=True)


if __name__ == "__main__":
    main()
