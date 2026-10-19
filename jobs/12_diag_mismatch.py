"""Why is the frozen released policy's FM loss ~2.3 on LaWAM's LIBERO set? Probe data-convention mismatches.

1) Released dataset_stats vs stats of our data (per-dim action/state ranges, gripper sign).
2) Frozen policy FM loss on fixed batches under: as-is, 180-degree flip (both cams / main only / wrist only),
   eval-mode processor (no augmentation). Same RNG seed per variant so noise/time samples are identical.
"""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import importlib

T = importlib.import_module("07_train_hybrid")  # reuse setup helpers (chdirs into GalaxeaVLA)

from omegaconf import OmegaConf  # noqa: E402

dev = torch.device("cuda")
torch.manual_seed(0)
rel = json.load(open(T.STATS))
ours = json.load(open(f"{T.PROJECT}/data/stats/libero_merged_stats.json"))


def show(stats, tag):
    s = stats.get("libero", stats)
    def walk(d, path):
        if isinstance(d, dict):
            for k, v in d.items():
                walk(v, path + [str(k)])
        else:
            arr = np.array(d, dtype=float).reshape(-1)
            if any(x in path[-1] for x in ("q01", "q99", "mean", "min", "max", "std")):
                print(f"[{tag}] {'/'.join(path)} = {np.round(arr[:24], 3).tolist()}")
    walk(s, [])
    return
    for k in ("action", "state"):
        if k not in s:
            continue
        for part, v in s[k].items():
            if not isinstance(v, dict):
                continue
            arr = {m: np.round(np.array(v[m]).reshape(-1)[:8], 3).tolist() for m in ("q01", "q99", "mean", "min", "max") if m in v}
            print(f"[{tag}] {k}.{part}: {arr}")


show(rel, "released")
show(ours, "ours")

cfg = T.load_config_from_run_dir(T.find_run_dir(T.CKPT), T.CKPT, [])
if cfg.get("eval_embodiment") and "embodiment_datasets" in cfg.data:
    T.filter_embodiment(cfg, cfg.eval_embodiment)
policy, _ = T.g05_setup(cfg, device="cuda")
policy.eval().requires_grad_(False)
OmegaConf.set_struct(cfg, False)
cfg.data = OmegaConf.load(T.DATA_CFG)
ds = T.instantiate_dataset(cfg, is_training_set=True)
proc = T.build_processors(cfg)
proc.set_normalizer_from_stats(T.load_dataset_stats_from_json(T.STATS))
ds.set_processor(proc)
collate = T.partial(T.collate_fn_pad_sequences, padding_input_id=proc.pad_token_id)
rng = np.random.RandomState(0)
idx = rng.choice(len(ds), 96, replace=False)


def batches(eval_mode):
    if eval_mode:
        proc.eval()
    else:
        proc.train()
    out = []
    for s in range(0, len(idx), 32):
        out.append(collate([ds[int(i)] for i in idx[s:s + 32]]))
    return out


def fm_loss(bs, flip=None, grip=False):
    losses = []
    for k, b in enumerate(bs):
        b = T.to_dev(b, dev)
        acts = b["action"].clone()
        if grip:  # invert gripper semantics in normalized space ([0,1] -> [-1,1] is symmetric)
            acts[..., 19] = -acts[..., 19]
        pv = {kk: vv.clone() for kk, vv in b["pixel_values"].items()}
        for cam in pv:
            if flip == "all" or (flip == "main" and cam == "exterior") or (flip == "wrist" and cam != "exterior"):
                pv[cam] = pv[cam].flip(-1, -2)
        torch.manual_seed(1000 + k)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            loss, _ = policy.forward_train(samples=b["samples"], pixel_values=pv, actions=acts,
                                           action_pad_masks=b["action_is_pad"], action_dim_is_pad=b.get("action_dim_is_pad"))
        losses.append(loss.item())
    return float(np.mean(losses))


b_eval = batches(True)
b0 = T.to_dev(b_eval[0], dev)
a = b0["action"].float()
real = [10, 11, 12, 13, 14, 15, 19]
print("normalized action real dims", real, "mean", a[:, :, real].mean((0, 1)).cpu().numpy().round(3),
      "min", a[:, :, real].amin((0, 1)).cpu().numpy().round(3), "max", a[:, :, real].amax((0, 1)).cpu().numpy().round(3))
print("gripper(dim19) normalized values (first step):", sorted(set(a[:, 0, 19].cpu().numpy().round(2).tolist())))
print("action_dim_is_pad[0]", b0.get("action_dim_is_pad")[0].int().tolist() if b0.get("action_dim_is_pad") is not None else None)
s0 = b0["samples"][0]
print("sample keys", list(s0.keys()))
if False:
    print("proprio sample", {k: (np.round(np.asarray(v, dtype=float).reshape(-1)[:10], 3).tolist() if hasattr(v, "__len__") else v)
                            for k, v in s0["proprio"].items() if k != "embodiment"})
res = {}
for name, fl, gr in [("as_is", None, False), ("grip_inverted", None, True), ("flip_all", "all", False),
                     ("flip_all+grip_inverted", "all", True), ("flip_main", "main", False), ("flip_wrist", "wrist", False)]:
    res[name] = fm_loss(b_eval, fl, gr)
    print(f"FM loss eval-proc {name:10s}: {res[name]:.4f}", flush=True)
res["as_is_train_proc"] = fm_loss(batches(False), None)
print(f"FM loss train-proc as_is    : {res['as_is_train_proc']:.4f}")
json.dump(res, open(f"{T.G}/results/diag_mismatch.json", "w"), indent=2)
