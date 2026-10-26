"""One PrefPI step for G0.5 (Rho, Kim, Xu, Ha, "PrefPI: Preference-Guided Steering into Out-of-Distribution Behaviors",
arXiv:2609.40165): preference-conditioned generative modelling + classifier-free guidance as policy iteration.

Given the rollouts of iterations 0..k (95_frs_client.py --record_dir: successful episodes only), rank each iteration's
successful episodes by the preference score and
  * unconditional branch (plain instruction): every successful episode so far (the accumulated set D_acc);
  * preference-conditioned branch ("[cfg] " + instruction): the top 40% of each iteration's successful episodes, kept in
    a FIFO buffer of the 40 most recent;
then fine-tune G0.5 from iteration k's weights on both, sampled unconditional : conditional = 0.4 : 1 by episode,
frames uniform within an episode (paper Table 2). The next iteration's rollouts run the result with CFG w = 5 between
the two prompts (94_frs_serve.py cfg rows: v = v_plain + w (v_pref - v_plain)).
Scores: speed = episode length in policy steps (lower preferred); height = highest point of the bowl above its start
(higher preferred; the end-effector height while the gripper is closed when the env reports no bowl position).
Trainable part: G0.5's action expert (635M, fp32 master weights; the paper LoRA-tunes pi0.5's language model and
action expert and fully tunes SigLIP). The vision-language backbone stays frozen: the preference condition reaches the
expert through its attention to the prompt's prefix tokens. No base demonstration data is used, as in the paper.
Data path: G0.5's own LIBERO dataset object with its frame source swapped (07b_train_lmem.make_recovery_dataset), so
normalisation, chunking (32 steps) and the processor are those of training.
"""
import argparse
import glob
import importlib
import json
import os
import random
import sys
import time
from functools import partial

import numpy as np
import torch

JOBS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, JOBS)
T = importlib.import_module("07b_train_lmem")  # chdir to GalaxeaVLA, G0.5 imports, t4_fp16, dataset helpers


def score(f, obj):
    z = np.load(f, allow_pickle=True)
    if obj == "speed":
        return -float(len(z["action"]))  # higher score = preferred
    bowl = [k for k in z.files if k.startswith("objz_") and "bowl" in k]
    if bowl:
        h = z[bowl[0]].astype(np.float64)
        return float(np.nanmax(h) - h[0])
    closed = z["action"][:, -1] < 0.5  # G0.5 convention: 1 open, 0 closed
    eef = z["state"][:, 2]
    return float(eef[closed].max() - eef[0]) if closed.any() else 0.0


class Frames:
    """LeRobot-format samples (the merged LIBERO set's keys) from recorded episodes, each with its own prompt."""

    def __init__(self, episodes, cache_dir):
        os.makedirs(cache_dir, exist_ok=True)
        self.eps, self.prompt, self.index, self.ep_of = [], [], [], []
        for e, (f, prompt) in enumerate(episodes):
            base = f"{cache_dir}/" + os.path.relpath(f, "/").replace("/", "_")[:-4]
            if not os.path.exists(base + "_wrist.npy"):
                z = np.load(f, allow_pickle=True)
                np.save(base + "_img.npy", z["image"]); np.save(base + "_wrist.npy", z["wrist_image"])
                np.save(base + "_state.npy", z["state"]); np.save(base + "_action.npy", z["action"])
            n = len(np.load(base + "_action.npy", mmap_mode="r"))
            self.eps.append(base); self.prompt.append(prompt)
            self.index += [(e, t) for t in range(n)]; self.ep_of += [e] * n
        self._mm = {}

    def _arr(self, e, k):
        if (e, k) not in self._mm:
            self._mm[(e, k)] = np.load(f"{self.eps[e]}_{k}.npy", mmap_mode="r")
        return self._mm[(e, k)]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        e, t = self.index[i]
        img, wr, st, act = (self._arr(e, k) for k in ("img", "wrist", "state", "action"))
        n = len(act); fut = np.arange(t, t + 32)
        return {
            "observation.images.image": torch.from_numpy(np.ascontiguousarray(img[t])).permute(2, 0, 1).float().div(255),
            "observation.images.wrist_image": torch.from_numpy(np.ascontiguousarray(wr[t])).permute(2, 0, 1).float().div(255),
            "observation.state": torch.from_numpy(np.asarray(st[t:t + 1], dtype=np.float32)),
            "action": torch.from_numpy(np.asarray(act[np.minimum(fut, n - 1)], dtype=np.float32)),
            "timestamp": torch.tensor(t / 20.0), "frame_index": torch.tensor(t), "episode_index": torch.tensor(200000 + e),
            "index": torch.tensor(i), "task_index": torch.tensor(0),
            "observation.images.image_is_pad": torch.zeros(1, dtype=torch.bool),
            "observation.images.wrist_image_is_pad": torch.zeros(1, dtype=torch.bool),
            "observation.state_is_pad": torch.zeros(1, dtype=torch.bool),
            "action_is_pad": torch.from_numpy(fut >= n), "chunked_task_index": None, "task": self.prompt[e],
            "step_is_qualified": True, "dataset_index": torch.tensor(0),
        }


def make_dataset(full_dataset, frames):
    """As 07b_train_lmem.make_recovery_dataset, for an arbitrary frame source."""
    import copy
    inner = full_dataset.datasets[0]

    class PrefDataset(type(inner)):
        def __len__(self):
            return len(frames)

        def _locate_sample(self, idx):
            return f"prefpi_frame={idx}"

        def _resample_random_idx(self):
            return random.randrange(len(frames))

    ds = copy.copy(inner); ds.__class__ = PrefDataset
    ds.multi_dataset = frames; ds._start_idx = 0
    if hasattr(ds, "_overfit_indices"):
        del ds._overfit_indices
    ds.processor.train()
    return ds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollouts", required=True, help="comma-separated rollout dirs of iterations 0..k (each with rec/*.npz)")
    ap.add_argument("--obj", required=True, choices=["speed", "height"])
    ap.add_argument("--init_expert", default="", help="iteration k's action expert ('' = the released checkpoint)")
    ap.add_argument("--out", required=True, help="output action-expert state dict (fp16)")
    ap.add_argument("--prefix", default="[cfg] ")
    ap.add_argument("--top", type=float, default=0.4)
    ap.add_argument("--buffer", type=int, default=40)
    ap.add_argument("--uncond_ratio", type=float, default=0.4)
    ap.add_argument("--anchor_frac", type=float, default=0.0,
                    help="share of samples from the task's original demonstrations (plain prompt); 0 = rollouts only, as the paper")
    ap.add_argument("--steps", type=int, default=1500,
                    help="optimizer steps per iteration (paper: 3000 at batch 12 on an L40S; the first G0.5 iteration ran 3000, "
                         "later ones 1500 from the previous iteration's weights: 2.7 s/step on a T4)")
    ap.add_argument("--batch", type=int, default=4, help="micro-batch")
    ap.add_argument("--accum", type=int, default=3, help="gradient accumulation (effective batch = batch x accum)")
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--optim", default="adamw8bit", choices=["adamw8bit", "adamw"])
    ap.add_argument("--gdn", default="auto", choices=["auto", "torch", "fla"], help="linear-attention kernels (auto: torch on T4)")
    a = ap.parse_args()
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    out_dir = os.path.dirname(os.path.abspath(a.out)); os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()

    # ---- preference selection
    acc, pref, stats = [], [], []
    for k, d in enumerate(a.rollouts.split(",")):
        files = sorted(glob.glob(f"{d}/rec/*.npz"))
        sc = sorted(((score(f, a.obj), f) for f in files), reverse=True)
        n_top = int(round(a.top * len(sc)))
        acc += [f for _, f in sc]; pref += [f for _, f in sc[:n_top]]
        res = glob.glob(f"{d}/*_parallel_results.json")
        succ = json.load(open(res[0]))["tasks"][0] if res else {}
        stats.append({"iter": k, "dir": d, "successes": len(sc), "episodes": succ.get("total_episodes"),
                      "score_mean": float(np.mean([s for s, _ in sc])) if sc else None,
                      "score_median": float(np.median([s for s, _ in sc])) if sc else None,
                      "pref_threshold": sc[n_top - 1][0] if n_top else None})
    pref = pref[-a.buffer:]
    task = str(np.load(acc[0], allow_pickle=True)["task_description"])
    episodes = [(f, task) for f in acc] + [(f, a.prefix + task) for f in pref]
    print(json.dumps({"event": "selection", "obj": a.obj, "iterations": stats, "acc": len(acc), "pref_buffer": len(pref),
                      "task": task}), flush=True)

    # ---- policy: frozen released G0.5 (fp16 on T4/V100), trainable action expert (fp32 master weights)
    cfg = T.load_config_from_run_dir(T.find_run_dir(T.CKPT), T.CKPT, [])
    if cfg.get("eval_embodiment") and "embodiment_datasets" in cfg.data:
        T.filter_embodiment(cfg, cfg.eval_embodiment)
    policy, _ = T.g05_setup(cfg, device="cuda")
    policy.eval().requires_grad_(False)
    expert = next(m for n, m in policy.named_modules() if n.endswith("action_expert"))
    if T.t4_fp16.T4_MODE:
        for _name, _mod in list(sys.modules.items()):
            if "g05" in _name:
                for _attr in ("_flash_attn_varlen", "flash_attn_func"):
                    if getattr(_mod, _attr, None) is not None:
                        setattr(_mod, _attr, None)
        policy.half()
        torch.cuda.empty_cache()

        def _cast_in(mod, args):
            if args and torch.is_tensor(args[0]) and args[0].is_floating_point() and args[0].dtype != mod.weight.dtype:
                return (args[0].to(mod.weight.dtype),) + tuple(args[1:])
        for _m in policy.modules():
            if isinstance(_m, (torch.nn.Linear, torch.nn.Conv2d, torch.nn.Conv3d, torch.nn.LayerNorm)) and getattr(_m, "weight", None) is not None:
                _m.register_forward_pre_hook(_cast_in)
    if a.gdn == "torch" or (a.gdn == "auto" and T.t4_fp16.T4_MODE):
        # G0.5's linear-attention layers (Qwen3.5 gated delta rule) use FLA's Triton chunk kernels, which fail to
        # compile on a T4 from a clean Triton cache; GalaxeaVLA's own PyTorch implementation computes the same thing
        from g05.models.g05.qwen35 import gated_deltanet as GDN
        n_gdn = 0
        for m in policy.modules():
            if callable(getattr(m, "chunk_gated_delta_rule", None)) and not isinstance(m, torch.nn.ModuleList):
                m.chunk_gated_delta_rule = GDN.torch_chunk_gated_delta_rule; n_gdn += 1
        print(json.dumps({"event": "gdn_backend", "torch_layers": n_gdn}), flush=True)
    if a.init_expert:
        sd = torch.load(a.init_expert, map_location="cpu")
        expert.load_state_dict({k: v.float() for k, v in sd.items()}, strict=True)
    expert.float().requires_grad_(True).train()
    params = list(expert.parameters())

    # ---- data: G0.5's LIBERO dataset object as the template (normalisation = released stats)
    from omegaconf import OmegaConf
    OmegaConf.set_struct(cfg, False)
    cfg.data = OmegaConf.load(T.DATA_CFG)
    full = T.instantiate_dataset(cfg, is_training_set=True)
    proc = T.build_processors(cfg)
    proc.set_normalizer_from_stats(T.load_dataset_stats_from_json(T.STATS))
    full.set_processor(proc)
    frames = Frames(episodes, f"{os.environ.get('G05_ROOT', '/export/home/kaziz/g05')}/results/prefpi/frame_cache")
    ds = make_dataset(full, frames)
    n_unc = len(acc)
    ep_w = np.array([a.uncond_ratio / max(n_unc, 1)] * n_unc + [1.0 / max(len(pref), 1)] * len(pref))
    ep_len = np.bincount(frames.ep_of, minlength=len(episodes))
    w = torch.tensor(ep_w[frames.ep_of] / ep_len[frames.ep_of], dtype=torch.double)
    if a.anchor_frac > 0:  # original demonstrations of the same task, plain prompt: anchors the policy against self-training drift
        import pandas as pd
        dsd = f"{os.environ.get('G05_ROOT', '/export/home/kaziz/g05')}/data/libero_merged_g05grip"
        dd = pd.concat([pd.read_parquet(f, columns=["index", "task_index"]) for f in sorted(glob.glob(f"{dsd}/data/*/*.parquet"))]).sort_values("index")
        tp = pd.read_parquet(f"{dsd}/meta/tasks.parquet").reset_index()
        tcol = [c for c in tp.columns if tp[c].dtype == object][0]
        nrm = lambda s: " ".join(str(s).lower().replace("_", " ").split())  # noqa: E731
        tid = {nrm(r[tcol]): int(r["task_index"]) for _, r in tp.iterrows()}[nrm(task)]
        rows = dd["index"].values[dd["task_index"].values == tid]
        rows = rows[rows < len(full)].tolist()
        ds = torch.utils.data.ConcatDataset([ds, torch.utils.data.Subset(full, rows)])
        w = torch.cat([w / w.sum() * (1 - a.anchor_frac), torch.full((len(rows),), a.anchor_frac / len(rows), dtype=torch.double)])
        print(json.dumps({"event": "anchor", "demo_frames": len(rows), "frac": a.anchor_frac}), flush=True)
    sampler = torch.utils.data.WeightedRandomSampler(w, num_samples=(a.steps * a.accum + 1) * a.batch, replacement=True,
                                                     generator=torch.Generator().manual_seed(a.seed))
    _collate = partial(T.collate_fn_pad_sequences, padding_input_id=proc.pad_token_id)

    def _fill(items):  # recorded frames lack some metadata keys of the base set: fill from a donor (as 07b)
        keys = set().union(*[it.keys() for it in items])
        for k in keys:
            donor = next((it[k] for it in items if k in it), None)
            for it in items:
                if k not in it:
                    it[k] = donor
            sub = [it[k] for it in items]
            if sub and all(isinstance(x, dict) for x in sub):
                _fill(sub)

    loader = torch.utils.data.DataLoader(ds, batch_size=a.batch, sampler=sampler, num_workers=a.workers,
                                         collate_fn=lambda b: (_fill(b), _collate(b))[1], drop_last=True,
                                         persistent_workers=True, prefetch_factor=2, pin_memory=True)
    print(json.dumps({"event": "setup", "frames": len(frames), "episodes": len(episodes), "uncond_eps": n_unc,
                      "cond_eps": len(pref), "trainable": sum(p.numel() for p in params), "init": a.init_expert or "released",
                      "t": round(time.time() - t0)}), flush=True)

    if a.optim == "adamw8bit":  # the G0.5 env's torch has no sm70 kernels, so training runs on a 16 GB T4
        import bitsandbytes as bnb
        opt = bnb.optim.AdamW8bit(params, lr=a.lr, betas=(0.9, 0.95), weight_decay=1e-10)
    else:
        opt = torch.optim.AdamW(params, lr=a.lr, betas=(0.9, 0.95), weight_decay=1e-10)
    scaler = torch.cuda.amp.GradScaler(enabled=T.t4_fp16.T4_MODE)
    dev = torch.device("cuda")
    step, batches = 0, iter(loader)
    while step < a.steps:  # effective batch = batch x accum (the paper's 12 = 4 x 3 on a 16 GB T4)
        for g in opt.param_groups:
            g["lr"] = a.lr * min(1.0, (step + 1) / a.warmup)
        opt.zero_grad(set_to_none=True)
        loss_sum = 0.0
        for _ in range(a.accum):
            batch = T.to_dev(next(batches), dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss, _ = policy.forward_train(samples=batch["samples"], pixel_values=batch["pixel_values"], actions=batch["action"],
                                               action_pad_masks=batch["action_is_pad"], action_dim_is_pad=batch.get("action_dim_is_pad"))
            scaler.scale(loss / a.accum).backward()
            loss_sum += loss.item() / a.accum
        scaler.unscale_(opt)
        gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
        scaler.step(opt); scaler.update()
        step += 1
        if step == 1 or step % 100 == 0 or step == a.steps:
            print(json.dumps({"event": "train", "step": step, "loss": round(loss_sum, 5), "gnorm": round(gn.item(), 4),
                              "max_mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2), "t": round(time.time() - t0)}), flush=True)
    torch.save({k: v.half() for k, v in expert.state_dict().items()}, a.out)
    json.dump({"obj": a.obj, "iterations": stats, "acc": len(acc), "pref_buffer": len(pref), "steps": step,
               "init": a.init_expert, "out": a.out}, open(a.out[:-3] + "_selection.json", "w"), indent=1)
    print(json.dumps({"event": "done", "out": a.out, "t": round(time.time() - t0)}), flush=True)


if __name__ == "__main__":
    main()
