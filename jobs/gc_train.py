"""Train the goal-conditioned adapter (gc_hybrid.py) on top of the frozen released G0.5 FM checkpoint.

pi0.7 recipe, on pooled DINOv3 features of the main camera (results/feat_cache, 22_cache_feats.py):
  * a goal is present in --goal_p of examples (0.75: Kuan Yi Wang's pi0.7 rebuild found 25% hurt, 75% helped);
  * the goal is the episode's last frame with --goal_end_p (0.25), otherwise a frame uniform 0..--goal_max_ahead
    frames ahead (80 = 4 s at 20 Hz), clamped to the episode end;
  * when a goal is present the instruction is blanked with --instr_drop (0.3, pi0.7's subtask dropout);
  * no goal = goal tokens masked out of the adapter's cross-attention.
Episodes with episode_index % 20 == 0 are held out for the offline check at the end (frozen-noise FM loss under
instruction / blank x real goal / no goal / shuffled goal).

--unfreeze_expert also fine-tunes G0.5's action expert (635M parameters, LR x --expert_lr_scale), as pi0.7 trains its
whole policy; with --goal_p 0 the same run is the matched no-goal control. Saves expert_last.pt next to the adapter.

Uses 07b_train_lmem.py as a library (same policy loading, fp16 T4/V100 path, data, normalisation).
"""
import argparse
import copy
import importlib
import json
import os
import random
import sys
import time
from functools import partial

import numpy as np
import torch
import torch.nn.functional as F

JOBS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, JOBS)
T = importlib.import_module("07b_train_lmem")  # chdir to GalaxeaVLA, imports G0.5, t4_fp16
import gc_hybrid as GC  # noqa: E402
import lawam_hybrid as H  # noqa: E402

t4_fp16 = T.t4_fp16
G = T.G


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--init_ckpt", default=f"{G}/results/hybrid/F01_s0/hybrid_last.pt")
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--warmup_frac", type=float, default=0.05)
    p.add_argument("--flow_samples", type=int, default=4)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--heldout_mod", type=int, default=20)
    p.add_argument("--goal_p", type=float, default=0.75)
    p.add_argument("--goal_end_p", type=float, default=0.25)
    p.add_argument("--goal_max_ahead", type=int, default=80)
    p.add_argument("--instr_drop", type=float, default=0.3)
    p.add_argument("--goal_input", default="abs", choices=["abs", "delta"],
                   help="abs: goal features; delta: [goal, goal - current frame] (gc_hybrid.GoalAdapter)")
    p.add_argument("--eval_n", type=int, default=768)
    p.add_argument("--eval_ahead", type=int, default=32, help="offline check: goal = frame this far ahead (1.6 s)")
    p.add_argument("--save_every", type=int, default=1000)
    p.add_argument("--unfreeze_expert", action="store_true", help="also fine-tune G0.5's action expert (fp32 master weights)")
    p.add_argument("--expert_lr_scale", type=float, default=0.2, help="expert LR = lr x this")
    p.add_argument("--expert_init", default=None, help="start the expert from these weights (expert_last.pt), e.g. for --eval_only")
    p.add_argument("--eval_only", default=None, help="skip training; run the offline check on this checkpoint")
    p.add_argument("--out", required=True)
    return p.parse_args()


class GoalCache:
    """Pooled DINOv3 features of every frame (row-aligned with the training dataset)."""

    def __init__(self):
        c = f"{G}/results/feat_cache"
        m = np.load(f"{c}/meta.npz")
        self.ep, self.fr, self.L = m["episode_index"], m["frame_index"], m["ep_length"]
        self.remaining = self.L - 1 - self.fr
        self.feats = np.load(f"{c}/feats.npy", mmap_mode="r")

    def get(self, rows):
        rows = np.asarray(rows)
        order = np.argsort(rows)
        got = np.asarray(self.feats[rows[order]])
        out = np.empty_like(got)
        out[order] = got
        return torch.from_numpy(out)

    def sample_targets(self, rows, rng, a):
        rem = self.remaining[rows]
        end = rng.random(len(rows)) < a.goal_end_p
        k = rng.integers(0, a.goal_max_ahead + 1, len(rows))
        return rows + np.where(end, rem, np.minimum(k, rem))


def main():
    a = parse_args()
    os.makedirs(a.out, exist_ok=True)
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    dev = torch.device("cuda")
    log_f = open(f"{a.out}/train_log.jsonl", "a")
    t0 = time.time()

    def log(d):
        d["t"] = round(time.time() - t0, 1)
        print(json.dumps(d), flush=True)
        log_f.write(json.dumps(d) + "\n"); log_f.flush()

    # ---- frozen released policy, loaded and cast exactly as in 07b_train_lmem.py
    cfg = T.load_config_from_run_dir(T.find_run_dir(T.CKPT), T.CKPT, [])
    if cfg.get("eval_embodiment") and "embodiment_datasets" in cfg.data:
        T.filter_embodiment(cfg, cfg.eval_embodiment)
    policy, _ = T.g05_setup(cfg, device="cuda")
    policy.eval().requires_grad_(False)
    expert = next(m for n, m in policy.named_modules() if n.endswith("action_expert"))
    if t4_fp16.T4_MODE:
        for _name, _mod in list(sys.modules.items()):
            if "g05" in _name:
                for _attr in ("_flash_attn_varlen", "flash_attn_func"):
                    if getattr(_mod, _attr, None) is not None:
                        setattr(_mod, _attr, None)
        policy.half()
        if a.unfreeze_expert:
            expert.float()  # fp32 master weights for the trained expert (as in 07b_train_lmem.py)
        torch.cuda.empty_cache()

        def _cast_in(mod, args):
            if args and torch.is_tensor(args[0]) and args[0].is_floating_point() and args[0].dtype != mod.weight.dtype:
                return (args[0].to(mod.weight.dtype),) + tuple(args[1:])
        for _m in policy.modules():
            if isinstance(_m, (torch.nn.Linear, torch.nn.Conv2d, torch.nn.Conv3d, torch.nn.LayerNorm)) and getattr(_m, "weight", None) is not None:
                _m.register_forward_pre_hook(_cast_in)
    n_set = 0
    for m in [policy] + list(policy.modules()):
        for v in list(vars(m).values()):
            if hasattr(v, "num_flow_samples") and not isinstance(v, torch.nn.Module):
                v.num_flow_samples = a.flow_samples; n_set += 1
        if hasattr(m, "num_flow_samples") and isinstance(getattr(m, "num_flow_samples"), int):
            m.num_flow_samples = a.flow_samples; n_set += 1

    hybrid = GC.build_goal_hybrid(ckpt=a.eval_only or a.init_ckpt, seed=a.seed, goal_input=None if a.eval_only else a.goal_input)
    H.attach(policy, hybrid)
    for n, p in hybrid.named_parameters():
        p.requires_grad_(n.startswith("adapter."))
    hybrid.prior.eval()
    hybrid.adapter.train()
    params = [p for p in hybrid.parameters() if p.requires_grad]
    goal_params = [p for n, p in hybrid.named_parameters() if ".res_goal." in n or n.endswith("goal_emb")]
    expert_params = []
    if a.unfreeze_expert:
        if a.expert_init:
            sd = torch.load(a.expert_init, map_location="cpu")
            dt = next(expert.parameters()).dtype
            expert.load_state_dict({k: v.to(dt) for k, v in sd.items()}, strict=True)
        expert.requires_grad_(True); expert.train()
        expert_params = list(expert.parameters())
    log({"event": "setup", "n_trainable": sum(p.numel() for p in params) + sum(p.numel() for p in expert_params),
         "n_expert": sum(p.numel() for p in expert_params), "n_goal_branch": sum(p.numel() for p in goal_params),
         "flow_samples_set_on": n_set, "t4_mode": bool(t4_fp16.T4_MODE), "gpu": torch.cuda.get_device_name(0), "args": vars(a)})

    # ---- data: released stats, LaWAM's merged LIBERO set, held-out episodes removed
    T.OmegaConf.set_struct(cfg, False)
    cfg.data = T.OmegaConf.load(T.DATA_CFG)
    full = T.instantiate_dataset(cfg, is_training_set=True)
    proc = T.build_processors(cfg)
    proc.set_normalizer_from_stats(T.load_dataset_stats_from_json(T.STATS))
    full.set_processor(proc)
    collate = partial(T.collate_fn_pad_sequences, padding_input_id=proc.pad_token_id)
    gc = GoalCache()
    rowmap = T.load_rowmap()
    assert np.array_equal(rowmap["episode_index"].numpy(), gc.ep) and np.array_equal(rowmap["frame_index"].numpy(), gc.fr), \
        "feat_cache rows != dataset rows"
    train_idx = np.nonzero(gc.ep % a.heldout_mod != 0)[0]
    train_idx = train_idx[train_idx < len(full)]
    dataset = torch.utils.data.Subset(full, train_idx.tolist())
    log({"event": "data", "n_train_frames": len(dataset), "n_frames": len(full),
         "heldout_episodes": int(len(np.unique(gc.ep[gc.ep % a.heldout_mod == 0])))})

    if not a.eval_only:
        gen = torch.Generator().manual_seed(a.seed)
        loader = T.DataLoader(dataset, batch_size=a.batch, sampler=T.RandomSampler(dataset, generator=gen),
                              num_workers=a.workers, collate_fn=collate, drop_last=True, persistent_workers=True,
                              prefetch_factor=2, pin_memory=True)
        groups = [{"params": params, "lr_scale": 1.0}]
        if expert_params:
            groups.append({"params": expert_params, "lr_scale": a.expert_lr_scale})
        opt = torch.optim.AdamW(groups, lr=a.lr, weight_decay=0.01, betas=(0.9, 0.95))
        all_params = params + expert_params
        scaler = torch.cuda.amp.GradScaler(enabled=t4_fp16.T4_MODE)
        step, it, checked = 0, iter(loader), False
        stats = {"goal_on": 0, "instr_blank": 0, "goal_end": 0, "n": 0, "ahead_sum": 0}
        while step < a.steps:
            try:
                batch = next(it)
            except StopIteration:
                it = iter(loader); batch = next(it)
            rows = T.batch_indices(batch).numpy()
            batch = T.to_dev(batch, dev)
            b = len(rows)
            if not checked:  # the cached features of each row must be the frame G0.5 sees in this sample
                with torch.no_grad():
                    u = H.dino_from_qwen_pixels(hybrid.lawm, hybrid._main_camera(batch["pixel_values"]))
                    u = F.avg_pool2d(u.view(b, 16, 16, -1).permute(0, 3, 1, 2), 2).permute(0, 2, 3, 1).reshape(b, 64, -1)
                    c = gc.get(rows).to(dev).float()
                    cos = F.cosine_similarity(u.flatten(1), c.flatten(1), dim=-1)
                    cos_sh = F.cosine_similarity(u.flatten(1), c.roll(1, 0).flatten(1), dim=-1)
                log({"event": "feat_alignment", "cos_matched": cos.mean().item(), "cos_min": cos.min().item(),
                     "cos_shuffled": cos_sh.mean().item(), "same_episode_in_batch": int(len(np.unique(gc.ep[rows])) < b)})
                if cos.mean().item() < cos_sh.mean().item() + 0.02:
                    raise RuntimeError("feat_cache rows do not match the frames G0.5 sees: goals would be misaligned")
                checked = True
            on = rng.random(b) < a.goal_p
            tgt = gc.sample_targets(rows, rng, a)
            hybrid.goal_feats = gc.get(tgt).to(dev)
            hybrid.goal_on = torch.from_numpy(on)
            blank = on & (rng.random(b) < a.instr_drop)
            for i, smp in enumerate(batch["samples"]):
                if blank[i]:
                    smp["command"] = ""
            stats["n"] += b; stats["goal_on"] += int(on.sum()); stats["instr_blank"] += int(blank.sum())
            stats["goal_end"] += int((on & (tgt - rows == gc.remaining[rows])).sum()); stats["ahead_sum"] += int((tgt - rows)[on].sum())
            lr = T.lr_at(step, a.steps, a.lr, a.warmup_frac)
            for g in opt.param_groups:
                g["lr"] = lr * g.get("lr_scale", 1.0)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss, _ = policy.forward_train(samples=batch["samples"], pixel_values=batch["pixel_values"],
                                               actions=batch["action"], action_pad_masks=batch["action_is_pad"],
                                               action_dim_is_pad=batch.get("action_dim_is_pad"))
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            g_goal = torch.norm(torch.stack([p.grad.norm() for p in goal_params if p.grad is not None])).item() \
                if any(p.grad is not None for p in goal_params) else 0.0
            gnorm = torch.nn.utils.clip_grad_norm_(all_params, 1.0)
            scaler.step(opt); scaler.update()
            step += 1
            if step % 25 == 0 or step == 1:
                log({"event": "train", "step": step, "loss": round(loss.item(), 5), "lr": lr, "gnorm": round(gnorm.item(), 4),
                     "gnorm_goal_branch": round(g_goal, 5),
                     "frac_goal": round(stats["goal_on"] / stats["n"], 3), "frac_blank": round(stats["instr_blank"] / stats["n"], 3),
                     "frac_goal_is_end": round(stats["goal_end"] / max(1, stats["goal_on"]), 3),
                     "mean_ahead_frames": round(stats["ahead_sum"] / max(1, stats["goal_on"]), 1)})
            if step % a.save_every == 0 or step == a.steps:
                st = {"state": hybrid.trainable_state(), "step": step, "args": vars(a)}
                torch.save(st, f"{a.out}/hybrid_step{step}.pt")
                torch.save(st, f"{a.out}/hybrid_last.pt")
                if expert_params:
                    torch.save({k: v.half() for k, v in expert.state_dict().items()}, f"{a.out}/expert_last.pt")
        hybrid.goal_feats = hybrid.goal_on = None
        if expert_params:
            expert.eval()
    evaluate(a, policy, hybrid, full, proc, collate, gc, dev, log)
    log({"event": "done"})


@torch.no_grad()
def evaluate(a, policy, hybrid, dataset, proc, collate, gc, dev, log):
    """Held-out episodes: frozen-noise FM loss (lower is better) with the goal 1.6 s ahead real / absent / shuffled."""
    hybrid.adapter.eval()
    if hasattr(proc, "eval"):
        proc.eval()
    rng = np.random.RandomState(7)
    cand = np.nonzero((gc.ep % a.heldout_mod == 0) & (gc.remaining >= a.eval_ahead))[0]
    cand = cand[cand < len(dataset)]
    rows_all = rng.choice(cand, min(a.eval_n, len(cand)), replace=False)
    conds = {"instr_nogoal": (False, None), "instr_goal": (False, "real"), "instr_shuffled_goal": (False, "shuffled"),
             "blank_nogoal": (True, None), "blank_goal": (True, "real"), "blank_shuffled_goal": (True, "shuffled")}
    totals = {k: [] for k in conds}
    for s in range(0, len(rows_all), 32):
        base = collate([dataset[int(r)] for r in rows_all[s:s + 32]])
        rows = T.batch_indices(base).numpy()  # trust what the dataset returned
        goal = gc.get(rows + np.minimum(a.eval_ahead, gc.remaining[rows])).to(dev)
        perm = np.roll(np.arange(len(rows)), 1)
        for name, (blank, mode) in conds.items():
            b = T.to_dev(copy.deepcopy(base), dev)
            if blank:
                for smp in b["samples"]:
                    smp["command"] = ""
            hybrid.goal_feats = goal if mode != "shuffled" else goal[torch.from_numpy(perm).to(dev)]
            hybrid.goal_on = torch.full((len(rows),), mode is not None)
            torch.manual_seed(10_000 + s)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss, _ = policy.forward_train(samples=b["samples"], pixel_values=b["pixel_values"], actions=b["action"],
                                               action_pad_masks=b["action_is_pad"], action_dim_is_pad=b.get("action_dim_is_pad"))
            totals[name].append(loss.item())
    hybrid.goal_feats = hybrid.goal_on = None
    res = {k: float(np.mean(v)) for k, v in totals.items()}
    res["n"] = int(len(rows_all)); res["eval_ahead_frames"] = a.eval_ahead
    res["ckpt"] = a.eval_only or f"{a.out}/hybrid_last.pt"
    json.dump(res, open(f"{a.out}/goal_eval.json", "w"), indent=2)
    log({"event": "goal_eval", **res})


if __name__ == "__main__":
    main()
