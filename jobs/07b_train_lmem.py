"""Train the LaWAM-conditioning parts on top of the frozen released G0.5 FM checkpoint (g05-libero).

  --mode prior                      F00: train latent prior P (z* regression); then evaluate on held-out episodes
  --mode adapter --branch current   F01: adapter A with (u_t, u_t)
  --mode adapter --branch future    F02: adapter A with (u_t, u_hat), u_hat = D(u_t, P(h_t)), P frozen

Base policy, DINOv3, LAM encoder and LaWM decoder are frozen throughout. Data: LaWAM's merged LIBERO LeRobot
set through G0.5's own dataset/processor, normalised with the *released checkpoint's* stats.
"""
import argparse
import glob
import json
import math
import os
import random
import re
import sys
import time
from functools import partial

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

JOBS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, JOBS)
import t4_fp16  # noqa: E402,F401  (T4/V100: bf16 autocast -> fp16; no-op on Ampere+)
import lawam_hybrid as H  # noqa: E402

G = H.G05_ROOT
PROJECT = f"{G}/GalaxeaVLA"
os.chdir(PROJECT)
sys.path.insert(0, PROJECT)

from omegaconf import OmegaConf  # noqa: E402
from torch.utils.data import DataLoader, RandomSampler  # noqa: E402
from torchcodec.decoders import VideoDecoder  # noqa: E402

from g05.utils.config.config_resolvers import register_default_resolvers  # noqa: E402

register_default_resolvers()
from g05.models.g05 import g05_policy as GP  # noqa: E402
from g05.utils.checkpoint.ckpt_utils import find_run_dir, load_config_from_run_dir  # noqa: E402
from g05.utils.data.data_utils import collate_fn_pad_sequences  # noqa: E402
from g05.utils.data.normalizer import load_dataset_stats_from_json  # noqa: E402
from g05.utils.data.processor_utils import build_processors, instantiate_dataset  # noqa: E402
from g05.utils.eval.eval_utils import filter_embodiment  # noqa: E402
from scripts.serve_policy import setup as g05_setup  # noqa: E402

CKPT = f"{PROJECT}/checkpoints/g05-libero/model.pt"
STATS = f"{PROJECT}/checkpoints/g05-libero/dataset_stats.json"
DATA_CFG = f"{PROJECT}/configs/data/libero_merged.yaml"
DS = f"{G}/data/libero_merged_no_noops_20hz"
CAM = "observation.images.image"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["prior", "adapter", "memory"], required=True)
    p.add_argument("--mem_mask_all", action="store_true", help="memory control: same adapter, memory tokens masked")
    p.add_argument("--instr_dropout", type=float, default=0.5, help="memory mode: P(blank instruction) after --instr_after")
    p.add_argument("--instr_after", type=int, default=40, help="frame index after which the instruction may be blanked")
    p.add_argument("--mem_eval_n", type=int, default=768)
    p.add_argument("--mem_select", choices=["offsets", "snapshot"], default="offsets",
                   help="memory frames: fixed offsets 1/2/4/8 s back, or SM1 event-based snapshots (27_snapshot_memory.py)")
    p.add_argument("--init_ckpt", default=None, help="start from these hybrid weights (adapter/prior)")
    p.add_argument("--recovery_dir", default=None, help="mix in recorded recovery episodes (25_collect_recovery.sh)")
    p.add_argument("--recovery_frac", type=float, default=0.5)
    p.add_argument("--unfreeze_expert", action="store_true", help="LIBERO-Mem: also fine-tune G0.5's action expert")
    p.add_argument("--expert_lr_scale", type=float, default=0.2, help="expert LR = lr x this")
    p.add_argument("--flow_samples", type=int, default=None, help="override FMHelper.num_flow_samples (memory/speed)")
    p.add_argument("--joint", action="store_true", help="J1/J2: train P (+decoder) together with the adapter")
    p.add_argument("--joint_nodetach", action="store_true", help="J2: action loss also flows into P / decoder")
    p.add_argument("--unfreeze_decoder", action="store_true")
    p.add_argument("--lambda_distill", type=float, default=0.1)
    p.add_argument("--lambda_perc", type=float, default=0.1)
    p.add_argument("--future_dropout", type=float, default=0.0, help="J3: P(replace u_hat by u_t) during training")
    p.add_argument("--branch", choices=["current", "future"], default="future")
    p.add_argument("--prior_ckpt", default=None)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--warmup_frac", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--heldout_mod", type=int, default=20, help="episode_index %% mod == 0 is held out (prior)")
    p.add_argument("--eval_n", type=int, default=2048)
    p.add_argument("--save_every", type=int, default=1000)
    p.add_argument("--out", required=True)
    p.add_argument("--zstar", default=f"{G}/results/lawm_cache/zstar_tau0.4.pt")
    return p.parse_args()


def to_dev(x, dev):
    if isinstance(x, torch.Tensor):
        return x.to(dev, non_blocking=True)
    if isinstance(x, dict):
        return {k: to_dev(v, dev) for k, v in x.items()}
    if isinstance(x, list):
        return [to_dev(v, dev) for v in x]
    return x


def batch_indices(batch):
    locs = batch.get("dataset_locator")
    if locs is not None:
        out = []
        for s in locs:
            m = re.search(r"local_idx=(\d+)", str(s))
            out.append(int(m.group(1)) if m else -1)
        if all(i >= 0 for i in out):
            return torch.tensor(out, dtype=torch.long)
    idx = batch["idx"]
    return idx.long().cpu() if isinstance(idx, torch.Tensor) else torch.tensor([int(i) for i in idx])


def load_rowmap():
    """Dataset row index -> (episode_index, frame_index), straight from the LeRobot parquet."""
    data = pd.concat([pd.read_parquet(f, columns=["index", "episode_index", "frame_index"])
                      for f in sorted(glob.glob(f"{DS}/data/*/*.parquet"))], ignore_index=True)
    data = data.sort_values("index").reset_index(drop=True)
    assert (data["index"].values == np.arange(len(data))).all()
    return {"episode_index": torch.from_numpy(data["episode_index"].values.astype(np.int64)),
            "frame_index": torch.from_numpy(data["frame_index"].values.astype(np.int64))}


class FrameReader:
    """Raw main-camera frames by dataset row index (for the idx-alignment check and future-frame targets)."""

    def __init__(self, zcache):
        self.info = json.load(open(f"{DS}/meta/info.json"))
        self.fps = self.info["fps"]
        eps = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"{DS}/meta/episodes/*/*.parquet"))])
        self.eps = eps.set_index("episode_index")
        self.ep = zcache["episode_index"]
        self.fr = zcache["frame_index"]
        self.dec = {}

    def frames(self, rows):
        out = []
        for r in rows.tolist():
            e = self.eps.loc[int(self.ep[r])]
            path = f"{DS}/" + self.info["video_path"].format(video_key=CAM, chunk_index=int(e[f"videos/{CAM}/chunk_index"]),
                                                            file_index=int(e[f"videos/{CAM}/file_index"]))
            if path not in self.dec:
                self.dec[path] = VideoDecoder(path)
            ts = float(e[f"videos/{CAM}/from_timestamp"]) + int(self.fr[r]) / self.fps
            out.append(self.dec[path].get_frames_played_at([ts]).data[0])
        return torch.stack(out)  # [B,3,H,W] uint8

    def future_rows(self, rows, k):
        out = []
        for r in rows.tolist():
            L = int(self.eps.loc[int(self.ep[r])]["length"])
            out.append(r + min(k, L - 1 - int(self.fr[r])))
        return torch.tensor(out)


def prefill_only(policy, batch, dev):
    samples, pv = batch["samples"], batch["pixel_values"]
    input_ids, _labels, attention_mask, split_index = policy.processor.encode_train(
        samples, device=dev, training=False, max_chunk_token_length=policy.max_chunk_token_length,
        max_pad_token_length=policy.max_pad_token_length)
    pvp = policy.process_pixel_values(pv)
    proprio = (GP.build_proprio_batch(samples, device=dev, dtype=torch.float32,
                                      zero_values=policy.model.proprio_encoder == "zeros")
               if policy.model.proprio_embedder is not None else None)
    dtype = next(iter(pvp.values())).dtype
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        policy.model.vlm_prefill(input_ids, attention_mask, pvp, dtype=dtype, split_index=split_index, proprio=proprio)


def lr_at(step, total, base, warm_frac):
    warm = max(1, int(total * warm_frac))
    if step < warm:
        return base * (step + 1) / warm
    return base * 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, total - warm)))


def main():
    a = parse_args()
    os.makedirs(a.out, exist_ok=True)
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    dev = torch.device("cuda")
    log_f = open(f"{a.out}/train_log.jsonl", "a")

    def log(d):
        d["t"] = round(time.time() - t0, 1)
        print(json.dumps(d), flush=True)
        log_f.write(json.dumps(d) + "\n"); log_f.flush()

    t0 = time.time()
    # ---- frozen released policy, loaded exactly as the eval server loads it
    cfg = load_config_from_run_dir(find_run_dir(CKPT), CKPT, [])
    if cfg.get("eval_embodiment") and "embodiment_datasets" in cfg.data:
        filter_embodiment(cfg, cfg.eval_embodiment)
    policy, _ = g05_setup(cfg, device="cuda")
    policy.eval().requires_grad_(False)
    expert = next(m for n, m in policy.named_modules() if n.endswith("action_expert"))
    if t4_fp16.T4_MODE:  # no bf16 / no FlashAttention on sm<80: fp16 weights, SDPA paths, inputs cast to weight dtype
        for _name, _mod in list(sys.modules.items()):
            if "g05" in _name:
                for _attr in ("_flash_attn_varlen", "flash_attn_func"):
                    if getattr(_mod, _attr, None) is not None:
                        setattr(_mod, _attr, None)
        policy.half()
        if a.unfreeze_expert:
            expert.float()  # fp32 master weights for the trained expert
        torch.cuda.empty_cache()

        def _cast_in(mod, args):
            if args and torch.is_tensor(args[0]) and args[0].is_floating_point() and args[0].dtype != mod.weight.dtype:
                return (args[0].to(mod.weight.dtype),) + tuple(args[1:])
        for _m in policy.modules():
            if isinstance(_m, (torch.nn.Linear, torch.nn.Conv2d, torch.nn.Conv3d, torch.nn.LayerNorm)) and getattr(_m, "weight", None) is not None:
                _m.register_forward_pre_hook(_cast_in)
    if a.flow_samples:
        n_set = 0
        for m in [policy] + list(policy.modules()):
            for v in list(vars(m).values()):
                if hasattr(v, "num_flow_samples") and not isinstance(v, torch.nn.Module):
                    v.num_flow_samples = a.flow_samples; n_set += 1
            if hasattr(m, "num_flow_samples") and isinstance(getattr(m, "num_flow_samples"), int):
                m.num_flow_samples = a.flow_samples; n_set += 1
        print(f"[lmem] num_flow_samples -> {a.flow_samples} on {n_set} objects", flush=True)

    hybrid = H.build_hybrid(a.mode, a.branch, ckpt=a.init_ckpt, prior_ckpt=a.prior_ckpt, seed=a.seed)
    H.attach(policy, hybrid)
    train_part = "prior" if a.mode == "prior" else "adapter"  # memory mode trains adapter.* (incl. res_mem)
    parts = ("adapter.", "prior.") if a.joint else (train_part + ".",)
    for n, p in hybrid.named_parameters():
        p.requires_grad_(n.startswith(parts))
    hybrid.joint, hybrid.joint_detach = a.joint, not a.joint_nodetach
    hybrid.future_dropout = a.future_dropout
    if a.unfreeze_decoder:
        hybrid.lawm.decoder.requires_grad_(True)
        hybrid.lawm.decoder.train()
        hybrid.save_decoder = True
    hybrid.prior.train(a.mode == "prior" or a.joint)
    hybrid.adapter.train(a.mode in ("adapter", "memory"))
    params = [p for p in hybrid.parameters() if p.requires_grad]
    expert_params = []
    if a.unfreeze_expert:
        expert.requires_grad_(True); expert.train()
        expert_params = [p for p in expert.parameters()]
    if a.unfreeze_decoder:
        params += [p for p in hybrid.lawm.decoder.parameters() if p.requires_grad]
    log({"event": "setup", "mode": a.mode, "branch": a.branch, "trainable": train_part,
         "n_trainable": sum(p.numel() for p in params) + sum(p.numel() for p in expert_params), "n_expert": sum(p.numel() for p in expert_params),
         "n_prior": hybrid.n_params("prior"),
         "n_adapter": hybrid.n_params("adapter"), "inject_layers": hybrid.adapter.inject_layers,
         "args": vars(a)})

    # ---- data (released stats; LaWAM's merged LIBERO set)
    OmegaConf.set_struct(cfg, False)
    cfg.data = OmegaConf.load(DATA_CFG)
    dataset = instantiate_dataset(cfg, is_training_set=True)
    proc = build_processors(cfg)
    proc.set_normalizer_from_stats(load_dataset_stats_from_json(STATS))
    dataset.set_processor(proc)
    _collate = partial(collate_fn_pad_sequences, padding_input_id=proc.pad_token_id)

    def _fill(items):  # recorded episodes lack some metadata keys of the base set (e.g. "embodiment"): fill from a donor
        keys = set().union(*[it.keys() for it in items])
        for k in keys:
            donor = next((it[k] for it in items if k in it), None)
            for it in items:
                if k not in it:
                    it[k] = donor
            sub = [it[k] for it in items]
            if sub and all(isinstance(x, dict) for x in sub):
                _fill(sub)  # nested dicts, e.g. the per-sample "samples" dicts that the policy collates again

    def collate(batch):
        _fill(batch)
        return _collate(batch)
    full_dataset = dataset
    mem = None
    if a.mode == "memory":
        mem = MemoryCache(a.mem_select)
        train_idx = np.nonzero(mem.ep % a.heldout_mod != 0)[0]
        train_idx = train_idx[train_idx < len(dataset)]
        dataset = torch.utils.data.Subset(full_dataset, train_idx.tolist())
    gen = torch.Generator().manual_seed(a.seed)
    sampler = RandomSampler(dataset, generator=gen)
    if a.recovery_dir:
        rec_ds = make_recovery_dataset(full_dataset, proc, a.recovery_dir, f"{a.out}/recovery_cache")
        n_full, n_rec = len(dataset), len(rec_ds)
        dataset = torch.utils.data.ConcatDataset([dataset, rec_ds])
        w = torch.cat([torch.full((n_full,), (1 - a.recovery_frac) / n_full), torch.full((n_rec,), a.recovery_frac / n_rec)])
        sampler = torch.utils.data.WeightedRandomSampler(w, num_samples=10_000_000, replacement=True, generator=gen)
        log({"event": "recovery_data", "frames": n_rec, "episodes": rec_ds.n_episodes, "frac": a.recovery_frac})
    loader = DataLoader(dataset, batch_size=a.batch, sampler=sampler,
                        num_workers=a.workers, collate_fn=collate, drop_last=True, persistent_workers=True,
                        prefetch_factor=2, pin_memory=True)
    rowmap = load_rowmap()
    joint_feats = None
    if a.joint:
        joint_feats = MemoryCache()  # pooled future-frame features for the perceptual loss
    if a.mode == "prior" or a.joint:
        zc = torch.load(a.zstar, weights_only=False)
        assert torch.equal(zc["episode_index"], rowmap["episode_index"]), "z* cache rows != dataset rows"
        zstar = zc["zstar"].to(dev)
        rowmap["clamped"] = zc["clamped"]
        k_fut = int(round(zc["tau_s"] * zc["fps"]))
    else:
        zstar, k_fut = torch.zeros(len(rowmap["episode_index"]), 32, device=dev), 8
    heldout_ep = (rowmap["episode_index"] % a.heldout_mod == 0)
    reader = FrameReader(rowmap)
    log({"event": "data", "n_samples": len(dataset), "n_zstar": int(zstar.shape[0]),
         "heldout_frames": int(heldout_ep.sum()), "k_future_frames": k_fut})

    groups = [{"params": params, "lr_scale": 1.0}]
    if expert_params:
        groups.append({"params": expert_params, "lr_scale": a.expert_lr_scale})
    opt = torch.optim.AdamW(groups, lr=a.lr, weight_decay=0.01, betas=(0.9, 0.95))
    scaler = torch.cuda.amp.GradScaler(enabled=t4_fp16.T4_MODE)
    all_params = params + expert_params
    step, it = 0, iter(loader)
    checked = False
    while step < a.steps:
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader); batch = next(it)
        rows = batch_indices(batch)
        batch = to_dev(batch, dev)
        if a.recovery_dir and not checked:
            checked = True  # mixed batches contain recorded frames without dataset rows; alignment was verified on the base data
        if not checked:  # idx -> frame alignment check (z* must belong to the frame G0.5 sees)
            seen = (hybrid._main_camera(batch["pixel_values"])[:16].float() * 0.5 + 0.5).clamp(0, 1)
            raw = reader.frames(rows[:16]).float().div(255)
            raw = F.interpolate(raw, size=tuple(seen.shape[-2:]), mode="bilinear", align_corners=False).to(dev)
            def _c(x, y):
                return torch.corrcoef(torch.stack([x.flatten(), y.flatten()]))[0, 1].item()
            n = len(raw)
            corr = [_c(raw[i], seen[i]) for i in range(n)]
            corr_shuf = [_c(raw[i], seen[(i + 1) % n]) for i in range(n)]
            pv_main = hybrid._main_camera(batch["pixel_values"]).float()
            log({"event": "idx_alignment", "pixel_corr_matched": float(np.mean(corr)), "pixel_corr_min": float(np.min(corr)),
                 "pixel_corr_shuffled": float(np.mean(corr_shuf)), "pv_min": pv_main.min().item(),
                 "pv_max": pv_main.max().item(),
                 "pv_keys": list(batch["pixel_values"].keys()) if isinstance(batch["pixel_values"], dict) else "tensor"})
            if np.mean(corr) < np.mean(corr_shuf) + 0.2:
                raise RuntimeError(f"dataset idx does not match decoded frames (matched {np.mean(corr):.3f} vs "
                                   f"shuffled {np.mean(corr_shuf):.3f}); z* would be misaligned")
            checked = True
        lr = lr_at(step, a.steps, a.lr, a.warmup_frac)
        for g in opt.param_groups:
            g["lr"] = lr * g.get("lr_scale", 1.0)

        if a.mode == "prior":
            prefill_only(policy, batch, dev)
            keep = ~heldout_ep[rows]
            if keep.sum() == 0:
                continue
            with torch.autocast("cuda", dtype=torch.bfloat16):
                z_hat = hybrid.prior(hybrid.cache["hidden"], hybrid.cache["mask"])
            loss, cos = prior_loss_rows(z_hat, zstar[rows.to(dev)], keep.to(dev))
            extra = {"cos": round(cos.item(), 4)}
        else:
            if a.mode == "memory":
                fr_b = mem.fr[rows.numpy()]
                if step == 0:
                    log({"event": "memory_batch_check", "heldout_leak": float((mem.ep[rows.numpy()] % a.heldout_mod == 0).mean()),
                         "command_example": str(batch["samples"][0].get("command"))[:80]})
                for i, smp in enumerate(batch["samples"]):
                    if fr_b[i] >= a.instr_after and random.random() < a.instr_dropout:
                        smp["command"] = ""
                hybrid.mem_feats = mem.memory(rows.numpy(), dev)
                hybrid.mem_off = torch.full((len(rows),), bool(a.mem_mask_all))
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss, ld = policy.forward_train(samples=batch["samples"], pixel_values=batch["pixel_values"],
                                                actions=batch["action"], action_pad_masks=batch["action_is_pad"],
                                                action_dim_is_pad=batch.get("action_dim_is_pad"))
            extra = {}
            if a.joint:
                c = hybrid.cache
                r_np = rows.numpy()
                l_dist, cos = H.prior_loss(c["z_hat"], zstar[rows.to(dev)][:, None])
                fut = r_np + np.minimum(k_fut, joint_feats_len(rowmap, r_np))
                tgt = torch.from_numpy(np.asarray(joint_feats.feats[np.sort(fut)])[np.argsort(np.argsort(fut))]).to(dev).float()
                u_hat = c["u_hat"].reshape(len(r_np), 16, 16, -1).permute(0, 3, 1, 2)
                u_hat = F.avg_pool2d(u_hat, 2).permute(0, 2, 3, 1).reshape(len(r_np), 64, -1)
                l_perc = F.mse_loss(u_hat, tgt)
                extra = {"fm": round(loss.item(), 5), "distill": round(l_dist.item(), 4), "cos": round(cos.item(), 4),
                         "perc": round(l_perc.item(), 4)}
                loss = loss + a.lambda_distill * l_dist + a.lambda_perc * l_perc
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        gnorm = torch.nn.utils.clip_grad_norm_(all_params, 1.0)
        scaler.step(opt); scaler.update()
        step += 1
        if step % 25 == 0 or step == 1:
            log({"event": "train", "step": step, "loss": round(loss.item(), 5), "lr": lr, "gnorm": round(gnorm.item(), 4), **extra})
        if step % a.save_every == 0 or step == a.steps:
            torch.save({"state": hybrid.trainable_state(), "step": step, "args": vars(a)}, f"{a.out}/hybrid_step{step}.pt")
            torch.save({"state": hybrid.trainable_state(), "step": step, "args": vars(a)}, f"{a.out}/hybrid_last.pt")
            if expert_params:
                torch.save({k: v.half() for k, v in expert.state_dict().items()}, f"{a.out}/expert_last.pt")

    if a.mode == "prior":
        evaluate_prior(a, policy, hybrid, dataset, proc, collate, zstar, rowmap, heldout_ep, reader, k_fut, dev, log)
    if a.mode == "memory":
        evaluate_memory(a, policy, hybrid, full_dataset, proc, collate, mem, dev, log)
    log({"event": "done", "steps": step})


class _RecoveryFrames:
    """LeRobot-format samples (same keys/dtypes as the merged LIBERO set) from recorded recovery episodes."""

    def __init__(self, src_dir, cache_dir):
        os.makedirs(cache_dir, exist_ok=True)
        self.eps, self.index, self.stride = [], [], []
        for e, f in enumerate(sorted(glob.glob(f"{src_dir}/*/rec/*.npz"))):
            base = f"{cache_dir}/ep{e:05d}"
            if not os.path.exists(base + "_wrist.npy"):
                z = np.load(f, allow_pickle=True)
                for k in ("image", "wrist_image", "state", "action"):
                    np.save(base + ("_img.npy" if k == "image" else "_wrist.npy" if k == "wrist_image" else f"_{k}.npy"), z[k])
                with open(base + "_task.txt", "w") as fh:
                    fh.write(str(z["task_description"]))
                np.save(base + "_stride.npy", np.array(int(z["img_stride"]) if "img_stride" in z.files else 1))
            n = len(np.load(base + "_action.npy", mmap_mode="r"))
            st = int(np.load(base + "_stride.npy")) if os.path.exists(base + "_stride.npy") else 1
            self.eps.append(base); self.stride.append(st)
            self.index += [(e, t) for t in range(0, n, st)]
        self._mm = {}

    def _arr(self, e, k):
        key = (e, k)
        if key not in self._mm:
            self._mm[key] = np.load(f"{self.eps[e]}_{k}.npy", mmap_mode="r")
        return self._mm[key]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        e, t = self.index[i]
        img, wr, st, act = (self._arr(e, k) for k in ("img", "wrist", "state", "action"))
        n = len(act)
        fut = np.arange(t, t + 32)
        pad = fut >= n
        chunk = act[np.minimum(fut, n - 1)]
        task = random.choice(open(f"{self.eps[e]}_task.txt").read().split("||"))
        return {
            "observation.images.image": torch.from_numpy(np.ascontiguousarray(img[t // self.stride[e]])).permute(2, 0, 1).float().div(255),
            "observation.images.wrist_image": torch.from_numpy(np.ascontiguousarray(wr[t // self.stride[e]])).permute(2, 0, 1).float().div(255),
            "observation.state": torch.from_numpy(np.asarray(st[t:t + 1], dtype=np.float32)),
            "action": torch.from_numpy(np.asarray(chunk, dtype=np.float32)),
            "timestamp": torch.tensor(t / 20.0), "frame_index": torch.tensor(t), "episode_index": torch.tensor(100000 + e),
            "index": torch.tensor(i), "task_index": torch.tensor(0),
            "observation.images.image_is_pad": torch.zeros(1, dtype=torch.bool),
            "observation.images.wrist_image_is_pad": torch.zeros(1, dtype=torch.bool),
            "observation.state_is_pad": torch.zeros(1, dtype=torch.bool),
            "action_is_pad": torch.from_numpy(pad), "chunked_task_index": None, "task": task,
            "step_is_qualified": True, "dataset_index": torch.tensor(0),
        }


def make_recovery_dataset(full_dataset, proc, src_dir, cache_dir):
    """Clone G0.5's inner LeRobot dataset object and swap its frame source, so the normal state/action/image
    extraction, normalisation and processor run unchanged on the recorded frames."""
    import copy

    inner = full_dataset.datasets[0]
    frames = _RecoveryFrames(src_dir, cache_dir)

    class RecoveryDataset(type(inner)):
        def __len__(self):
            return len(frames)

        def _locate_sample(self, idx):
            return f"recovery_frame={idx}"

        def _resample_random_idx(self):
            return random.randrange(len(frames))

    rec = copy.copy(inner)
    rec.__class__ = RecoveryDataset
    rec.multi_dataset = frames
    rec._start_idx = 0
    if hasattr(rec, "_overfit_indices"):
        del rec._overfit_indices
    # keep the concrete per-embodiment processor the clone inherited from the inner dataset (the mixture
    # processor is only a registry), in training mode like the base data
    assert getattr(rec, "processor", None) is not None, "inner dataset has no processor bound yet"
    rec.processor.train()
    rec.n_episodes = len(frames.eps)
    return rec


class MemoryCache:
    """Pooled DINOv3 features of past frames (1, 2, 4, 8 s back, clamped to episode start) from 22_cache_feats.py."""
    OFFSETS = (20, 40, 80, 160)

    def __init__(self, select="offsets"):
        c = f"{G}/results/feat_cache"
        m = np.load(f"{c}/meta.npz")
        self.ep, self.fr = m["episode_index"], m["frame_index"]
        self.feats = np.load(f"{c}/feats.npy", mmap_mode="r")
        self.select = select
        self.snap_rows = np.load(f"{c}/snapshot_mem_rows.npy") if select == "snapshot" else None

    def memory(self, rows, dev):
        if self.select == "snapshot":
            idx = self.snap_rows[rows]  # [B, 4]: first snapshot + 3 most recent events before t
        else:
            idx = np.stack([rows - np.minimum(d, self.fr[rows]) for d in self.OFFSETS], 1)  # [B, 4]
        flat = idx.reshape(-1)
        order = np.argsort(flat)
        got = np.asarray(self.feats[flat[order]])
        out = np.empty_like(got)
        out[order] = got
        return torch.from_numpy(out.reshape(len(rows), len(self.OFFSETS), 64, 768)).to(dev)


@torch.no_grad()
def evaluate_memory(a, policy, hybrid, dataset, proc, collate, mem, dev, log):
    """Offline decision test on held-out episodes (frames >= instr_after): frozen-noise FM loss under
    instruction present / blank, memory real / shuffled / masked."""
    hybrid.adapter.eval()
    if hasattr(proc, "eval"):
        proc.eval()
    rng = np.random.RandomState(7)
    cand = np.nonzero((mem.ep % a.heldout_mod == 0) & (mem.fr >= a.instr_after))[0]
    cand = cand[cand < len(dataset)]
    rows_all = rng.choice(cand, min(a.mem_eval_n, len(cand)), replace=False)
    conds = {"instr_mem": (False, "real"), "blank_mem": (True, "real"), "blank_shuffled_mem": (True, "shuffled"),
             "blank_masked_mem": (True, "masked"), "instr_masked_mem": (False, "masked")}
    totals = {k: [] for k in conds}
    for s in range(0, len(rows_all), 64):
        items = [dataset[int(r)] for r in rows_all[s:s + 64]]
        base = collate(items)
        rows = batch_indices(base).numpy()
        for name, (blank, mode) in conds.items():
            b = to_dev(collate([dataset[int(r)] for r in rows_all[s:s + 64]]), dev)
            if blank:
                for smp in b["samples"]:
                    smp["command"] = ""
            m = mem.memory(rows, dev)
            if mode == "shuffled":
                m = m[torch.roll(torch.arange(len(rows)), 1)]
            hybrid.mem_feats = m
            hybrid.mem_off = torch.full((len(rows),), mode == "masked" or bool(a.mem_mask_all))
            torch.manual_seed(10_000 + s)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss, _ = policy.forward_train(samples=b["samples"], pixel_values=b["pixel_values"], actions=b["action"],
                                               action_pad_masks=b["action_is_pad"], action_dim_is_pad=b.get("action_dim_is_pad"))
            totals[name].append(loss.item())
    res = {k: float(np.mean(v)) for k, v in totals.items()}
    res["n"] = int(len(rows_all)); res["mem_mask_all"] = bool(a.mem_mask_all); res["mem_select"] = a.mem_select
    json.dump(res, open(f"{a.out}/memory_eval.json", "w"), indent=2)
    log({"event": "memory_eval", **res})


def joint_feats_len(rowmap, rows):
    """frames remaining after each row inside its episode (for clamped future targets)."""
    if "remaining" not in rowmap:
        ep = rowmap["episode_index"].numpy()
        fr = rowmap["frame_index"].numpy()
        last = np.zeros(ep.max() + 1, dtype=np.int64)
        np.maximum.at(last, ep, fr)
        rowmap["remaining"] = last[ep] - fr
    return rowmap["remaining"][rows]


def prior_loss_rows(z_hat, z_star, keep):
    return H.prior_loss(z_hat[keep], z_star[keep][:, None])


@torch.no_grad()
def evaluate_prior(a, policy, hybrid, dataset, proc, collate, zstar, rowmap, heldout_ep, reader, k_fut, dev, log):
    """F00 on held-out episodes: latent agreement and decoded-feature error vs persistence and the W00 oracle."""
    hybrid.prior.eval()
    if hasattr(proc, "eval"):
        proc.eval()
    rows_all = torch.nonzero(heldout_ep & ~rowmap["clamped"]).flatten()
    g = torch.Generator().manual_seed(123)
    rows_all = rows_all[torch.randperm(len(rows_all), generator=g)[: a.eval_n]]
    recs = []
    for s in range(0, len(rows_all), 64):
        rows = rows_all[s:s + 64]
        items = [dataset[int(r)] for r in rows]
        batch = collate(items)
        rows = batch_indices(batch)  # trust what the dataset says it returned
        batch = to_dev(batch, dev)
        prefill_only(policy, batch, dev)
        c = hybrid.cache
        with torch.autocast("cuda", dtype=torch.bfloat16):
            z_hat = hybrid.prior(c["hidden"], c["mask"]).float()
        zs = zstar[rows.to(dev)][:, None]
        u_t = c["u_t"]
        u_pred = H.lawm_decode(hybrid.lawm, u_t, z_hat)
        u_orc = H.lawm_decode(hybrid.lawm, u_t, zs)
        u_tgt = H.dino_from_uint8(hybrid.lawm, reader.frames(reader.future_rows(rows, k_fut)).to(dev))
        perm = torch.roll(torch.arange(len(rows)), 1)
        u_wrong = H.lawm_decode(hybrid.lawm, u_t, z_hat[perm])
        mse = lambda x: ((x - u_tgt) ** 2).mean(dim=(1, 2))  # noqa: E731
        cos = F.cosine_similarity(z_hat.reshape(len(rows), -1), zs.reshape(len(rows), -1), dim=-1)
        for i in range(len(rows)):
            recs.append({"row": int(rows[i]), "lat_cos": cos[i].item(),
                         "lat_mse": F.mse_loss(z_hat[i], zs[i]).item(),
                         "mse_pred": mse(u_pred)[i].item(), "mse_oracle_W00": mse(u_orc)[i].item(),
                         "mse_persist": mse(u_t)[i].item(), "mse_wrong_latent": mse(u_wrong)[i].item()})
    df = pd.DataFrame(recs)
    df.to_csv(f"{a.out}/f00_heldout_records.csv", index=False)
    moving = df[df["mse_persist"] >= df["mse_persist"].median()]

    def summ(d):
        return {"n": len(d), "latent_cos": d["lat_cos"].mean(), "latent_mse": d["lat_mse"].mean(),
                "mse_pred": d["mse_pred"].mean(), "mse_oracle_W00": d["mse_oracle_W00"].mean(),
                "mse_persist": d["mse_persist"].mean(), "mse_wrong_latent": d["mse_wrong_latent"].mean(),
                "pred_over_persist": d["mse_pred"].mean() / d["mse_persist"].mean(),
                "oracle_over_persist": d["mse_oracle_W00"].mean() / d["mse_persist"].mean(),
                "wrong_over_pred": d["mse_wrong_latent"].mean() / d["mse_pred"].mean(),
                "frac_pred_beats_persist": float((d["mse_pred"] < d["mse_persist"]).mean())}

    res = {"all": summ(df), "moving_half": summ(moving)}
    json.dump(res, open(f"{a.out}/f00_summary.json", "w"), indent=2, default=float)
    log({"event": "F00", **{f"{k}/{kk}": (round(vv, 4) if isinstance(vv, float) else vv) for k, v in res.items() for kk, vv in v.items()}})


if __name__ == "__main__":
    main()
