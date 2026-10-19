"""LaWAM world-model conditioning for a frozen G0.5 flow-matching policy.

Protocol v2 (16 Oct 2026), first tier:
  u_t   = DINOv3 features of the current main-camera frame (LaWM's frozen encoder, layer -2)
  z*    = Q(u_t, u_t+tau)            frozen LAM inverse-dynamics encoder, offline target only
  z_hat = P(h_t)                     small latent prior on G0.5's causal VLM prefix states   (F00)
  u_hat = D(u_t, z_hat)              frozen LaWM decoder (lawam_pretrain)
  A(u_t, u_fut) -> residual cross-attention into the frozen G0.5 FM action expert
      F01: u_fut = u_t     (current-only control)
      F02: u_fut = u_hat   (predicted future)
      F03: evaluation-time corruption of the future branch (current / shuffled / suppressed)

The base policy stays frozen. A's output projections are zero-initialised, so with an untrained adapter
the policy is exactly the released checkpoint. The same hooks run in training (FMHelper.train_step) and
serving (FMHelper.infer), and u_t is always computed from the processed pixel tensor G0.5 itself sees,
so there is no train/eval mismatch in the conditioning path.
"""
from __future__ import annotations

import collections
import os
import random
import sys
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

G05_ROOT = os.environ.get("G05_ROOT", "/export/home/kaziz/g05")
LAWAM_ROOT = os.environ.get("LAWAM_ROOT", f"{G05_ROOT}/LaWAM")
LAM_DIR = f"{LAWAM_ROOT}/latent_action_model/logs/dino_large_vae/lam_release"
DINO_DIR = f"{LAWAM_ROOT}/weights/dinov3-vitb16-pretrain-lvd1689m"
LAWM_STATE = f"{G05_ROOT}/results/lawm_cache/lawm_pretrain_lam_state.pt"  # written by 06_cache_zstar.py
IMNET_MEAN = (0.485, 0.456, 0.406)
IMNET_STD = (0.229, 0.224, 0.225)
QWEN_MEAN, QWEN_STD = 0.5, 0.5  # checkpoints/qwen3_5_2b_base_processor/preprocessor_config.json


# --------------------------------------------------------------------------------------------------------
# Frozen LaWM (DINOv3 + LAM encoder Q + decoder D)
# --------------------------------------------------------------------------------------------------------
def load_lawm(state_path: Optional[str] = LAWM_STATE, device="cuda"):
    """LAM release architecture with lawam_pretrain weights (only the decoder differs from lawam_lam)."""
    if LAWAM_ROOT not in sys.path:
        sys.path.insert(0, LAWAM_ROOT)
    from latent_action_model.core.lam_model import load_latent_action_model

    cfg = yaml.safe_load(open(f"{LAM_DIR}/dino_large_vae.yaml"))
    cfg["model"]["vision_model_id"] = DINO_DIR
    os.makedirs(f"{G05_ROOT}/results/lawm_cache", exist_ok=True)
    local_yaml = f"{G05_ROOT}/results/lawm_cache/lam_local.yaml"
    with open(local_yaml, "w") as f:
        yaml.safe_dump(cfg, f)
    lam = load_latent_action_model(f"{LAM_DIR}/checkpoints/pytorch_model.pt", local_yaml)
    if state_path is not None:
        sd = torch.load(state_path, map_location="cpu", weights_only=False)
        missing, unexpected = lam.load_state_dict(sd, strict=False)
        assert not missing and not unexpected, f"LaWM state mismatch: missing={missing[:5]} unexpected={unexpected[:5]}"
    lam.eval().requires_grad_(False).to(device)
    return lam


def imagenet_norm(x01: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(IMNET_MEAN, device=x01.device, dtype=x01.dtype).view(1, 3, 1, 1)
    std = torch.tensor(IMNET_STD, device=x01.device, dtype=x01.dtype).view(1, 3, 1, 1)
    return (x01 - mean) / std


@torch.no_grad()
def dino_from_uint8(lam, frames_uint8: torch.Tensor) -> torch.Tensor:
    """[B,3,H,W] uint8 raw frames -> [B,256,768] (LaWM convention: 256x256, ImageNet norm, layer -2)."""
    x = frames_uint8.float().div(255.0)
    if x.shape[-1] != 256 or x.shape[-2] != 256:
        x = F.interpolate(x, size=(256, 256), mode="bilinear", align_corners=False)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        f = lam.extract_vision_features(imagenet_norm(x)[:, None])
    return f[:, 0].float()


@torch.no_grad()
def dino_from_qwen_pixels(lam, pv: torch.Tensor) -> torch.Tensor:
    """G0.5 processed pixels [B,3,224,224] (Qwen norm mean=std=0.5) -> [B,256,768]."""
    x = (pv.float() * QWEN_STD + QWEN_MEAN).clamp(0, 1)
    x = F.interpolate(x, size=(256, 256), mode="bilinear", align_corners=False)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        f = lam.extract_vision_features(imagenet_norm(x)[:, None])
    return f[:, 0].float()


@torch.no_grad()
def lawm_decode(lam, u_t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """u_t [B,256,768], z [B,1,32] -> predicted future features [B,256,768]."""
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = lam.decoder(features=u_t[:, None], actions=z)
    out = out[0] if isinstance(out, tuple) else out
    return out.reshape(u_t.shape[0], -1, u_t.shape[-1]).float()


def lawm_decode_grad(lam, u_t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """Like lawm_decode but keeps the autograd graph (joint training: grads to z and/or the decoder)."""
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = lam.decoder(features=u_t[:, None], actions=z)
    out = out[0] if isinstance(out, tuple) else out
    return out.reshape(u_t.shape[0], -1, u_t.shape[-1]).float()


# --------------------------------------------------------------------------------------------------------
# Trainable parts
# --------------------------------------------------------------------------------------------------------
class Attn(nn.Module):
    def __init__(self, q_dim, kv_dim, dim, heads, out_dim, zero_out=False):
        super().__init__()
        self.h, self.dim = heads, dim
        self.q = nn.Linear(q_dim, dim)
        self.k = nn.Linear(kv_dim, dim)
        self.v = nn.Linear(kv_dim, dim)
        self.o = nn.Linear(dim, out_dim)
        if zero_out:
            nn.init.zeros_(self.o.weight)
            nn.init.zeros_(self.o.bias)

    def forward(self, x, ctx, key_mask: Optional[torch.Tensor] = None):
        b, lq, _ = x.shape
        lk = ctx.shape[1]
        d = self.dim // self.h
        q = self.q(x).view(b, lq, self.h, d).transpose(1, 2)
        k = self.k(ctx).view(b, lk, self.h, d).transpose(1, 2)
        v = self.v(ctx).view(b, lk, self.h, d).transpose(1, 2)
        mask = None if key_mask is None else key_mask[:, None, None, :]
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return self.o(out.transpose(1, 2).reshape(b, lq, self.dim))


def mlp(width, mult=4):
    return nn.Sequential(nn.Linear(width, width * mult), nn.GELU(), nn.Linear(width * mult, width))


class LatentPrior(nn.Module):
    """P: G0.5 causal VLM prefix states -> latent action z_hat [B,1,code_dim] (LayerNorm'd like the LAM VAE latent)."""

    def __init__(self, vlm_dim=2048, width=512, n_query=8, depth=2, heads=8, code_dim=32):
        super().__init__()
        self.in_norm = nn.LayerNorm(vlm_dim)
        self.in_proj = nn.Linear(vlm_dim, width)
        self.queries = nn.Parameter(torch.randn(n_query, width) * 0.02)
        self.blocks = nn.ModuleList()
        for _ in range(depth):
            self.blocks.append(nn.ModuleDict(dict(
                ln_x=nn.LayerNorm(width), x=Attn(width, width, width, heads, width),
                ln_s=nn.LayerNorm(width), s=Attn(width, width, width, heads, width),
                ln_m=nn.LayerNorm(width), m=mlp(width))))
        self.pool_q = nn.Parameter(torch.randn(1, width) * 0.02)
        self.pool = Attn(width, width, width, heads, width)
        self.out = nn.Linear(width, code_dim)
        self.out_norm = nn.LayerNorm(code_dim)

    def forward(self, hidden, mask):
        kv = self.in_proj(self.in_norm(hidden.float()))
        q = self.queries.unsqueeze(0).expand(hidden.shape[0], -1, -1)
        for blk in self.blocks:
            q = q + blk["x"](blk["ln_x"](q), kv, mask)
            qs = blk["ln_s"](q)
            q = q + blk["s"](qs, qs)
            q = q + blk["m"](blk["ln_m"](q))
        z = self.pool(self.pool_q.unsqueeze(0).expand(q.shape[0], -1, -1), q)
        return self.out_norm(self.out(z))  # [B,1,32]


def prior_loss(z_hat, z_star):
    """LaWAM-style latent regression: (1 - cos) + MSE (MSE also pins the scale the frozen decoder expects)."""
    zh, zs = z_hat.float().reshape(-1, z_hat.shape[-1]), z_star.float().reshape(-1, z_star.shape[-1])
    cos = F.cosine_similarity(zh, zs, dim=-1)
    return (1 - cos).mean() + F.mse_loss(zh, zs), cos.mean()


class Resampler(nn.Module):
    """Perceiver resampler over 16x16 DINO patch tokens with learned 2D positions (keeps spatial info)."""

    def __init__(self, in_dim=768, n_in=256, width=256, n_query=16, heads=4):
        super().__init__()
        self.in_norm = nn.LayerNorm(in_dim)
        self.in_proj = nn.Linear(in_dim, width)
        self.pos = nn.Parameter(torch.randn(n_in, width) * 0.02)
        self.queries = nn.Parameter(torch.randn(n_query, width) * 0.02)
        self.ln_q = nn.LayerNorm(width)
        self.attn = Attn(width, width, width, heads, width)
        self.ln_m = nn.LayerNorm(width)
        self.m = mlp(width)

    def forward(self, feats):
        kv = self.in_proj(self.in_norm(feats.float())) + self.pos
        q = self.queries.unsqueeze(0).expand(feats.shape[0], -1, -1)
        q = q + self.attn(self.ln_q(q), kv)
        return q + self.m(self.ln_m(q))


class FutureAdapter(nn.Module):
    """A: two visual branches (current, future) -> residual cross-attention into selected expert layers."""

    def __init__(self, expert_dim=1024, n_layers=24, inject_every=3, width=256, heads=4, n_query=16):
        super().__init__()
        self.res_cur = Resampler(width=width, n_query=n_query, heads=heads)
        self.res_fut = Resampler(width=width, n_query=n_query, heads=heads)
        self.branch_emb = nn.Parameter(torch.zeros(2, width))
        self.ctx_norm = nn.LayerNorm(width)
        self.n_query = n_query
        self.inject_layers = list(range(inject_every - 1, n_layers, inject_every))
        self.xattn = nn.ModuleDict({
            str(i): nn.ModuleDict(dict(ln=nn.LayerNorm(expert_dim),
                                       attn=Attn(expert_dim, width, width, heads, expert_dim, zero_out=True)))
            for i in self.inject_layers})
        # episodic memory branch (used only in mode="memory"): n_mem past frames x 64 pooled DINO tokens
        self.n_mem = 4
        self.res_mem = Resampler(n_in=64 * self.n_mem, width=width, n_query=n_query, heads=heads)
        self.mem_slot = nn.Parameter(torch.zeros(self.n_mem, 1, 768))

    def build_context(self, u_cur, u_fut, mem=None):
        t_c = self.res_cur(u_cur) + self.branch_emb[0]
        t_f = self.res_fut(u_fut) + self.branch_emb[1]
        toks = [t_c, t_f]
        if mem is not None:  # mem [B, n_mem, 64, 768]
            m = (mem.float() + self.mem_slot).reshape(mem.shape[0], -1, mem.shape[-1])
            toks.append(self.res_mem(m))
        return self.ctx_norm(torch.cat(toks, dim=1))


# --------------------------------------------------------------------------------------------------------
# Hybrid container + hooks
# --------------------------------------------------------------------------------------------------------
class LaWAMHybrid(nn.Module):
    """mode: 'prior' (train P only; no adapter) | 'adapter' (A active).
    branch: 'current' (F01) | 'future' (F02).
    future_mode (eval only, F03): 'normal' | 'current' | 'shuffled' | 'suppressed'."""

    def __init__(self, lam, mode="adapter", branch="future"):
        super().__init__()
        self.lam = [lam]  # list => not registered as a submodule (frozen, saved separately)
        self.mode, self.branch = mode, branch
        self.future_mode = "normal"
        self.prior = LatentPrior()
        self.adapter = FutureAdapter()
        self.ctx: Optional[torch.Tensor] = None
        self.ctx_keymask: Optional[torch.Tensor] = None
        self.cache: dict = {}
        self._ring = collections.deque(maxlen=512)  # other-episode predictions for the 'shuffled' control
        self._rng = random.Random(0)
        self.joint = False            # J1/J2: P (+decoder) train together with the adapter
        self.joint_detach = True      # J1: future detached before the adapter; J2: action loss flows into P/decoder
        self.save_decoder = False
        self.future_dropout = 0.0     # J3: during training replace u_hat by u_t with this probability
        self.mem_feats: Optional[torch.Tensor] = None
        self.mem_off: Optional[torch.Tensor] = None  # bool [B]: True = memory masked

    @property
    def lawm(self):
        return self.lam[0]

    def trainable_state(self):
        keep = ("prior.", "adapter.")
        st = {k: v for k, v in self.state_dict().items() if k.startswith(keep)}
        if self.save_decoder:
            st.update({"lam_decoder." + k: v for k, v in self.lawm.decoder.state_dict().items()})
        return st

    def n_params(self, part):
        return sum(p.numel() for p in getattr(self, part).parameters())

    @staticmethod
    def _main_camera(pixel_values):
        if isinstance(pixel_values, dict):
            key = "exterior" if "exterior" in pixel_values else next(iter(pixel_values))
            pv = pixel_values[key]
        else:
            pv = pixel_values
        return pv[:, -1] if pv.ndim == 5 else pv  # [B,3,224,224]

    def on_prefill(self, vlm_hidden, attention_mask, split_index, pixel_values):
        s = vlm_hidden.shape[1] if split_index is None else int(split_index)
        mask = attention_mask[:, :s] != 0
        hidden = vlm_hidden[:, :s].detach()
        u_t = dino_from_qwen_pixels(self.lawm, self._main_camera(pixel_values))
        self.cache = {"hidden": hidden, "mask": mask, "u_t": u_t}
        if self.mode not in ("adapter", "memory"):
            self.ctx = None
            return
        keymask = None
        if self.joint and self.training and self.mode == "adapter" and self.branch == "future":
            with torch.autocast("cuda", dtype=torch.bfloat16):
                z_hat = self.prior(hidden, mask)
            u_hat = lawm_decode_grad(self.lawm, u_t, z_hat.float())
            self.cache["z_hat"], self.cache["u_hat"] = z_hat.float(), u_hat
            u_f = u_hat.detach() if self.joint_detach else u_hat
            self.ctx = self.adapter.build_context(u_t, u_f)
            self.ctx_keymask = None
            return
        with torch.no_grad():
            if self.branch == "current" or self.mode == "memory":
                u_f = u_t
            else:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    z_hat = self.prior(hidden, mask)
                u_f = lawm_decode(self.lawm, u_t, z_hat.float())
                self.cache["z_hat"] = z_hat.float()
                self.cache["u_hat"] = u_f
            if not self.training and self.future_mode != "normal":
                if self.future_mode == "current":
                    u_f = u_t
                elif self.future_mode == "shuffled":
                    picked = []
                    for i in range(u_f.shape[0]):
                        if len(self._ring) > 64:  # a prediction made >= 64 inference calls ago (other episode/time)
                            picked.append(self._ring[self._rng.randrange(0, len(self._ring) - 64)].to(u_f.device))
                        else:
                            picked.append(u_f[(i + 1) % u_f.shape[0]])
                    for i in range(u_f.shape[0]):
                        self._ring.append(u_f[i].detach().to("cpu", torch.bfloat16))
                    u_f = torch.stack(picked).float()
                elif self.future_mode == "suppressed":
                    n = self.adapter.n_query
                    keymask = torch.ones(u_f.shape[0], 2 * n, dtype=torch.bool, device=u_f.device)
                    keymask[:, n:] = False
                else:
                    raise ValueError(self.future_mode)
        if self.training and self.future_dropout > 0 and self.mode == "adapter" and self.branch == "future":
            drop = (torch.rand(u_f.shape[0], device=u_f.device) < self.future_dropout)[:, None, None]
            u_f = torch.where(drop, u_t, u_f)
        mem = None
        if self.mode == "memory":
            mem = self.mem_feats  # set by the caller per batch: [B, n_mem, 64, 768]
            n = self.adapter.n_query
            if keymask is None:
                keymask = torch.ones(u_f.shape[0], 3 * n, dtype=torch.bool, device=u_f.device)
            else:
                keymask = torch.cat([keymask, torch.ones(u_f.shape[0], n, dtype=torch.bool, device=u_f.device)], 1)
            if self.mem_off is not None:  # masked empty memory (control / unavailable)
                keymask[:, 2 * n:] = ~self.mem_off.to(u_f.device)[:, None]
        self.ctx = self.adapter.build_context(u_t, u_f, mem)
        self.ctx_keymask = keymask

    def layer_hook(self, layer_idx):
        blk = self.adapter.xattn[str(layer_idx)]

        def hook(_mod, _inp, out):
            if self.ctx is None:
                return out
            h = out[0] if isinstance(out, tuple) else out
            ctx, km = self.ctx, self.ctx_keymask
            rep = h.shape[0] // ctx.shape[0]
            if rep > 1:  # FMHelper.train_step stacks num_flow_samples copies of the batch
                ctx = ctx.repeat(rep, 1, 1)
                km = km.repeat(rep, 1) if km is not None else None
            delta = blk["attn"](blk["ln"](h.float()), ctx.float(), km)
            h = h + delta.to(h.dtype)
            return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h

        return hook


def attach(policy, hybrid: LaWAMHybrid):
    """Wrap vlm_prefill (captures h_t and pixels in both training and inference) and hook expert layers."""
    model = policy.model
    orig_prefill = model.vlm_prefill

    def vlm_prefill(input_ids, attention_mask, pixel_values, dtype=torch.float32, split_index=None, proprio=None):
        out = orig_prefill(input_ids, attention_mask, pixel_values, dtype=dtype, split_index=split_index,
                           proprio=proprio)
        hybrid.on_prefill(out[0], attention_mask, split_index, pixel_values)
        return out

    model.vlm_prefill = vlm_prefill
    handles = []
    if hybrid.mode in ("adapter", "memory"):
        for i in hybrid.adapter.inject_layers:
            handles.append(model.action_expert.layers[i].register_forward_hook(hybrid.layer_hook(i)))
    policy.lawam_hybrid_handles = handles
    return policy


def build_hybrid(mode, branch, ckpt=None, prior_ckpt=None, device="cuda", seed=0):
    lam = load_lawm(device=device)
    torch.manual_seed(seed)  # identical adapter init across matched conditions
    hybrid = LaWAMHybrid(lam, mode=mode, branch=branch)
    if prior_ckpt:
        sd = torch.load(prior_ckpt, map_location="cpu", weights_only=False)["state"]
        hybrid.load_state_dict({k: v for k, v in sd.items() if k.startswith("prior.")}, strict=False)
    if ckpt:
        sd = torch.load(ckpt, map_location="cpu", weights_only=False)["state"]
        dec = {k[len("lam_decoder."):]: v for k, v in sd.items() if k.startswith("lam_decoder.")}
        sd = {k: v for k, v in sd.items() if not k.startswith("lam_decoder.")}
        missing, unexpected = hybrid.load_state_dict(sd, strict=False)
        assert not unexpected, unexpected[:5]
        if dec:
            m2, u2 = lam.decoder.load_state_dict(dec, strict=True), None
            print(f"[hybrid] loaded jointly-trained LaWM decoder ({len(dec)} tensors)", flush=True)
    return hybrid.to(device)
