"""Goal-conditioned adapter on the frozen G0.5 FM policy: pi0.7-style subgoals in DINOv3 latent space.

pi0.7 (arXiv 2604.15483) conditions its policy on subgoal images that are REAL future frames during training
(present in some examples; 25% the segment's last frame, 75% a frame 0-4 s ahead) and only uses a world model
to supply them at test time. F02 instead trained on its own 0.4 s predictions, always present, and over-trusted
them in closed loop (F03). This module is the pi0.7 recipe on our adapter:

  context = [ res_cur(u_t), res_fut(u_t), res_goal(g) ]      (16 tokens each, residual cross-attn into the expert)
  g       = pooled 8x8 DINOv3 features of a future frame of the same demo (feat_cache), or absent
  absent  = the goal tokens are masked out of the cross-attention (trained, so "no goal" is in distribution)

The first two token groups are exactly F01's context. Initialised from F01, the adapter with no goal is
therefore F01 itself, and F01cont_t4 (F01 + 2,000 steps on the same data) is a matched control.

Also holds the per-batch state the server fills in: goal features and classifier-free-guidance pairs.
"""
from __future__ import annotations

import os
import sys
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lawam_hybrid as H  # noqa: E402


class GoalAdapter(H.FutureAdapter):
    """goal_input: 'abs' = the goal's pooled features; 'delta' = [LN(goal), LN(goal - current)] per token, so the
    change the scene still has to undergo is an explicit input."""

    def __init__(self, goal_input="abs", **kw):
        super().__init__(**kw)
        width = self.res_cur.in_proj.out_features
        self.goal_input = goal_input
        if goal_input == "delta":
            self.goal_ln_abs, self.goal_ln_delta = nn.LayerNorm(768), nn.LayerNorm(768)
            self.res_goal = H.Resampler(in_dim=1536, n_in=64, width=width, n_query=self.n_query, heads=4)
        else:
            self.res_goal = H.Resampler(n_in=64, width=width, n_query=self.n_query, heads=4)
        self.goal_emb = nn.Parameter(torch.zeros(width))

    def build_goal_context(self, u_cur, goal):
        t_c = self.res_cur(u_cur) + self.branch_emb[0]
        t_f = self.res_fut(u_cur) + self.branch_emb[1]
        if self.goal_input == "delta":
            b = u_cur.shape[0]
            u_pool = F.avg_pool2d(u_cur.view(b, 16, 16, -1).permute(0, 3, 1, 2), 2).permute(0, 2, 3, 1).reshape(b, 64, -1)
            goal = torch.cat([self.goal_ln_abs(goal), self.goal_ln_delta(goal - u_pool)], dim=-1)
        t_g = self.res_goal(goal) + self.goal_emb
        return self.ctx_norm(torch.cat([t_c, t_f, t_g], dim=1))


class GoalHybrid(H.LaWAMHybrid):
    """Set goal_feats [B,64,768] and goal_on bool [B] before each forward (trainer / server)."""

    def __init__(self, lam, goal_input="abs"):
        super().__init__(lam, mode="adapter", branch="current")
        self.adapter = GoalAdapter(goal_input=goal_input)
        self.goal_feats: Optional[torch.Tensor] = None
        self.goal_on: Optional[torch.Tensor] = None

    def on_prefill(self, vlm_hidden, attention_mask, split_index, pixel_values):
        u_t = H.dino_from_qwen_pixels(self.lawm, self._main_camera(pixel_values))
        self.cache = {"u_t": u_t}
        b, n = u_t.shape[0], self.adapter.n_query
        if self.goal_feats is None:
            goal = torch.zeros(b, 64, u_t.shape[-1], device=u_t.device)
            on = torch.zeros(b, dtype=torch.bool, device=u_t.device)
        else:
            goal, on = self.goal_feats.to(u_t.device).float(), self.goal_on.to(u_t.device)
            assert goal.shape[0] == b and on.shape[0] == b, (goal.shape, on.shape, b)
        self.ctx = self.adapter.build_goal_context(u_t, goal)
        keymask = torch.ones(b, 3 * n, dtype=torch.bool, device=u_t.device)
        keymask[:, 2 * n:] = on[:, None]
        self.ctx_keymask = keymask


def build_goal_hybrid(ckpt=None, device="cuda", seed=0, goal_input=None):
    """ckpt: an F01 checkpoint (goal branch starts fresh) or a trained goal-adapter checkpoint.
    goal_input: 'abs' | 'delta'; by default taken from a goal-adapter checkpoint's training args, else 'abs'."""
    lam = H.load_lawm(device=device)
    ck = torch.load(ckpt, map_location="cpu", weights_only=False) if ckpt else None
    if goal_input is None:
        goal_input = (ck.get("args") or {}).get("goal_input", "abs") if ck else "abs"
    torch.manual_seed(seed)
    hybrid = GoalHybrid(lam, goal_input=goal_input)
    print(f"[gc] goal_input={goal_input}", flush=True)
    if ckpt:
        sd = ck["state"]
        missing, unexpected = hybrid.load_state_dict(sd, strict=False)
        assert not unexpected, unexpected[:5]
        fresh = sorted({k.split(".")[1] for k in missing if k.startswith("adapter.")})
        print(f"[gc] loaded {ckpt}; adapter parts not in the checkpoint (fresh init): {fresh}", flush=True)
    return hybrid.to(device)


class CFGState:
    """Classifier-free guidance across a batch: pairs (i, j, w) of (prompt row, negative-prompt row, scale).
    Row j starts from row i's noise; at every Euler step both get v_j + w * (v_i - v_j)."""
    pairs: list = []


def install_cfg(policy):
    fm = policy.model.fm_helper
    ae = policy.model.action_expert
    orig_noise, orig_decode = fm._sample_noise, ae.decode

    def sample_noise(actions, dtype, embodiment_types=None):
        x = orig_noise(actions, dtype, embodiment_types)
        for i, j, _w in CFGState.pairs:
            x[j] = x[i]
        return x

    def decode(hidden):
        v = orig_decode(hidden)
        if CFGState.pairs:
            v = v.clone()
            for i, j, w in CFGState.pairs:
                g = v[j] + w * (v[i] - v[j])
                v[i] = g
                v[j] = g
        return v

    fm._sample_noise = sample_noise
    ae.decode = decode
