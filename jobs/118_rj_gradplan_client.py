"""RoboJEPA's RoboCasa planning client (scripts/deployment/agents/world_model/policy_client.py) with the CEM planner
replaced by planning through the world model's gradients: the SuperMimic idea (analytic gradients through a learned,
differentiable model instead of a sampling-based optimiser) applied to RoboJEPA's latent planning.

mpc_utils.cem is swapped at runtime for a function with the same signature and return value, so the authors' client,
env server, goal encoding, execution and metrics are unchanged. PLANNER selects:
  grad     K action sequences (the first one = the plan CEM would start from, the others perturbed by the CEM initial
           std) optimised with Adam on the same objective CEM ranks candidates by (view-weighted negative L1 between the
           imagined and goal features), backpropagated through the H-step world-model rollout; the best sequence wins
  cemgrad  CEM with a third of its iterations, then the same gradient refinement from the CEM mean
Actions keep CEM's constraints: translation projected onto the maxnorm ball, rotation 0 under translation_only,
gripper clipped to maxgripper (0 unless use_gripper). The rollout runs WorldModel.step(no_grad=False), i.e. full
attention over the history (the KV-cache path is inference-only), so the launcher drops --use_cache for these runs.
Env: PLANNER (grad | cemgrad), GP_ITERS (40), GP_K (8), GP_LR (0.03). Run from the client's directory, same CLI.
"""
import os
import runpy
import sys
import time

import torch

sys.path.insert(0, os.getcwd())  # the client imports its sibling modules (distributed, ...) by name
import app.robo_jepa.world_model as wm_module  # noqa: E402
from app.robo_jepa.mpc import mpc_utils  # noqa: E402

_cem = mpc_utils.cem
_compute_new_pose = wm_module.compute_new_pose


def _pose_no_grad(poses, action, **kw):
    """The pose integration for the state tokens runs in numpy: feed it detached inputs (the gradient reaches the
    actions through the predictor's action tokens; the integrated pose of later steps is treated as a constant)."""
    return _compute_new_pose(poses.detach() if torch.is_tensor(poses) else poses, action.detach(), **kw)


wm_module.compute_new_pose = _pose_no_grad
LOG = []


def _project(u, maxnorm, maxgripper, translation_only, use_gripper):
    xyz = u[..., :3]
    n = xyz.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    xyz = torch.where(n < maxnorm, xyz, xyz / n * maxnorm)
    rot = torch.zeros_like(u[..., 3:6]) if translation_only else u[..., 3:6]
    grip = u[..., 6:].clamp(-maxgripper, maxgripper) if use_gripper else torch.zeros_like(u[..., 6:])
    return torch.cat([xyz, rot, grip], -1)


def grad_refine(world_model, wm_state, goal_features, init=None, rollout=3, maxnorm=0.05, maxgripper=0.1,
                objective=mpc_utils.l1, robot_init_std=0.05, gripper_init_std=0.1, view_weights=None, total_steps=100,
                global_step=0, use_gripper=False, translation_only=True, device="cuda", action_repeat=1, action_dim=7,
                verbose=False, rank=0, **_unused):
    iters, K, lr = int(os.environ.get("GP_ITERS", 40)), int(os.environ.get("GP_K", 8)), float(os.environ.get("GP_LR", 0.03))
    rollout = min(rollout, total_steps - global_step)
    mean0 = torch.zeros(rollout, action_dim)
    if init is not None:
        init = torch.as_tensor(init, dtype=torch.float32)[:rollout]
        mean0[: len(init)] = init
    u = mean0[None].repeat(K, 1, 1)
    u[1:, :, :3] += torch.randn(K - 1, rollout, 3) * robot_init_std
    if use_gripper:
        u[1:, :, 6:] += torch.randn(K - 1, rollout, 1) * gripper_init_std
    u = u.to(device).requires_grad_(True)
    opt = torch.optim.Adam([u], lr=lr)
    for m in (getattr(world_model, "model", None), getattr(world_model, "predictor", None)):
        if isinstance(m, torch.nn.Module):
            m.requires_grad_(False)  # gradients for the actions only
    base = wm_state.repeat(K).to(device)  # the uncached forward does not move the pose tensor itself
    V = base.features.shape[2]
    w = torch.tensor(view_weights if view_weights is not None else [1.0 / V] * V, device=device, dtype=torch.float32)
    best, best_r, rewards, means, stds = None, -float("inf"), [], [], []
    for it in range(iters):
        a = _project(u, maxnorm, maxgripper, translation_only, use_gripper)
        hist, per_view = base, torch.zeros(K, V, device=device)
        for h in range(rollout):
            for _ in range(action_repeat):
                feat, hist = world_model.step(a[:, h:h + 1] / action_repeat, hist, no_grad=False, use_cache=False, device=device)
            if isinstance(goal_features, dict) and (h + global_step + 1) in goal_features:
                per_view = per_view + objective(goal_features[h + global_step + 1].reshape(1, -1).to(device),
                                                feat.reshape(K, -1).float())[:, None]
        if not isinstance(goal_features, dict):
            per_view = objective(goal_features.flatten(3, 5).to(device).float(), feat.flatten(3, 5).float())[:, 0]
        score = (per_view.float() * w).sum(1)
        opt.zero_grad(); (-score.sum()).backward(); opt.step()
        with torch.no_grad():
            i = int(score.argmax())
            if score[i].item() > best_r:
                best_r, best = score[i].item(), a[i].detach().cpu().clone()
            rewards.append(per_view[i].detach().cpu()); means.append(best.clone()); stds.append(torch.zeros_like(best))
        if verbose and rank == 0 and (it % 10 == 0 or it == iters - 1):
            print(f"[gradplan] step {global_step} iter {it}: best score {best_r:.4f} (batch mean {score.mean().item():.4f})", flush=True)
    return rewards, means, stds


def planner(world_model, wm_state, goal_features, **kw):
    mode, t0 = os.environ.get("PLANNER", "grad"), time.time()
    init = kw.get("prev_actions")
    if mode == "cemgrad":
        kw_cem = dict(kw, cem_steps=max(1, int(kw.get("cem_steps", 5)) // 3))
        _, m, _ = _cem(world_model, wm_state, goal_features, **kw_cem)
        init = m[-1]
    out = grad_refine(world_model, wm_state, goal_features, init=init, **kw)
    LOG.append(time.time() - t0)
    print(f"[gradplan] {mode}: planner step {kw.get('global_step')} took {LOG[-1]:.1f}s (mean {sum(LOG) / len(LOG):.1f}s)", flush=True)
    return out


mpc_utils.cem = planner
sys.argv[0] = "policy_client.py"
runpy.run_path("policy_client.py", run_name="__main__")
