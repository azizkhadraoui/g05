"""G0.5 batched policy server for the pi0.7-inspired experiments (no edits to GalaxeaVLA or serve_policy_hybrid.py).

Same CLI as scripts/serve_policy_batched.py plus:
  --gc_ckpt PATH   goal-conditioned adapter (gc_train.py); omit to serve the plain released policy
  --expert_ckpt P  fine-tuned action expert (gc_train.py --unfreeze_expert)
  --seed S         torch seed for FM sampling noise

Extra per-request fields a client may put in raw_obs (all optional; requests without them behave as before):
  goal_feat   float16 [64, 768]  pooled DINOv3 goal features for the goal adapter (absent = no goal)
  cfg_neg     str                negative prompt for classifier-free guidance on the flow velocity
  cfg_w       float              guidance scale w: v = v(neg) + w * (v(task) - v(neg)); w = 1 is plain v(task)
"""
import os
import sys
import time

JOBS = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.join(os.environ.get("G05_ROOT", "/export/home/kaziz/g05"), "GalaxeaVLA")


def pop_arg(name, default=None):
    if name in sys.argv:
        i = sys.argv.index(name)
        val = sys.argv[i + 1]
        del sys.argv[i:i + 2]
        return val
    return default


GC_CKPT = pop_arg("--gc_ckpt")
EXPERT_CKPT = pop_arg("--expert_ckpt")  # fine-tuned action-expert weights (gc_train.py --unfreeze_expert)
SEED = int(pop_arg("--seed", "7"))

sys.path.insert(0, JOBS)
sys.path.insert(0, PROJECT)
os.chdir(PROJECT)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import t4_fp16  # noqa: E402,F401  (bf16 -> fp16 autocast on T4/V100; no-op on H200)

import scripts.serve_policy_batched as S  # noqa: E402
import gc_hybrid as GC  # noqa: E402

HYBRID = None


def disable_flash_attn():
    """FlashAttention kernels need sm80+: route G0.5's vision tower and action codec to their SDPA paths."""
    for _name, _mod in list(sys.modules.items()):
        if "g05" in _name:
            for _attr in ("_flash_attn_varlen", "flash_attn_func"):
                if getattr(_mod, _attr, None) is not None:
                    setattr(_mod, _attr, None)


if t4_fp16.T4_MODE:
    disable_flash_attn()

_orig_setup = S.setup


def gc_setup(cfg, device="cuda"):
    global HYBRID
    torch.manual_seed(SEED)
    policy, processor = _orig_setup(cfg, device=device)
    if t4_fp16.T4_MODE:
        disable_flash_attn()
        policy.half()
        torch.cuda.empty_cache()

        def _cast_in(mod, args):  # fp32 activations created outside autocast -> match the fp16 weights
            if args and torch.is_tensor(args[0]) and args[0].is_floating_point() and args[0].dtype != mod.weight.dtype:
                return (args[0].to(mod.weight.dtype),) + tuple(args[1:])
        for _m in policy.modules():
            if isinstance(_m, (torch.nn.Linear, torch.nn.Conv2d, torch.nn.Conv3d, torch.nn.LayerNorm)) and getattr(_m, "weight", None) is not None:
                _m.register_forward_pre_hook(_cast_in)
    GC.install_cfg(policy)
    if EXPERT_CKPT:
        expert = next(m for n, m in policy.named_modules() if n.endswith("action_expert"))
        dt = next(expert.parameters()).dtype
        sd = torch.load(EXPERT_CKPT, map_location="cpu")
        expert.load_state_dict({k: v.to(dt) for k, v in sd.items()}, strict=True)
        print(f"[gc] loaded fine-tuned action expert from {EXPERT_CKPT}", flush=True)
    if GC_CKPT:
        import lawam_hybrid as H

        HYBRID = GC.build_goal_hybrid(ckpt=GC_CKPT, device=device)
        HYBRID.eval()
        H.attach(policy, HYBRID)
        for _n, _c in HYBRID.lawm.named_children():  # the goal adapter only needs DINOv3 from LaWM
            if _n != "vision_encoder":
                _c.to("cpu")
        torch.cuda.empty_cache()
        print(f"[gc] goal adapter attached: {GC_CKPT}; allocated={torch.cuda.memory_allocated() / 2**30:.1f} GiB", flush=True)
    else:
        print("[gc] no adapter: serving the released policy (CFG available)", flush=True)
    return policy, processor


def _process_batch(self, requests):
    """DynamicBatcher._process_batch plus goal features and CFG negative-prompt rows."""
    outcomes, valid, obs_dicts, extras = [], [], [], []
    for req in requests:
        ro = req.raw_obs
        ex = {"goal": ro.pop("goal_feat", None), "neg": ro.pop("cfg_neg", None), "w": float(ro.pop("cfg_w", 1.0))}
        try:
            S._parse_task_and_plan(ro)
            od = S.build_obs_dict(ro, self.processor)
        except Exception as exc:
            outcomes.append(S.RequestOutcome(request=req, error=exc))
            continue
        valid.append(req); obs_dicts.append(od); extras.append(ex)
    if not obs_dicts:
        return outcomes
    n = len(obs_dicts)
    pairs = []
    for i in range(n):
        ex = extras[i]
        if ex["neg"] is not None and ex["w"] != 1.0:
            ro = dict(valid[i].raw_obs)
            ro["task"] = str(ex["neg"])
            S._parse_task_and_plan(ro)
            obs_dicts.append(S.build_obs_dict(ro, self.processor))
            extras.append(ex)  # the negative row sees the same goal
            pairs.append((i, len(obs_dicts) - 1, ex["w"]))
    if HYBRID is not None:
        goals = [ex["goal"] for ex in extras]
        on = torch.tensor([g is not None for g in goals], dtype=torch.bool)
        HYBRID.goal_feats = torch.from_numpy(np.stack([np.asarray(g, np.float32) if g is not None
                                                       else np.zeros((64, 768), np.float32) for g in goals]))
        HYBRID.goal_on = on
    GC.CFGState.pairs = pairs
    t0 = time.monotonic()
    try:
        actions, _timing = self.inferencer.infer_with_timing(obs_dicts)
    except Exception as exc:
        S.logger.exception("Dynamic batch inference failed")
        outcomes.extend(S.RequestOutcome(request=r, error=exc) for r in valid)
        return outcomes
    finally:
        GC.CFGState.pairs = []
        if HYBRID is not None:
            HYBRID.goal_feats = HYBRID.goal_on = None
    for req, action in zip(valid, actions[:n]):
        outcomes.append(S.RequestOutcome(request=req, action=action))
    S.logger.info("Batch forward: batch_size=%d (+%d cfg rows, %d with goal), infer=%.1fms", n, len(pairs),
                  sum(ex["goal"] is not None for ex in extras[:n]), (time.monotonic() - t0) * 1000.0)
    return outcomes


S.setup = gc_setup
S.DynamicBatcher._process_batch = _process_batch
if __name__ == "__main__":
    S.main()
