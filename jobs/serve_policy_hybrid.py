"""G0.5 batched policy server with optional LaWAM conditioning attached (no edits to GalaxeaVLA).

Same CLI as scripts/serve_policy_batched.py plus:
  --hybrid_ckpt PATH      trained adapter (+prior) state; omit to serve the plain released policy
  --hybrid_branch B       current (F01) | future (F02)
  --future_mode M         normal | current | shuffled | suppressed   (F03 evaluation-time controls)
  --seed S                torch seed for FM sampling noise
"""
import os
import sys

JOBS = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.join(os.environ.get("G05_ROOT", "/export/home/kaziz/g05"), "GalaxeaVLA")


def pop_arg(name, default=None):
    if name in sys.argv:
        i = sys.argv.index(name)
        val = sys.argv[i + 1]
        del sys.argv[i:i + 2]
        return val
    return default


HY_CKPT = pop_arg("--hybrid_ckpt")
HY_BRANCH = pop_arg("--hybrid_branch", "future")
FUTURE_MODE = pop_arg("--future_mode", "normal")
SEED = int(pop_arg("--seed", "7"))
EXPERT_CKPT = pop_arg("--expert_ckpt")  # fine-tuned action-expert weights (07b_train_lmem.py --unfreeze_expert)

sys.path.insert(0, JOBS)
sys.path.insert(0, PROJECT)
os.chdir(PROJECT)

import torch  # noqa: E402
import t4_fp16  # noqa: E402,F401  (bf16 -> fp16 autocast on T4; no-op on H200)

import scripts.serve_policy_batched as S  # noqa: E402



def disable_flash_attn():
    """FlashAttention kernels need sm80+: route G0.5's vision tower and action codec to their SDPA paths (T4 only)."""
    for _name, _mod in list(sys.modules.items()):
        if "g05" in _name:
            for _attr in ("_flash_attn_varlen", "flash_attn_func"):
                if getattr(_mod, _attr, None) is not None:
                    setattr(_mod, _attr, None)
                    print(f"[t4] {_name}.{_attr} -> SDPA fallback", flush=True)


if t4_fp16.T4_MODE:
    disable_flash_attn()

_orig_setup = S.setup


def hybrid_setup(cfg, device="cuda"):
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
                _m.register_forward_pre_hook(_cast_in)  # free the fp32 blocks so cuBLAS can allocate its workspace
        print(f"[t4] policy cast to fp16, bf16 autocast mapped to fp16; "
              f"allocated={torch.cuda.memory_allocated() / 2**30:.1f} GiB", flush=True)
    if EXPERT_CKPT:
        expert = next(m for n, m in policy.named_modules() if n.endswith("action_expert"))
        sd = torch.load(EXPERT_CKPT, map_location="cpu")
        dt = next(expert.parameters()).dtype
        missing, unexpected = expert.load_state_dict({k: v.to(dt) for k, v in sd.items()}, strict=True), None
        print(f"[lmem] loaded fine-tuned action expert from {EXPERT_CKPT}", flush=True)
    if HY_CKPT:
        import lawam_hybrid as H

        hybrid = H.build_hybrid("adapter", HY_BRANCH, ckpt=HY_CKPT, device=device)
        hybrid.eval()
        hybrid.future_mode = FUTURE_MODE
        H.attach(policy, hybrid)
        if HY_BRANCH == "current" and t4_fp16.T4_MODE:  # current-frame adapter only needs DINOv3: keep the rest of LaWM on CPU
            for _n, _c in hybrid.lawm.named_children():
                if _n != "vision_encoder":
                    _c.to("cpu")
            torch.cuda.empty_cache()
            print(f"[t4] LaWM offloaded except vision_encoder; allocated={torch.cuda.memory_allocated() / 2**30:.1f} GiB", flush=True)
        print(f"[hybrid] attached: ckpt={HY_CKPT} branch={HY_BRANCH} future_mode={FUTURE_MODE} "
              f"adapter_params={hybrid.n_params('adapter')} inject_layers={hybrid.adapter.inject_layers}", flush=True)
    else:
        print("[hybrid] none: serving the released policy", flush=True)
    return policy, processor


S.setup = hybrid_setup
if __name__ == "__main__":
    S.main()
