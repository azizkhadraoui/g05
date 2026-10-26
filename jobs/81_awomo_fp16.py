"""Launch Awomo's deploy/server.py with an optional dtype swap (T4/V100 have no native bf16; emulated bf16 runs ~7x slower
than fp16 on a T4: 4.3 vs 30.1 TFLOP/s).

AWOMO_DTYPE=fp16 makes Awomo-0.5I's model code (awomo/awomo_05i/model.py: DiT, FLUX.2 VAE, Qwen3-VL) build and run in
float16. Only that module sees the swap, through a proxy for its `torch` name, so torch itself (e.g. the checkpoint
unpickler, which maps storages by dtype) is untouched; bf16 weights are cast on load_state_dict / from_pretrained.
bf16 (default) leaves the released code untouched. Used by 80_awomo_eval.sh via AWOMO_LAUNCH; run from the Awomo-0.5 root.
"""
import os
import runpy
import sys
import types

import torch

if os.environ.get("AWOMO_DTYPE", "bf16") == "fp16":
    sys.path.insert(0, os.getcwd())
    import awomo.awomo_05i.model as M

    class _TorchFP16(types.ModuleType):
        def __getattr__(self, name):
            return getattr(torch, name)

    proxy = _TorchFP16("torch")
    proxy.bfloat16 = torch.float16
    M.torch = proxy
    print("[awomo] model code runs in float16", flush=True)
sys.argv = ["deploy/server.py"] + sys.argv[1:]
runpy.run_path("deploy/server.py", run_name="__main__")
