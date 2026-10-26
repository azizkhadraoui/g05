"""On GPUs without native bf16 (T4, sm75): map bf16 autocast to fp16. Import before any model code. No-op on Ampere+."""
import torch

T4_MODE = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] < 8
if T4_MODE:
    _Orig = torch.autocast

    class _Autocast16(_Orig):
        def __init__(self, device_type, dtype=None, *args, **kwargs):
            if dtype == torch.bfloat16:
                dtype = torch.float16
            super().__init__(device_type, dtype, *args, **kwargs)

    torch.autocast = _Autocast16
    torch.amp.autocast = _Autocast16

# V100 (sm70): fla's Triton chunk kernel for the gated delta rule fails to compile for some prompt lengths ("PassManager::run
# failed"). G0.5's Qwen3.5 linear-attention layers pick their kernel when they are built, so point them at their own fp32
# PyTorch implementation before the model is constructed (prefill only; decoding is unaffected).
if torch.cuda.is_available() and torch.cuda.get_device_capability() == (7, 0):
    try:
        import g05.models.g05.qwen35.gated_deltanet as _gdn
        _gdn.fla_chunk_gated_delta_rule = None
        print("[v100] gated delta rule: PyTorch chunk implementation (fla Triton kernel off)", flush=True)
    except ImportError:
        pass
