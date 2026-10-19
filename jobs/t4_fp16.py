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
