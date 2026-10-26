"""Run GR00T N1.7's gr00t/experiment/launch_finetune.py on V100/T4 (no bf16, no TF32, no FlashAttention-2).

launch_finetune.py builds its config inside `if __name__ == "__main__"`, so this wrapper executes that file with one
hook inserted just before `run(config)`: fp16 mixed precision instead of bf16/TF32, and the local Cosmos-Reason2-2B
backbone. flash_attn is hidden so the backbone falls back to SDPA. Same CLI as launch_finetune.py; run from the repo root.
"""
import os
import sys
import types

import huggingface_hub

sys.modules["flash_attn"] = None  # `import flash_attn` -> ImportError -> SDPA
_model_info = huggingface_hub.model_info


def _model_info_offline(repo_id, *args, **kwargs):
    """transformers' tokenizer loader asks the Hub whether a model descends from Mistral (is_base_mistral); offline
    that query raises instead of being skipped. Answer 'no tags' when the Hub is unreachable."""
    try:
        return _model_info(repo_id, *args, **kwargs)
    except Exception:  # noqa: BLE001 - offline / no network
        return types.SimpleNamespace(tags=None)


huggingface_hub.model_info = _model_info_offline

if os.environ.get("GR00T_BACKBONE_FP16", "1") == "1":  # frozen backbone in fp16: halves its memory (fits a 16 GB T4)
    import torch
    import transformers
    _qwen_fp = transformers.Qwen3VLForConditionalGeneration.from_pretrained.__func__

    @classmethod
    def _qwen_fp16(cls, *args, **kwargs):
        kwargs["torch_dtype"] = torch.float16
        return _qwen_fp(cls, *args, **kwargs)
    transformers.Qwen3VLForConditionalGeneration.from_pretrained = _qwen_fp16
PATH = "gr00t/experiment/launch_finetune.py"
BACKBONE = "/export/home/kaziz/g05/models/Cosmos-Reason2-2B"


def v100_patch(config):
    t = config.training
    t.fp16, t.bf16, t.eval_bf16, t.tf32 = True, False, False, False
    config.model.model_name = BACKBONE
    # 1.6B trainable parameters: fp32 weights + grads + AdamW moments ~26 GB, over a V100's 32 GB with activations;
    # GR00T_OPTIM=adamw_bnb_8bit keeps AdamW with 8-bit moments (bitsandbytes), ~19 GB
    t.optim = os.environ.get("GR00T_OPTIM", t.optim)
    print(f"[v100] fp16={t.fp16} bf16={t.bf16} tf32={t.tf32} optim={t.optim} backbone={BACKBONE}", flush=True)


src = open(PATH).read()
assert src.count("    run(config)") == 1, "launch_finetune.py changed: hook point not found"
src = src.replace("    run(config)", "    _v100_patch(config)\n    run(config)")
sys.argv[0] = PATH
exec(compile(src, PATH, "exec"), {"__name__": "__main__", "__file__": PATH, "_v100_patch": v100_patch})
