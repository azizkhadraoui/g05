"""Dump the exact format of one inner LeRobot sample and one processed sample (to build the recovery dataset)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import importlib

T = importlib.import_module("07_train_hybrid")
from omegaconf import OmegaConf  # noqa: E402
import torch  # noqa: E402

cfg = T.load_config_from_run_dir(T.find_run_dir(T.CKPT), T.CKPT, [])
if cfg.get("eval_embodiment") and "embodiment_datasets" in cfg.data:
    T.filter_embodiment(cfg, cfg.eval_embodiment)
OmegaConf.set_struct(cfg, False)
cfg.data = OmegaConf.load(T.DATA_CFG)
ds = T.instantiate_dataset(cfg, is_training_set=True)
proc = T.build_processors(cfg)
proc.set_normalizer_from_stats(T.load_dataset_stats_from_json(T.STATS))
ds.set_processor(proc)


def desc(x, pre=""):
    if isinstance(x, dict):
        for k, v in x.items():
            desc(v, f"{pre}{k}.")
    elif isinstance(x, torch.Tensor):
        print(f"  {pre[:-1]}: Tensor {tuple(x.shape)} {x.dtype} min={x.float().min().item():.3f} max={x.float().max().item():.3f}")
    else:
        print(f"  {pre[:-1]}: {type(x).__name__} {str(x)[:80]}")


print("type(ds)", type(ds), "n inner", len(ds.datasets))
inner = ds.datasets[0]
print("inner", type(inner), [type(d) for d in getattr(inner.multi_dataset, "_datasets", [])][:2])
print("== lerobot_sample = inner.multi_dataset[1000]")
ls = inner.multi_dataset[1000]
desc(ls)
print("== inner attrs", [k for k in ("state_meta", "action_meta", "image_meta", "model_fps", "_start_idx") if hasattr(inner, k)])
print("state_meta", inner.state_meta); print("action_meta", inner.action_meta); print("image_meta", inner.image_meta)
print("== processed ds[1000]")
desc(ds[1000])
