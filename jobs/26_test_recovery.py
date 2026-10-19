"""Validate the recovery-dataset path on synthetic recordings before real data exists."""
import os
import shutil
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import importlib

T = importlib.import_module("07_train_hybrid")
from omegaconf import OmegaConf  # noqa: E402

tmp = "/export/home/kaziz/g05/results/_rec_test"
shutil.rmtree(tmp, ignore_errors=True)
os.makedirs(f"{tmp}/src/t0_s1/rec")
for e in range(2):
    n = 50
    np.savez_compressed(f"{tmp}/src/t0_s1/rec/task0_trial{e}.npz", step=np.arange(40, 40 + n),
                        image=np.random.randint(0, 255, (n, 256, 256, 3), np.uint8),
                        wrist_image=np.random.randint(0, 255, (n, 256, 256, 3), np.uint8),
                        state=np.random.randn(n, 8).astype(np.float32) * 0.1,
                        action=np.concatenate([np.random.randn(n, 6) * 0.3, np.random.randint(0, 2, (n, 1))], 1).astype(np.float32),
                        prompt=np.array(["open the middle drawer of the cabinet"] * n),
                        task_description="open the middle drawer of the cabinet")
cfg = T.load_config_from_run_dir(T.find_run_dir(T.CKPT), T.CKPT, [])
if cfg.get("eval_embodiment") and "embodiment_datasets" in cfg.data:
    T.filter_embodiment(cfg, cfg.eval_embodiment)
OmegaConf.set_struct(cfg, False)
cfg.data = OmegaConf.load(T.DATA_CFG)
ds = T.instantiate_dataset(cfg, is_training_set=True)
proc = T.build_processors(cfg)
proc.set_normalizer_from_stats(T.load_dataset_stats_from_json(T.STATS))
ds.set_processor(proc)
rec = T.make_recovery_dataset(ds, proc, f"{tmp}/src", f"{tmp}/cache")
print("recovery frames", len(rec), "episodes", rec.n_episodes)
a, b = rec[0], ds[1000]
for k in ("pixel_values", "action", "action_is_pad", "action_dim_is_pad", "proprio"):
    va, vb = a[k], b[k]
    if isinstance(va, dict):
        print(k, {kk: tuple(vv.shape) for kk, vv in va.items()}, "| base", {kk: tuple(vv.shape) for kk, vv in vb.items()})
    else:
        print(k, tuple(va.shape), va.dtype, "| base", tuple(vb.shape), vb.dtype)
print("command:", a["samples"]["command"])
print("end-of-episode pad:", rec[49]["action_is_pad"].sum().item(), "of 32")
batch = T.collate_fn_pad_sequences([rec[0], ds[5], rec[10]], padding_input_id=proc.pad_token_id)
print("mixed collate ok:", list(batch.keys())[:8])
print("RECOVERY_TEST_OK")
