"""LingBot-VLA 2.0 6B smoke test on one A100: load the released checkpoint through the repo's own inference server class
(deploy/lingbot_vla_v2_policy.py) with the bundled RoboTwin robot config, run dummy observations, report the action
chunk shape, inference latency and peak GPU memory. Checks the environment (flash-attention, Qwen3-VL backbone) before
the RoboCasa365 port and post-training.
"""
import os
import sys
import time

import numpy as np
import torch

L = "/export/home/kaziz/g05/baselines/lingbot-vla-v2"
sys.path.insert(0, L); os.chdir(L)
os.environ.setdefault("QWEN3VL_PATH", "/export/home/kaziz/g05/models/Qwen3-VL-4B-Instruct")
from deploy.lingbot_vla_v2_policy import LingbotVLAv2Server  # noqa: E402

t0 = time.time()
srv = LingbotVLAv2Server("/export/home/kaziz/g05/models/lingbot-vla-v2-6b", use_length=50, chunk_ret=True, use_bf16=True,
                         use_compile=False)
srv.reset(robo_name="robotwin")
print(f"loaded in {time.time() - t0:.0f}s; GPU memory {torch.cuda.memory_allocated() / 1e9:.1f} GB", flush=True)
rng = np.random.default_rng(0)
obs = {"observation.state": rng.normal(0, 0.1, 14).astype(np.float32),
       "observation.images.cam_high": rng.integers(0, 255, (480, 640, 3), dtype=np.uint8),
       "observation.images.cam_left_wrist": rng.integers(0, 255, (480, 640, 3), dtype=np.uint8),
       "observation.images.cam_right_wrist": rng.integers(0, 255, (480, 640, 3), dtype=np.uint8),
       "prompt": "pick up the cup and place it on the plate"}
for i in range(6):
    torch.cuda.synchronize(); t = time.time()
    act = srv.infer(dict(obs))
    torch.cuda.synchronize()
    print(f"call {i}: {1000 * (time.time() - t):.0f} ms; " + ", ".join(f"{k} {tuple(np.asarray(v).shape)}" for k, v in act.items()), flush=True)
print(f"peak GPU memory {torch.cuda.max_memory_allocated() / 1e9:.1f} GB", flush=True)
print("LINGBOT_SMOKE_OK", flush=True)
