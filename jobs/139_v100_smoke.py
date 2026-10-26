"""V100 smoke test for the G0.5 env (139_v100_env.sh): kernels, fp16 matmul, LIBERO rendering, Qwen3-VL import."""
import pathlib
import sys

import numpy as np
import PIL
import torch

print("torch", torch.__version__, torch.cuda.get_device_name(0), torch.cuda.get_arch_list(), "numpy", np.__version__, "pillow", PIL.__version__)
x = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)
print("fp16 matmul ok:", float((x @ x).float().abs().mean()))

from libero.libero import benchmark, get_libero_path  # noqa: E402
from libero.libero.envs import OffScreenRenderEnv  # noqa: E402

suite = benchmark.get_benchmark_dict()[sys.argv[1]]()
task = suite.get_task(0)
env = OffScreenRenderEnv(bddl_file_name=str(pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file),
                         camera_heights=256, camera_widths=256)
env.seed(0)
obs = env.reset()
for _ in range(5):
    obs, *_ = env.step(np.zeros(7))
print(sys.argv[1], "render ok:", obs["agentview_image"].shape, float(obs["agentview_image"].mean()))
env.close()

import transformers  # noqa: E402
from transformers import AutoModelForImageTextToText  # noqa: E402,F401  (robobrain.py loads RoboBrain 2.5 / Qwen3-VL this way)

print("transformers", transformers.__version__, "VLM loader import ok")
