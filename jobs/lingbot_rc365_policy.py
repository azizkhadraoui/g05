"""LingBot-VLA 2.0 (post-trained on RoboCasa365 by 135_lingbot_rc365.sh) as a policy for Awomo's RoboCasa365 harness
(baselines/Awomo-0.5/deploy/server.py + eval_robocasa365/dynamic_eval.py), so it is scored by the same evaluator as
Awomo, pi0.5 and GR00T: same task list, seeds, cameras, control rate and 16-step replanning.

Request -> the raw dataset features the robot config (configs/robot_configs/robocasa365.yaml) maps: the 16-D state in
dataset order (the harness sends the same order), the three 256x256 cameras under their dataset keys, the instruction as
the prompt. LingBot's inference server (deploy/lingbot_vla_v2_policy.py) normalises, runs the model and maps the
unified action back to the 12-D dataset action; as for GR00T, base motion and arm deltas are clipped to [-1, 1], the
gripper and base/arm mode binarised at 0 (dataset convention -1 / 1), and the chunk converted to engine order.
Run with the LingBot venv: PYTHONPATH=<Awomo repo>:<jobs> python <Awomo repo>/deploy/server.py --policy lingbot_rc365_policy
--checkpoint <post-training .../hf_ckpt>
"""
from __future__ import annotations

import os
import sys
import time
from typing import Any

import numpy as np

AWOMO = "/export/home/kaziz/g05/baselines/Awomo-0.5"
LINGBOT = "/export/home/kaziz/g05/baselines/lingbot-vla-v2"
sys.path.insert(0, AWOMO)
from eval_robocasa365.common import CAMERA_NAMES, REPLAN_STEPS, parse_request, to_engine_order  # noqa: E402

KEYS = {c: f"observation.images.{c}" for c in CAMERA_NAMES}


class LingbotRC365:
    def __init__(self, checkpoint: str) -> None:
        os.environ.setdefault("QWEN3VL_PATH", "/export/home/kaziz/g05/models/Qwen3-VL-4B-Instruct")
        sys.path.insert(0, LINGBOT)
        cwd = os.getcwd(); os.chdir(LINGBOT)  # robot configs and norm stats are resolved relative to the repo
        try:
            from deploy.lingbot_vla_v2_policy import LingbotVLAv2Server
            self.srv = LingbotVLAv2Server(checkpoint, use_length=-1, chunk_ret=True, use_bf16=True, use_compile=False)
            self.srv.reset(robo_name="robocasa365")
        finally:
            os.chdir(cwd)
        self.metadata = {"model": "lingbot-vla-v2-6b-rc365", "benchmark": "robocasa365", "action_horizon": REPLAN_STEPS,
                         "camera_names": list(CAMERA_NAMES)}

    def act(self, request: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
        started = time.perf_counter()
        _steps, frames, s = parse_request(request)
        f = frames[-1]  # [3 cameras, 256, 256, 3], CAMERA_NAMES order
        obs = {KEYS[c]: np.ascontiguousarray(f[i]) for i, c in enumerate(CAMERA_NAMES)}
        # the server runs LingBot's training-time transform, which reads the instruction from LeRobot's "task" key
        obs.update({"observation.state": np.asarray(s, np.float32), "prompt": str(request["instruction"]), "task": str(request["instruction"])})
        out = self.srv.infer(obs)
        a = np.asarray(next(iter(out.values())), np.float32)[:REPLAN_STEPS]  # [16, 12], dataset order
        a12 = np.concatenate([np.clip(a[:, 0:4], -1, 1), np.where(a[:, 4:5] > 0, 1.0, -1.0), np.clip(a[:, 5:11], -1, 1),
                              np.where(a[:, 11:12] > 0, 1.0, -1.0)], axis=1).astype(np.float32)
        return to_engine_order(a12), {"latency_seconds": time.perf_counter() - started}


def load_policy(checkpoint: str, **_kwargs) -> LingbotRC365:
    return LingbotRC365(checkpoint)
