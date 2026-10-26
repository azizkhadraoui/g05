"""GR00T N1.7 (fine-tuned on RoboCasa365 by 84_gr00t_finetune.sh) as a policy for Awomo's RoboCasa365 harness
(baselines/Awomo-0.5/deploy/server.py + eval_robocasa365/dynamic_eval.py), so every model is scored by the same
evaluator: same task list, seeds, cameras, control rate and 16-step replanning.

Request -> GR00T observation: the latest frame of the 3 cameras (agentview left / right, eye in hand, 256x256), the
16-D state split into the dataset's 5 state keys, and the instruction. GR00T's 5 action keys (16-step chunk, dataset
units) -> the 12-D RoboCasa action in the harness's engine order: base motion and arm deltas clipped to [-1, 1],
gripper and base/arm mode binarised at 0 (dataset convention -1 / 1).
Run with the GR00T venv: PYTHONPATH=<Awomo repo>:<jobs> python <Awomo repo>/deploy/server.py --policy gr00t_rc365_policy
--checkpoint <GR00T checkpoint dir>   (GR00T_FP16=1 casts the model to fp16 for T4/V100)
"""
from __future__ import annotations

import importlib.util
import os
import sys
import time
import types
from typing import Any

import huggingface_hub
import numpy as np
import torch

AWOMO = "/export/home/kaziz/g05/baselines/Awomo-0.5"
sys.path.insert(0, AWOMO)
sys.modules.setdefault("flash_attn", None)
_model_info = huggingface_hub.model_info


def _model_info_offline(repo_id, *args, **kwargs):
    """The backbone tokenizer loader asks the Hub whether the model descends from Mistral (is_base_mistral); offline
    that raises instead of being skipped. Same patch as 83_gr00t_v100_launch.py."""
    try:
        return _model_info(repo_id, *args, **kwargs)
    except Exception:  # noqa: BLE001 - offline / no network
        return types.SimpleNamespace(tags=None)


huggingface_hub.model_info = _model_info_offline
from eval_robocasa365.common import CAMERA_NAMES, REPLAN_STEPS, parse_request, to_engine_order  # noqa: E402

_spec = importlib.util.spec_from_file_location("rc365_config", "/export/home/kaziz/g05/jobs/82_gr00t_rc365_config.py")
_cfg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_cfg)  # registers the RoboCasa365 modality config under NEW_EMBODIMENT

STATE_SLICES = {"base_position": (0, 3), "base_rotation": (3, 7), "end_effector_position_relative": (7, 10),
                "end_effector_rotation_relative": (10, 14), "gripper_qpos": (14, 16)}
VIDEO_KEYS = ("robot0_agentview_left", "robot0_agentview_right", "robot0_eye_in_hand")  # = CAMERA_NAMES order


class Gr00tRC365:
    def __init__(self, checkpoint: str, device: str = "cuda") -> None:
        from gr00t.policy.gr00t_policy import Gr00tPolicy
        self.policy = Gr00tPolicy(embodiment_tag="new_embodiment", model_path=checkpoint, device=device, strict=False)
        if os.environ.get("GR00T_FP16", "1") == "1":
            self.policy.model.to(torch.float16)
        self.metadata = {"model": "gr00t-n1.7-rc365", "benchmark": "robocasa365", "action_horizon": REPLAN_STEPS,
                         "camera_names": list(CAMERA_NAMES)}

    def act(self, request: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
        started = time.perf_counter()
        _steps, frames, state16 = parse_request(request)
        f = frames[-1]  # [3 cameras, 256, 256, 3], the most recent observation
        obs = {"video": {k: f[i][None, None] for i, k in enumerate(VIDEO_KEYS)},
               "state": {k: state16[a:b][None, None].astype(np.float32) for k, (a, b) in STATE_SLICES.items()},
               "language": {"annotation.human.task_description": [[str(request["instruction"])]]}}
        with torch.autocast("cuda", dtype=torch.float16, enabled=os.environ.get("GR00T_FP16", "1") == "1"):
            action, _ = self.policy.get_action(obs)
        g = lambda k: np.asarray(action[k], np.float32)[0]  # noqa: E731  [16, d]
        a12 = np.concatenate([np.clip(g("base_motion"), -1, 1), np.where(g("control_mode") > 0, 1.0, -1.0),
                              np.clip(g("end_effector_position"), -1, 1), np.clip(g("end_effector_rotation"), -1, 1),
                              np.where(g("gripper_close") > 0, 1.0, -1.0)], axis=1).astype(np.float32)
        return to_engine_order(a12[:REPLAN_STEPS]), {"latency_seconds": time.perf_counter() - started}


def load_policy(checkpoint: str, **_kwargs) -> Gr00tRC365:
    return Gr00tRC365(checkpoint)
