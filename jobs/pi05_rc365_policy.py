"""pi0.5 multitask, the official RoboCasa365 checkpoint (robocasa/robocasa365_checkpoints:
pi05_pretrain_human300/multitask_learning/75000, robocasa-benchmark/openpi fork), as a policy for Awomo's RoboCasa365
harness (baselines/Awomo-0.5/deploy/server.py + eval_robocasa365/dynamic_eval.py): same task list, seeds, cameras,
control rate and 16-step replanning as every other model.

Request -> the observation the fork's own evaluation builds (examples/robocasa/main.py): agentview left as the base
image, eye-in-hand as the wrist image, agentview right as the right image (resize_with_pad to 224), the 16-D state
reordered to [eef_pos_rel(3), eef_rot_rel(4), base_pos(3), base_rot(4), gripper_qpos(2)] (the harness sends base first),
and the instruction as the prompt. The model returns a 50-step chunk of 12-D actions already in
robocasa.utils.env_utils.convert_action order, which is the harness's engine order; the first 16 are returned unchanged
(the fork's evaluation also passes them to the environment raw; it replans every 5 steps, the harness every 16).
Run with the rcb-openpi venv: PYTHONPATH=<Awomo repo>:<jobs> python <Awomo repo>/deploy/server.py
--policy pi05_rc365_policy --checkpoint <.../75000>
"""
from __future__ import annotations

import dataclasses
import os
import sys
import time
from typing import Any

import numpy as np

AWOMO = "/export/home/kaziz/g05/baselines/Awomo-0.5"
sys.path.insert(0, AWOMO)
from eval_robocasa365.common import CAMERA_NAMES, REPLAN_STEPS, parse_request  # noqa: E402

LEFT, RIGHT, WRIST = (CAMERA_NAMES.index(c) for c in ("robot0_agentview_left", "robot0_agentview_right", "robot0_eye_in_hand"))


class Pi05RC365:
    def __init__(self, checkpoint: str) -> None:
        # the fork's training config imports robocasa's dataset registry; robocasa/__init__.py asserts numpy == 2.2.5
        # and imports the simulator stack, neither needed here (the simulator runs in its own env). Register the
        # package by its path without running __init__, so only the registry submodules load.
        if "robocasa" not in sys.modules:
            import importlib.util
            import types
            spec = importlib.util.find_spec("robocasa")
            pkg = types.ModuleType("robocasa")
            pkg.__path__, pkg.__file__, pkg.__spec__ = list(spec.submodule_search_locations), spec.origin, spec
            sys.modules["robocasa"] = pkg
        from openpi.policies import policy_config
        from openpi.shared import normalize
        from openpi.training import config as _config
        from openpi_client import image_tools
        cfg = _config.get_config("pi05_pretrain_human300")
        # norm stats come from the checkpoint (as in the fork's policy_config); the 300 pretraining datasets the
        # config lists are not on disk, so the data-dir fallback is switched off
        cfg = dataclasses.replace(cfg, data=dataclasses.replace(cfg.data, data_dirs=None))
        self.policy = policy_config.create_trained_policy(
            cfg, checkpoint, norm_stats=normalize.load(os.path.join(checkpoint, "assets")))
        self.tools = image_tools
        self.metadata = {"model": "pi05-robocasa365-pretrain-human300", "benchmark": "robocasa365",
                         "action_horizon": REPLAN_STEPS, "camera_names": list(CAMERA_NAMES)}

    def act(self, request: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
        started = time.perf_counter()
        _steps, frames, s = parse_request(request)
        f = frames[-1]  # [3 cameras, 256, 256, 3], the most recent observation, CAMERA_NAMES order

        def img(i):
            return self.tools.convert_to_uint8(self.tools.resize_with_pad(f[i], 224, 224))
        state = np.concatenate([s[7:10], s[10:14], s[0:3], s[3:7], s[14:16]]).astype(np.float32)
        out = self.policy.infer({"observation/image": img(LEFT), "observation/wrist_image": img(WRIST),
                                 "observation/right_image": img(RIGHT), "observation/state": state,
                                 "prompt": str(request["instruction"])})
        actions = np.asarray(out["actions"], dtype=np.float32)[:REPLAN_STEPS]
        return np.ascontiguousarray(actions), {"latency_seconds": time.perf_counter() - started}


def load_policy(checkpoint: str, **_kwargs) -> Pi05RC365:
    return Pi05RC365(checkpoint)
