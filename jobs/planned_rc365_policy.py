"""High-level planner + low-level VLA as one policy for Awomo's RoboCasa365 harness: the planner (128_planner_server.py,
Qwen3-VL-4B) splits the episode's task into short single-step instructions and decides when each is done; the low-level
policy (pi05_rc365_policy.py by default) only ever sees the current short instruction. Same harness, seeds and task list
as the plain low-level run, so the two compare trial by trial.

Per episode (keyed by the request's session id): at the first request and then every PLAN_EVERY policy steps, the planner
gets the agent-view and wrist images, the overall task, the steps already done and the current step, and replies
  DONE: yes|no        whether the current step is finished
  NEXT: <instruction> the next single-step instruction, or "finished"
A "finished" answer keeps the last instruction (the harness decides success). Env: INNER_POLICY (module, default
pi05_rc365_policy), PLANNER_PORT, PLAN_EVERY (64 steps = 3.2 s at 20 Hz), PLAN_LOG (jsonl of planner calls)
"""
from __future__ import annotations

import importlib
import json
import os
import re
import time
from multiprocessing.connection import Client
from typing import Any

import numpy as np

from pi05_rc365_policy import AWOMO  # noqa: F401  (puts the Awomo repo on sys.path)
from eval_robocasa365.common import CAMERA_NAMES, parse_request  # noqa: E402

LEFT, WRIST = CAMERA_NAMES.index("robot0_agentview_left"), CAMERA_NAMES.index("robot0_eye_in_hand")
PROMPT = """You supervise a mobile kitchen robot that can only follow ONE short instruction at a time, such as
"pick the apple from the counter and place it in the bowl", "open the left cabinet door", "turn on the sink faucet",
"navigate to the stove". Overall task: "{task}"
Steps completed so far: {done}
Current step: {cur}
Look at the two camera images (scene view, wrist view). Reply with exactly two lines:
DONE: yes or no (is the current step finished?)
NEXT: the next short instruction for the robot (one action, same style as the examples), or "finished" if the overall task is complete"""


class Planned:
    def __init__(self, checkpoint: str) -> None:
        self.inner = importlib.import_module(os.environ.get("INNER_POLICY", "pi05_rc365_policy")).load_policy(checkpoint)
        self.metadata = {**self.inner.metadata, "model": self.inner.metadata.get("model", "") + "+planner-qwen3vl4b"}
        self.conn = Client(("127.0.0.1", int(os.environ["PLANNER_PORT"])), authkey=b"g05-planner")
        self.every = int(os.environ.get("PLAN_EVERY", "64"))
        self.log = open(os.environ["PLAN_LOG"], "a") if os.environ.get("PLAN_LOG") else None
        self.sessions: dict[str, dict] = {}

    def _ask(self, frames, task, done, cur):
        prompt = PROMPT.format(task=task, done="; ".join(done) if done else "none", cur=cur or "(none yet: give the first step)")
        self.conn.send({"images": [frames[LEFT], frames[WRIST]], "prompt": prompt, "max_new_tokens": 48})
        r = self.conn.recv()
        txt = r["text"]
        is_done = bool(re.search(r"DONE:\s*yes", txt, re.I))
        m = re.search(r"NEXT:\s*(.+)", txt, re.I)
        nxt = m.group(1).strip().strip('"').strip() if m else ""
        return is_done, nxt, txt, r["seconds"]

    def act(self, request: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
        steps, frames, _ = parse_request(request)
        sid, task, step = str(request["session_id"]), str(request["instruction"]), int(steps[-1])
        s = self.sessions.setdefault(sid, {"cur": None, "done": [], "last_plan": -10 ** 9, "calls": 0})
        f = frames[-1]
        if s["cur"] is None or step - s["last_plan"] >= self.every:
            is_done, nxt, txt, sec = self._ask(f, task, s["done"], s["cur"])
            s["last_plan"], s["calls"] = step, s["calls"] + 1
            if s["cur"] is None:
                s["cur"] = nxt if nxt and nxt.lower() != "finished" else task
            elif is_done and nxt:
                s["done"].append(s["cur"])
                if nxt.lower() != "finished":
                    s["cur"] = nxt
            if self.log:
                self.log.write(json.dumps({"session": sid, "step": step, "task": task, "answer": txt, "current": s["cur"],
                                           "done": s["done"], "seconds": round(sec, 2)}) + "\n"); self.log.flush()
        req = dict(request); req["instruction"] = s["cur"]
        actions, info = self.inner.act(req)
        return actions, {**info, "subtask": s["cur"]}


def load_policy(checkpoint: str, **_kwargs) -> Planned:
    return Planned(checkpoint)
