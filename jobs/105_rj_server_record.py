"""RoboJEPA's RoboCasa env server (scripts/deployment/robots/robocasa/robot_env_server.py) with full-motion recording.

The authors' episode videos hold one frame per planning step (about 1.5 s for a 5-step episode). Between planning steps
the blocking wrapper drives the arm with up to 200 simulator sub-steps; this wrapper records the camera observations of
those sub-steps (both views side by side, every REC_EVERY-th sub-step) so the executed motion of RoboJEPA's planned
actions can be watched. Their code is not edited: BlockingRoboCasaEnv.step wraps the inner simulator's step on first
use, and every reset() closes the current recording (one mp4 per reset segment: the goal set-up and the planned episode).
Env: REC_DIR (output dir), REC_EVERY (2), REC_FPS (20). Arguments are passed through to robot_env_server.py.
"""
import atexit
import os
import runpy
import sys

import numpy as np

SERVER = os.path.join(os.environ["REPO"], "scripts/deployment/robots/robocasa/robot_env_server.py")
REC = {"frames": [], "n": 0, "k": 0, "mode": "start", "dir": os.environ.get("REC_DIR", "rec"),
       "every": int(os.environ.get("REC_EVERY", "2")), "fps": int(os.environ.get("REC_FPS", "20"))}
os.makedirs(REC["dir"], exist_ok=True)
CAMS = ("robot0_agentview_left_image", "robot0_eye_in_hand_image")


def flush():
    if len(REC["frames"]) > 4:
        import imageio_ffmpeg
        fr = REC["frames"]
        path = os.path.join(REC["dir"], f"rec_{REC['n']:02d}_{REC['mode']}.mp4")
        w = imageio_ffmpeg.write_frames(path, fr[0].shape[1::-1], fps=REC["fps"], quality=8, macro_block_size=1)
        w.send(None)
        for f in fr:
            w.send(np.ascontiguousarray(f))
        w.close()
        print(f"[rec] wrote {path} ({len(fr)} frames)", flush=True)
        REC["n"] += 1
    REC["frames"] = []


def wrap_inner(env):
    inner = env._env._env
    if getattr(inner, "_rec_wrapped", False):
        return
    orig = inner.step

    def step(a):
        out = orig(a)
        REC["k"] += 1
        if REC["k"] % REC["every"] == 0 and isinstance(out[0], dict):
            ims = [np.asarray(out[0][c]) for c in CAMS if c in out[0]]
            if ims:
                h = min(i.shape[0] for i in ims)
                REC["frames"].append(np.concatenate([i[:h] for i in ims], 1)[::-1].copy())  # robosuite renders upside down
        return out
    inner.step = step
    inner._rec_wrapped = True


sys.path.insert(0, os.path.join(os.environ["REPO"], "scripts/deployment"))
import robots.robocasa.blocking_robot_env as B  # noqa: E402
import robots.robocasa.robot_env as RE  # noqa: E402

_step = B.BlockingRoboCasaEnv.step


def step(self, action):
    wrap_inner(self)
    return _step(self, action)


B.BlockingRoboCasaEnv.step = step


def patch_reset(cls):
    orig = cls.reset

    def reset(self, *args, **kwargs):
        flush()
        REC["mode"] = str(kwargs.get("mode", args[0] if args else "init"))
        return orig(self, *args, **kwargs)
    cls.reset = reset


for cls in (RE.RoboCasaEnv, B.ObjectReachBlockingRoboCasaEnv, B.ObjectPushBlockingRoboCasaEnv):
    if "reset" in cls.__dict__:
        patch_reset(cls)
atexit.register(flush)
sys.argv = [SERVER] + sys.argv[1:]
runpy.run_path(SERVER, run_name="__main__")
