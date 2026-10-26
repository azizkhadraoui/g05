"""Activity-log counter for LIBERO-Mem's repetition tasks ("... back on the plate N times"): counts completed
pick-and-place cycles from the robot's own sensors only (finger opening and hand height), the way a real robot logs its
own actions. No camera, no simulator object state. A cycle = the fingers close (grasp), the hand rises while they stay
closed (lift), then the fingers open again (release). Hysteresis on the finger opening, relative to the widest opening
seen so far, so grasps of different objects (bowl rim, bottle neck) all register.
Shared by 138_proprio_check.py (offline check against the demos) and 51_lmem_memclient.py (--halt_mode proprio).
"""
import numpy as np


class ProprioCounter:
    def __init__(self, close_frac=0.6, open_frac=0.8, lift=0.04):
        self.close_frac, self.open_frac, self.lift = close_frac, open_frac, lift
        self.open_w, self.closed, self.z0, self.lifted, self.n = None, False, None, False, 0

    def update(self, gripper_qpos, eef_z):
        w = float(np.sum(np.abs(np.asarray(gripper_qpos, np.float64).reshape(-1)[:2])))
        self.open_w = w if self.open_w is None else max(self.open_w, w)
        if not self.closed and w < self.close_frac * self.open_w:  # grasp
            self.closed, self.z0, self.lifted = True, float(eef_z), False
        elif self.closed:
            if float(eef_z) > self.z0 + self.lift:
                self.lifted = True
            if w > self.open_frac * self.open_w:  # release
                self.closed = False
                if self.lifted:
                    self.n += 1
        return self.n
