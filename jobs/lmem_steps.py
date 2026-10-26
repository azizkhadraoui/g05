"""Step structure of the LIBERO-Mem object-identity tasks (suite ids 6-9), shared by 131_lmem_steps.py (labels and step
episodes from the demos) and 51_lmem_memclient.py (oracle instructions online).

  6  swap the 2 bowls on their plates using the empty plate          3 bowl moves
  7  rotate the 3 bowls on their plates from left to right ...       4 bowl moves
  8  cream cheese in the nearest basket, then *that* basket to the centre
  9  cream cheese in the nearest basket, then the *empty* basket to the centre
Bowls and plates are identical, so after the first move the current frame cannot tell which plate is next: that is the
memory the step instructions need. Plates and baskets are named by their left-to-right order (world y ascending, the
order of the rotate task's "from left to right") at the start of the episode.
An event is a completed move, detected from object positions: a bowl lifted (> 4 cm above its resting height for 5 steps)
and set down again (< 2 cm for 3 steps) on a different plate than the one it was lifted from (the nearest plate; the
lift-and-place rule of 54_lmem_relabel.py, robust to carrying a bowl low past another plate and to re-grasps); the cream
cheese resting inside a basket for 3 steps; a basket set down at least 8 cm from where it started. Demos end right after
their last move, so finish() records a move still in progress at the end of a demo. Objects start a few cm in the air and
drop onto the table in the first steps (bowls spawn at 1.08 m), so the reference positions are re-taken after SETTLE steps.
"""
SETTLE = 15
BASKET_DZ = 0.06  # a moved basket ends 1.5-5 cm above its settled height (still being set down when a demo ends)
EXPECTED_EVENTS = {6: 3, 7: 4, 8: 2, 9: 2}
import numpy as np

TASKS = (6, 7, 8, 9)
PLATE_NAMES = {6: ["left", "middle", "right"], 7: ["leftmost", "second-from-left", "second-from-right", "rightmost"]}
BASKET_NAMES = ["left", "right"]
CHEESE = "put the cream cheese in the nearest basket"
DONE = ""


def bowl_move(tid, a, b):
    return f"move the bowl from the {PLATE_NAMES[tid][a]} plate to the {PLATE_NAMES[tid][b]} plate"


def basket_move(k):
    return f"move the {BASKET_NAMES[k]} basket to the center"


CLASSES = ([bowl_move(6, a, b) for a in range(3) for b in range(3) if a != b] + [bowl_move(7, a, b) for a in range(4) for b in range(4) if a != b]
           + [CHEESE, basket_move(0), basket_move(1), DONE])
CLASS_ID = {c: i for i, c in enumerate(CLASSES)}


def objects(tid):
    if tid in (6, 7):
        nb, npl = (2, 3) if tid == 6 else (3, 4)
        return [f"akita_black_bowl_{i + 1}" for i in range(nb)] + [f"plate_{i + 1}" for i in range(npl)]
    return ["cream_cheese_1", "basket_1", "basket_2"]


class StepTracker:
    """Feed object positions every simulation step (dict name -> xyz); keeps the completed events and names."""

    def __init__(self, tid, pos0):
        self.tid, self.t = tid, 0
        self._init(pos0)

    def _init(self, pos0):
        self.events = []
        p0 = {k: np.asarray(v, np.float64) for k, v in pos0.items()}
        self.p0 = p0
        if self.tid in (6, 7):
            self.bowls = sorted(k for k in p0 if "bowl" in k)
            plates = sorted((k for k in p0 if k.startswith("plate")), key=lambda k: p0[k][1])  # left to right
            self.plates = plates
            self.plate_xy = np.stack([p0[k][:2] for k in plates])
            self.rest = {b: self._near(p0[b]) for b in self.bowls}  # plate index each bowl starts on
            self.z0 = {b: p0[b][2] for b in self.bowls}
            self.up = {b: 0 for b in self.bowls}; self.down = {b: 0 for b in self.bowls}; self.lifted = {b: False for b in self.bowls}
            self.empty0 = [j for j in range(len(plates)) if j not in self.rest.values()]
        else:
            self.baskets = sorted((k for k in p0 if k.startswith("basket")), key=lambda k: p0[k][1])
            self.cheese_in, self.cand_in, self.moved, self.still = None, (None, 0), None, {b: 0 for b in self.baskets}
            self.prev = {b: p0[b].copy() for b in self.baskets}; self.prevc = p0["cream_cheese_1"].copy()

    def _near(self, p):
        return int(np.linalg.norm(self.plate_xy - np.asarray(p)[:2], axis=1).argmin())

    def _place(self, b, p, t):
        self.lifted[b] = False
        j = self._near(p)
        if j != self.rest[b]:  # set down on the same plate = a re-grasp, not a move
            self.events.append(("bowl", self.rest[b], j, t)); self.rest[b] = j

    def finish(self):
        """End of a recorded demo: count a move whose final placement began in the last steps."""
        if self.tid in (6, 7):
            for b in self.bowls:
                if self.lifted[b] and self.down[b] >= 1:
                    self._place(b, self.last[b], self.t - self.down[b] + 1)
            self.events.sort(key=lambda e: e[-1])
            return
        if self.cheese_in is None:
            k, n = self.cand_in
            if k is not None and n >= 1:
                self.cheese_in = k; self.events.append(("cheese", k, self.t - n + 1))
        if self.moved is None:
            for i, b in enumerate(self.baskets):
                if np.linalg.norm(self.prev[b][:2] - self.p0[b][:2]) > 0.08 and self.prev[b][2] < self.p0[b][2] + BASKET_DZ:
                    self.moved = i; self.events.append(("basket", i, self.t - self.still[b])); break

    def update(self, pos):
        self.t += 1
        P = {k: np.asarray(v, np.float64) for k, v in pos.items()}
        if self.t <= SETTLE:  # objects still dropping into place: references from the settled scene
            if self.t == SETTLE:
                self._init(P)
            if self.tid in (6, 7):
                self.last = P
            return
        if self.tid in (6, 7):
            self.last = P
            for b in self.bowls:
                z = P[b][2]
                self.up[b] = self.up[b] + 1 if z > self.z0[b] + 0.03 else 0
                if self.up[b] >= 5:
                    self.lifted[b] = True
                self.down[b] = self.down[b] + 1 if z < self.z0[b] + 0.015 else 0
                if self.lifted[b] and self.down[b] >= 3:
                    self._place(b, P[b], self.t - 2)
            return
        ch = P["cream_cheese_1"]; cstill = np.linalg.norm(ch - self.prevc) < 0.002; self.prevc = ch
        if self.cheese_in is None:
            k = next((i for i, b in enumerate(self.baskets) if np.linalg.norm(ch[:2] - P[b][:2]) < 0.07 and ch[2] < P[b][2] + 0.10), None)
            c, n = self.cand_in
            self.cand_in = (k, n + 1) if (k == c and cstill) else (k, 1 if cstill else 0)
            if k is not None and self.cand_in[1] >= 3:
                self.cheese_in = k; self.events.append(("cheese", k, self.t - 2))
        for i, b in enumerate(self.baskets):
            self.still[b] = self.still[b] + 1 if np.linalg.norm(P[b] - self.prev[b]) < 0.001 else 0; self.prev[b] = P[b]
            if (self.moved is None and self.still[b] >= 5 and np.linalg.norm(P[b][:2] - self.p0[b][:2]) > 0.08
                    and P[b][2] < self.p0[b][2] + BASKET_DZ):
                self.moved = i; self.events.append(("basket", i, self.t - 4))

    def instruction(self, ev):
        return bowl_move(self.tid, ev[1], ev[2]) if ev[0] == "bowl" else CHEESE if ev[0] == "cheese" else basket_move(ev[1])

    def oracle_next(self):
        """The next instruction of a canonical plan, given the moves made so far (oracle condition)."""
        ev = [e for e in self.events]
        if self.tid == 6:
            occ0 = sorted(set(range(3)) - set(self.empty0)); E = self.empty0[0]
            if not ev:
                X = min(occ0, key=lambda j: np.linalg.norm(self.plate_xy[j] - self.plate_xy[E]))
                return bowl_move(6, X, E)
            s1, d1 = ev[0][1], ev[0][2]
            if len(ev) == 1:
                Y = next((j for j in occ0 if j != s1), next(j for j in range(3) if j not in (s1, d1)))  # both bowls read as one plate: the free one
                return bowl_move(6, Y, s1)
            if len(ev) == 2:
                return bowl_move(6, d1, ev[1][1])
            return DONE
        if self.tid == 7:  # the benchmark's sequence over plate identities 1-4 (plate_k), named by position
            idx = {k: i for i, k in enumerate(self.plates)}; p = [idx[f"plate_{k}"] for k in range(1, 5)]
            plan = [(p[0], p[3]), (p[1], p[0]), (p[2], p[1]), (p[3], p[2])]
            return bowl_move(7, *plan[len(ev)]) if len(ev) < 4 else DONE
        if self.cheese_in is None:
            return CHEESE
        if self.moved is None:
            return basket_move(self.cheese_in if self.tid == 8 else 1 - self.cheese_in)
        return DONE
