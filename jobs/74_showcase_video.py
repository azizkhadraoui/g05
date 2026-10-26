"""Single-mission showcase video: high-level policy + memory + world model working together in one episode.
Input: results/showcase/run_<k>.npz from 73_showcase.py (first successful run unless RUN=<k>). Output: results/showcase/showcase.mp4
"""
import glob, importlib, json, os, sys
from functools import lru_cache
import imageio.v2 as imageio
import matplotlib
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
G = "/export/home/kaziz/g05"; R = f"{G}/results"; D = os.environ.get("SHOWCASE_DIR", f"{R}/showcase")
W, H, FPS = 1280, 720, 20
FD = os.path.join(matplotlib.get_data_path(), "fonts", "ttf")
BG, INK, MUTED, BLUE, GREEN, RED, ORANGE, PURPLE, LINE = (247, 248, 250), (26, 32, 44), (100, 110, 125), (43, 108, 176), (47, 133, 90), (197, 48, 48), (221, 107, 32), (107, 70, 193), (226, 232, 240)


@lru_cache(maxsize=None)
def F(sz, bold=False):
    return ImageFont.truetype(os.path.join(FD, "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"), sz)


def wrap(d, text, font, width):
    words, lines, cur = text.split(), [], ""
    for w_ in words:
        t = (cur + " " + w_).strip()
        if d.textlength(t, font=font) <= width:
            cur = t
        else:
            lines.append(cur); cur = w_
    return lines + [cur] if cur else lines


def short(s, n):
    return s if len(s) <= n else s[: n - 1] + "…"


runs = sorted(glob.glob(f"{D}/run_*.npz"), key=lambda p: int(p.rsplit("_", 1)[1].split(".")[0]))
pick = os.environ.get("RUN")
if pick is not None:
    path = f"{D}/run_{pick}.npz"
else:
    ok = [p for p in runs if bool(np.load(p, allow_pickle=True)["success"])]
    path = (ok or runs)[0]
z = np.load(path, allow_pickle=True); meta = json.loads(str(z["meta"]))
frames, stage, cmd = z["frames"], z["stage"], [str(x) for x in z["cmd"]]
events = meta["events"]; wmlog = meta["wm"]; mem = meta["mem"]; langs = meta["langs"]
stages = meta["stages"]; mission = meta["mission"]; success = bool(z["success"])
print("using", path, "success", success, "steps", len(frames))
T = importlib.import_module("07_train_hybrid")
m = np.load(f"{R}/feat_cache/meta.npz")
reader = T.FrameReader({"episode_index": torch.from_numpy(m["episode_index"]), "frame_index": torch.from_numpy(m["frame_index"])})
goal_cache = {}


def goal_img(row):
    if row not in goal_cache:
        goal_cache[row] = reader.frames(torch.tensor([row])).permute(0, 2, 3, 1).numpy()[0]
    return goal_cache[row]


writer = imageio.get_writer(f"{D}/showcase.mp4", fps=FPS, codec="libx264", quality=7, macro_block_size=16)
count = [0]


def emit(im, k=1):
    arr = np.asarray(im.convert("RGB"))
    for _ in range(k):
        writer.append_data(arr); count[0] += 1


def card(title, lines, secs, sub=None):
    im = Image.new("RGB", (W, H), (22, 30, 46)); d = ImageDraw.Draw(im)
    d.text((80, 150), title, font=F(42, True), fill=(255, 255, 255))
    if sub:
        d.text((82, 212), sub, font=F(21), fill=(160, 180, 210))
    y = 280
    for line in lines:
        for wl in wrap(d, line, F(23), W - 180):
            d.text((90, y), wl, font=F(23), fill=(226, 232, 240)); y += 34
        y += 10
    emit(im, int(secs * FPS))


done_at = {}
for t_, kind, txt in events:
    if kind == "done":
        done_at[len(done_at)] = t_
drop_t = mem["total"] if mem else None
X0 = 670  # right-hand panel


def plan_panel(d, t):
    d.text((X0, 90), "MISSION PLAN", font=F(15, True), fill=MUTED)
    for i, (k_, what, _drop) in enumerate(stages):
        col = BLUE if k_ == "goal" else PURPLE
        y = 114 + i * 40
        cur = int(stage[min(t, len(stage) - 1)]) == i and t < len(frames)
        finished = i in done_at and t >= done_at[i]
        d.rounded_rectangle([X0, y, W - 30, y + 34], radius=8, fill=(255, 255, 255), outline=(col if cur else LINE), width=3 if cur else 1)
        d.text((X0 + 10, y + 6), "✓" if finished else ("▶" if cur else "·"), font=F(18, True), fill=GREEN if finished else (col if cur else MUTED))
        d.text((X0 + 36, y + 8), "goal picture" if k_ == "goal" else short(what, 48), font=F(15, True), fill=INK)


def base(t, title, col=INK):
    im = Image.new("RGB", (W, H), BG); d = ImageDraw.Draw(im)
    d.rectangle([0, 0, W, 60], fill=(255, 255, 255)); d.line([0, 60, W, 60], fill=LINE, width=2)
    d.text((24, 15), title, font=F(24, True), fill=col)
    d.text((W - 110, 20), f"{t / 20:4.1f} s", font=F(18), fill=MUTED)
    im.paste(Image.fromarray(frames[min(t, len(frames) - 1)]).resize((600, 600)), (32, 90))
    plan_panel(d, t)
    return im, d


def cmd_box(d, text, col):
    d.rounded_rectangle([X0, 250, W - 30, 296], radius=10, fill=(255, 255, 255), outline=col, width=3)
    d.text((X0 + 16, 262), text, font=F(17, True), fill=col)


# ---------------- intro ----------------
card("One mission", [f"“{mission}”"], 4, sub="high-level policy · memory · world model — G0.5 in the LIBERO kitchen")

# ---------------- high-level policy ----------------
for k in range(int(4 * FPS)):
    im, d = base(0, "High-level policy (RoboBrain 2.5)", PURPLE)
    if k > FPS * 1.5:
        d.rounded_rectangle([X0, 250, W - 30, 400], radius=12, fill=(255, 255, 255), outline=PURPLE, width=3)
        for j, (k_, what, _drop) in enumerate(stages):
            d.text((X0 + 16, 264 + 42 * j), f"{j + 1}.  " + ("picture → world model" if k_ == "goal" else what), font=F(17, True),
                   fill=BLUE if k_ == "goal" else PURPLE)
    emit(im)

# ---------------- episode ----------------
wm_i = 0
HOME = "(returning to rest pose)"
for t in range(len(frames)):
    s = int(stage[t]); kind, _what, drop = stages[min(s, len(stages) - 1)]
    while wm_i + 1 < len(wmlog) and wmlog[wm_i + 1]["total"] <= t:
        wm_i += 1
    first = t > 0 and cmd[t] != HOME and (int(stage[t - 1]) != s or cmd[t - 1] == HOME)
    if cmd[t] == HOME:
        im, d = base(t, "Rest pose", MUTED)
        emit(im)
    elif kind == "lang" and drop is not None:
        lost = drop_t is not None and t >= drop_t
        im, d = base(t, "Memory" if lost else "G0.5", PURPLE if lost else GREEN)
        if lost and t < drop_t + 30:
            cmd_box(d, "✕  COMMAND LOST", RED)
        else:
            cmd_box(d, f"“{short(cmd[t], 48)}”", PURPLE if lost else GREEN)
        if lost and mem:
            for j, ft in enumerate(mem["frames"]):
                ft = max(0, min(ft, len(frames) - 1))
                im.paste(Image.fromarray(frames[ft]).resize((104, 104)), (X0 + j * 116, 320))
            if t >= drop_t + 12:
                d.text((X0, 440), "recalled from memory ✓", font=F(17, True), fill=PURPLE)
        k = 1
        if drop_t is not None and t == drop_t:
            k = 30  # pause on the drop
        if drop_t is not None and t == drop_t + 12:
            k = 40  # pause on the recall
        emit(im, k)
    elif kind == "lang":
        im, d = base(t, "G0.5", GREEN)
        cmd_box(d, f"“{short(cmd[t], 48)}”", GREEN)
        emit(im, 20 if first else 1)
    else:
        rp = wmlog[wm_i] if wmlog and wmlog[wm_i]["total"] <= t else None
        im, d = base(t, "World model", BLUE)
        if rp:
            im.paste(Image.fromarray(goal_img(int(rp["goal_row"]))).resize((170, 170)), (X0, 256))
            d.text((X0, 432), "goal +1.6 s", font=F(14, True), fill=BLUE)
            sc = np.array(rp["scores"]); match = (sc.max() - sc) / (sc.max() - sc.min() + 1e-8)
            x1 = X0 + 190
            for j, c in enumerate(langs):
                y = 252 + j * 33; on = j == rp["choice"]
                d.text((x1, y), short(c, 40), font=F(12, on), fill=INK if on else MUTED)
                d.rectangle([x1, y + 15, x1 + (W - 40 - x1) * max(match[j], 0.02), y + 26], fill=BLUE if on else (190, 198, 210))
        emit(im, 20 if first else 1)

# ---------------- end ----------------
im = Image.new("RGB", (W, H), (22, 30, 46)); d = ImageDraw.Draw(im)
im.paste(Image.fromarray(frames[-1]).resize((560, 560)), ((W - 560) // 2, 120))
msg = "Mission complete" if success else "Mission ended"
d.text(((W - d.textlength(msg, font=F(34, True))) // 2, 50), msg, font=F(34, True), fill=(120, 220, 160) if success else (255, 160, 160))
emit(im, 4 * FPS)
writer.close()
print("wrote", f"{D}/showcase.mp4", count[0], "frames", round(count[0] / FPS, 1), "s")
