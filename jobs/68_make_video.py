"""Team explainer video: high-level policy, steerability, memory and world model on G0.5 (real LIBERO rollouts).

Chapters (all frames are real closed-loop rollouts or real model outputs; captions state the measured results):
  0 intro · 1 the challenge · 2 world-model imagination · 3 planning in the world model · 4 high-level policy (RoboBrain)
  5 steerability · 6 memory A (instruction withdrawn) · 7 memory B (count and stop) · 8 results
Output: results/video/g05_explainer.mp4 (1280x720, 20 fps, H.264)
"""
import glob, importlib, json, os, re, sys
import imageio.v2 as imageio
import matplotlib
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
G = "/export/home/kaziz/g05"; R = f"{G}/results"; OUT = f"{R}/video/g05_explainer.mp4"
W, H, FPS = 1280, 720, 20
FD = os.path.join(matplotlib.get_data_path(), "fonts", "ttf")
from functools import lru_cache


@lru_cache(maxsize=None)
def F(sz, bold=False):
    return ImageFont.truetype(os.path.join(FD, "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"), sz)
BG, INK, MUTED, BLUE, GREEN, RED, ORANGE, PURPLE = (247, 248, 250), (26, 32, 44), (100, 110, 125), (43, 108, 176), (47, 133, 90), (197, 48, 48), (221, 107, 32), (107, 70, 193)
writer = imageio.get_writer(OUT, fps=FPS, codec="libx264", quality=7, macro_block_size=16)
n_frames = [0]


def emit(img, k=1):
    arr = np.asarray(img.convert("RGB"))
    for _ in range(k):
        writer.append_data(arr); n_frames[0] += 1


def wrap(d, text, font, width):
    words, lines, cur = text.split(), [], ""
    for w_ in words:
        t = (cur + " " + w_).strip()
        if d.textlength(t, font=font) <= width:
            cur = t
        else:
            lines.append(cur); cur = w_
    return lines + [cur] if cur else lines


def base(chapter, title, caption):
    im = Image.new("RGB", (W, H), BG); d = ImageDraw.Draw(im)
    d.rectangle([0, 0, W, 64], fill=(255, 255, 255)); d.line([0, 64, W, 64], fill=(226, 232, 240), width=2)
    d.text((24, 14), chapter, font=F(20, True), fill=BLUE); d.text((24 + d.textlength(chapter, font=F(20, True)) + 14, 12), title, font=F(26, True), fill=INK)
    y = H - 92; d.rectangle([0, y, W, H], fill=(255, 255, 255)); d.line([0, y, W, y], fill=(226, 232, 240), width=2)
    for j, line in enumerate(wrap(d, caption, F(19), W - 48)[:3]):
        d.text((24, y + 10 + 25 * j), line, font=F(19), fill=INK)
    return im, d


def card(title, lines, secs=4.0, sub=None):
    im = Image.new("RGB", (W, H), (22, 30, 46)); d = ImageDraw.Draw(im)
    d.text((80, 170), title, font=F(44, True), fill=(255, 255, 255))
    if sub:
        d.text((82, 236), sub, font=F(22), fill=(160, 180, 210))
    y = 300
    for line in lines:
        for wl in wrap(d, line, F(24), W - 180):
            d.text((90, y), wl, font=F(24), fill=(226, 232, 240)); y += 36
        y += 8
    emit(im, int(secs * FPS))


def paste(im, frame, x, y, size, label=None, color=INK, border=None):
    fr = Image.fromarray(np.asarray(frame, np.uint8)).resize((size, size), Image.BILINEAR)
    im.paste(fr, (x, y))
    d = ImageDraw.Draw(im)
    if border:
        d.rectangle([x - 3, y - 3, x + size + 2, y + size + 2], outline=border, width=4)
    if label:
        d.text((x, y - 26), label, font=F(18, True), fill=color)


def pca_img(p, size):
    return Image.fromarray((np.asarray(p) * 255).astype(np.uint8)).resize((size, size), Image.NEAREST)


def short(cmd, n=34):
    return cmd if len(cmd) <= n else cmd[: n - 1] + "…"


def mp4_frames(path):
    fr = imageio.mimread(path, memtest=False)
    return [x[:, : x.shape[1] // 2] if x.shape[1] >= 2 * x.shape[0] else x for x in fr]


def pick(pattern, want_success, prefer=0):
    fs = sorted(glob.glob(pattern, recursive=True))
    ok = [f for f in fs if ("success" in os.path.basename(f)) == want_success]
    return (ok or fs)[min(prefer, len(ok or fs) - 1)] if (ok or fs) else None


def timeline(d, x, y, w, total, t, segs):
    """segs: list of (start_s, end_s, label, color); t: current time (s)."""
    for j, (s0, s1, lab, col) in enumerate(segs):
        x0 = x + w * s0 / total; x1 = x + w * min(s1, total) / total
        d.rectangle([x0, y + 14, x1, y + 26], fill=col)
        when = f"0-{s1:g} s" if s0 == 0 else f"from {s0:g} s"
        d.text((x + (j * (w // 2 + 8)), y - 8), f"{when}: {short(lab, 24)}", font=F(13, True), fill=col)
    cx = x + w * min(t, total) / total; d.line([cx, y + 8, cx, y + 32], fill=INK, width=3)


# ======================= 0. intro =======================
card("What helps a VLA in closed loop?", [
    "Released Galaxea G0.5 (2.9B, flow matching) in the LIBERO simulator, with four additions:",
    "• a small latent world model (43M) that imagines the future and plans over G0.5's proposals",
    "• a high-level policy (RoboBrain 2.5, an embodied VLM) that picks what to do next",
    "• steerability: can G0.5 change its mind mid-task?",
    "• memory: G0.5 sees only the present. Can a memory module fill the gap?"], secs=6, sub="Team explainer · real rollouts, measured results")

# ======================= 1. the challenge =======================
def load_log(path):
    z = np.load(path, allow_pickle=True)
    return z["frames"], z["steps"], bool(z["success"]), [str(x) for x in z["langs"]], json.loads(str(z["replans"])), int(z["task"])


orc = sorted(glob.glob(f"{R}/video/planner/oracle/logs/*.npz")); blk = sorted(glob.glob(f"{R}/video/planner/blank/logs/*.npz"))
if orc and blk:
    fa, _, sa, langs, _, tid = load_log(orc[0]); fb, _, sb, _, _, _ = load_log(blk[0])
    card("1 · The challenge", [f"Task: \"{langs[tid]}\" — two steps, one shared kitchen scene with 10 possible tasks.",
                               "With its instruction G0.5 is near-perfect (98%). Without it, it has no idea what to do (7%).",
                               "Everything below asks: what has to be added so the robot still succeeds when the instruction is replaced by a goal, changes, or disappears?"], secs=6)
    n = max(len(fa), len(fb))
    for k in range(0, n, 2):
        im, d = base("1 · THE CHALLENGE", f"\"{langs[tid]}\"", "Left: G0.5 with the instruction. Right: the same start, no instruction. G0.5 is a strong executor but a pure reflex: it needs to be told exactly what to do, at every moment.")
        paste(im, fa[min(k, len(fa) - 1)], 90, 110, 480, "WITH instruction", GREEN, GREEN if (k >= len(fa) - 2 and sa) else None)
        paste(im, fb[min(k, len(fb) - 1)], 710, 110, 480, "NO instruction", RED, RED if k >= len(fb) - 2 else None)
        emit(im)

# ======================= 2. world-model imagination =======================
z = np.load(f"{R}/video/wm_imagination.npz", allow_pickle=True)
card("2 · The world model", ["A 43M transformer trained in a few hours on LIBERO. It works in the feature space of a frozen DINOv3 encoder (8×8 tokens), not pixels.",
                             "Given the current features and G0.5's next actions, it predicts the features 0.4 s ahead; rolled out 4 times, 1.6 s ahead.",
                             "Below: a held-out demo. Colours are the features (PCA). Compare 'true future' with 'imagined future'."], secs=7)
ratio = float(z["err_pred"].mean() / z["err_persist"].mean())
for k in range(len(z["t"])):
    im, d = base("2 · WORLD MODEL", "Imagining 1.6 s ahead from the planned actions",
                 f"On held-out episodes the imagined future has {ratio:.2f}× the error of assuming nothing moves (0.27× on average at 1.6 s), it depends on the actions (shuffled actions: 2.4× worse) and beats LaWAM's 538M model in-domain.")
    paste(im, z["frame_now"][k], 40, 120, 300, "camera now", INK); paste(im, z["frame_future"][k], 360, 120, 300, "camera in 1.6 s (truth)", INK)
    im.paste(pca_img(z["pca_true"][k], 260), (700, 140)); im.paste(pca_img(z["pca_pred"][k], 260), (990, 140))
    d.text((700, 114), "true future features", font=F(18, True), fill=GREEN); d.text((990, 114), "IMAGINED future features", font=F(18, True), fill=BLUE)
    e, p = float(z["err_pred"][k]), float(z["err_persist"][k])
    d.text((700, 430), f"error imagined: {e:.2f}", font=F(18), fill=BLUE); d.text((700, 456), f"error 'nothing moves': {p:.2f}", font=F(18), fill=MUTED)
    d.rectangle([700, 490, 700 + 520 * min(e / max(p, 1e-6), 1), 506], fill=BLUE); d.rectangle([700, 512, 1220, 528], fill=(203, 213, 224))
    d.text((40, 440), f"t = {z['t'][k] / 20:.1f} s · task: {z['task']}", font=F(18), fill=MUTED)
    emit(im, 3)


# ======================= 3. planning in the world model =======================
def goal_demo_frames(langs):
    """Display subgoals: frames of the first demo of each task (feature-cache rows), as the planner's goal video."""
    import pandas as pd
    C = f"{R}/feat_cache"; DS = f"{G}/data/libero_merged_g05grip"
    m = np.load(f"{C}/meta.npz")
    dd = pd.concat([pd.read_parquet(f, columns=["index", "task_index"]) for f in sorted(glob.glob(f"{DS}/data/*/*.parquet"))]).sort_values("index")
    tidx = dd["task_index"].values
    tp = pd.read_parquet(f"{DS}/meta/tasks.parquet").reset_index(); tcol = [c for c in tp.columns if tp[c].dtype == object][0]
    nrm = lambda s: " ".join(str(s).lower().replace("_", " ").split())  # noqa: E731
    name2idx = {nrm(r[tcol]): int(r["task_index"]) for _, r in tp.iterrows()}
    T = importlib.import_module("07_train_hybrid")
    reader = T.FrameReader({"episode_index": torch.from_numpy(m["episode_index"]), "frame_index": torch.from_numpy(m["frame_index"])})
    out = {}
    for lang in langs:
        rows = np.nonzero((tidx == name2idx[nrm(lang)]) & (m["frame_index"] == 0))[0]
        s0 = int(rows[0]); Ln = int(m["ep_length"][s0])
        out[lang] = (reader, s0, Ln)
    return out


def demo_frame(gd, lang, f):
    reader, s0, Ln = gd[lang]
    return reader.frames(torch.tensor([s0 + min(f, Ln - 1)])).permute(0, 2, 3, 1).numpy()[0]


wmlogs = sorted(glob.glob(f"{R}/video/planner/wm/logs/*.npz"))
wm_ok = sorted([p for p in wmlogs if bool(np.load(p, allow_pickle=True)["success"])], key=lambda p: -len(np.load(p, allow_pickle=True)["steps"]))
if wm_ok:
    fr_, st_, suc, langs, reps, tid = load_log(wm_ok[0])
    gd = goal_demo_frames(langs)
    card("3 · Planning in the world model", ["No instruction is given, only a goal video (demonstrations of the target task).",
                                             "Every 0.5 s: G0.5 proposes one action chunk for each of the 10 kitchen commands → the world model imagines each outcome 1.6 s ahead → the chunk whose imagined future is closest to the goal video is executed.",
                                             "Nothing is retrained: the world model only chooses among G0.5's own proposals."], secs=7)
    rp_i = 0
    for k in range(0, len(fr_)):  # real time: the bar chart updates every 0.5 s
        step = int(st_[k])
        while rp_i + 1 < len(reps) and reps[rp_i + 1]["step"] <= step:
            rp_i += 1
        rp = reps[rp_i]
        im, d = base("3 · PLANNING IN THE WORLD MODEL", "Goal video only, no instruction",
                     "100 paired episodes: 77% success vs 2% (random command) and 7% (no instruction); the correct instruction gives 98%. The chosen command can differ from the target: the world model picks whichever chunk's motion best matches the goal video right now.")
        paste(im, fr_[k], 40, 110, 480, "robot (G0.5 executing the chosen chunk)", INK, GREEN if (k >= len(fr_) - 3 and suc) else None)
        paste(im, demo_frame(gd, langs[tid], step + 32), 560, 110, 200, "goal video, 1.6 s ahead", PURPLE)
        d.text((560, 330), "target:", font=F(16, True), fill=MUTED)
        for j, line in enumerate(wrap(d, langs[tid], F(16), 200)[:3]):
            d.text((560, 352 + 20 * j), line, font=F(16), fill=PURPLE)
        sc = rp.get("scores")
        d.text((790, 84), "world model: imagined match with goal", font=F(17, True), fill=INK)
        if sc:
            sc = np.array(sc); match = (sc.max() - sc) / (sc.max() - sc.min() + 1e-8)
            for j, c in enumerate(langs):
                y = 112 + j * 49; col = BLUE if j == rp["choice"] else (190, 198, 210)
                d.text((790, y), f"{j + 1}. {short(c, 40)}", font=F(14), fill=INK if j == rp["choice"] else MUTED)
                d.rectangle([790, y + 19, 790 + 440 * max(match[j], 0.02), y + 33], fill=col)
            d.text((790, 604), f"→ executing: {short(langs[rp['choice']], 44)}", font=F(16, True), fill=BLUE)
        emit(im)

# ======================= 4. high-level policy: RoboBrain =======================
vlogs = sorted(glob.glob(f"{R}/video/planner/vlm/logs/*.npz"))
v_ok = sorted([p for p in vlogs if bool(np.load(p, allow_pickle=True)["success"])], key=lambda p: -len(np.load(p, allow_pickle=True)["steps"]))
if v_ok:
    fr_, st_, suc, langs, reps, tid = load_log(v_ok[0])
    gd = goal_demo_frames(langs); goal_img = demo_frame(gd, langs[tid], 10 ** 6)
    card("4 · High-level policy: RoboBrain", ["RoboBrain 2.5 (4B) is an embodied vision-language model: it outputs text, points and plans, not actions.",
                                              "Here it is the high-level policy: every 2 s it sees the current scene and a goal image and answers which command G0.5 should run.",
                                              "Zero-shot, no training on our scenes."], secs=6)
    rp_i = 0; said = None
    for k in range(0, len(fr_)):
        step = int(st_[k])
        while rp_i + 1 < len(reps) and reps[rp_i + 1]["step"] <= step:
            rp_i += 1
            if reps[rp_i].get("vlm"):
                said = reps[rp_i]["vlm"]
        if reps[rp_i].get("vlm") and said is None:
            said = reps[rp_i]["vlm"]
        ch = reps[rp_i]["choice"]
        im, d = base("4 · HIGH-LEVEL POLICY (RoboBrain 2.5)", "Scene + goal image → which command?",
                     "100 paired episodes: RoboBrain picking commands from a single goal image reaches 42%, better than our world model with the same image (24%, p = 0.005) but below the world model with a goal video (77%). As memory or counter it does not help: it barely perceives these simulated scenes.")
        paste(im, fr_[k], 40, 110, 480, "robot (G0.5 executing)", INK, GREEN if (k >= len(fr_) - 2 and suc) else None)
        paste(im, goal_img, 560, 110, 240, "goal image", PURPLE)
        d.rounded_rectangle([840, 110, 1240, 330], radius=14, fill=(255, 255, 255), outline=PURPLE, width=3)
        d.text((860, 124), "RoboBrain answers:", font=F(18, True), fill=PURPLE)
        d.text((860, 156), f"\"{(said or '…').strip()[:12]}\"", font=F(30, True), fill=INK)
        d.text((860, 206), "→ G0.5 command:", font=F(16, True), fill=MUTED)
        for j, line in enumerate(wrap(d, langs[ch], F(18), 360)[:3]):
            d.text((860, 230 + 24 * j), line, font=F(18), fill=BLUE)
        d.text((560, 380), "target task (hidden from the robot):", font=F(15), fill=MUTED)
        d.text((560, 402), short(langs[tid], 60), font=F(16, True), fill=INK)
        emit(im)

# ======================= 5. steerability =======================
from libero.libero import benchmark  # noqa: E402
gl = [t.language for t in benchmark.get_benchmark_dict()["libero_goal"]().tasks]
sf = pick(f"{R}/demos/steer/switch_t4/**/*.mp4", False); ss = pick(f"{R}/demos/steer/switch_t1/**/*.mp4", True)
if sf and ss:
    card("5 · Steerability", ["The command changes after 2 s. Does G0.5 drop the old task and follow the new one?",
                              "Released G0.5 switches only 29% of the time (0/10 for 7 of 10 command pairs): once it starts a task, it finishes it."], secs=6)
    A, B = mp4_frames(sf), mp4_frames(ss); n = max(len(A), len(B))
    for k in range(0, n, 2):
        im, d = base("5 · STEERABILITY", "The command switches at 2 s",
                     "Fine-tuning on 145 recorded successful switches improves switching vs a matched control (20% → 28%, p = 0.008), but only back to G0.5's own level. A planner can't fix it either (15%): after a switch, all of G0.5's proposals still go to the old object. Steerability must be trained into the policy.")
        t = k / 20
        for x, frames, tt, ok in ((90, A, 4, False), (710, B, 1, True)):
            paste(im, frames[min(k, len(frames) - 1)], x, 138, 440, None, INK, (GREEN if ok else RED) if k >= len(frames) - 2 else None)
            s_ = (tt + 3) % 10
            timeline(d, x, 96, 440, max(len(frames) / 20, 4), t, [(0, 2, short(gl[s_], 30), ORANGE), (2, 99, short(gl[tt], 34), BLUE)])
        d.text((90, 588), "keeps doing the OLD task", font=F(18, True), fill=RED); d.text((710, 588), "switches to the NEW task", font=F(18, True), fill=GREEN)
        emit(im)

# ======================= 6. memory A =======================
nm = pick(f"{R}/demos/mem2/no_memory/**/*.mp4", False); mm = pick(f"{R}/demos/mem2/memory/**/*.mp4", True)
if nm and mm:
    js = glob.glob(f"{R}/demos/mem2/memory/*_parallel_results.json")
    inferred = list(json.load(open(js[0]))["tasks"][0].get("trial_inferred", {}).values())[0] if js else gl[0]
    card("6 · Memory: the instruction disappears", ["After 2 s the instruction is removed (an operator walks away, a message is lost).",
                                                     "G0.5 sees only the present, so it forgets what it was doing: 98% → 40%.",
                                                     "A memory module keeps the last frames, recognises the task, and re-issues the instruction."], secs=6)
    A, B = mp4_frames(nm), mp4_frames(mm); n = max(len(A), len(B))
    for k in range(0, n, 2):
        im, d = base("6 · MEMORY A", "Instruction removed at 2 s",
                     "100 paired episodes: 40% without memory → 86% with the memory module (p ≈ 1e-10). In this scene even the current frame often reveals the task; part B shows a case where only history works.")
        t = k / 20
        for x, frames, ok, lab in ((70, A, False, "no memory"), (690, B, True, "with memory module")):
            paste(im, frames[min(k, len(frames) - 1)], x, 138, 400, None, INK, (GREEN if ok else RED) if k >= len(frames) - 2 else None)
            timeline(d, x, 96, 400, max(len(frames) / 20, 4), t, [(0, 2, "instruction", BLUE), (2, 99, "— removed —", MUTED)])
            d.text((x, 548), lab, font=F(18, True), fill=GREEN if ok else RED)
        if k >= 40:
            mem = [B[j] for j in (0, 10, 20, 30, 40) if j < len(B)]
            for j, fm in enumerate(mem):
                im.paste(Image.fromarray(fm).resize((60, 60)), (1100 + (j % 2) * 64, 140 + (j // 2) * 64))
            d.text((1100, 112), "memory", font=F(15, True), fill=PURPLE)
            for j, line in enumerate(wrap(d, f"→ \"{inferred}\"", F(14), 170)[:4]):
                d.text((1100, 340 + 18 * j), line, font=F(14), fill=PURPLE)
        emit(im)

# ======================= 7. memory B =======================
def memb(cond, t, want):
    d_ = f"{R}/video/memb/{cond}_t{t}"
    js = glob.glob(f"{d_}/*_parallel_results.json")
    if not js:
        return None
    fin = json.load(open(js[0]))["tasks"][0].get("trial_final", {})
    for f in sorted(glob.glob(f"{d_}/**/*.mp4", recursive=True)):
        mt = re.search(r"episode(\d+)", os.path.basename(f))
        if mt and mt.group(1) in fin and bool(fin[mt.group(1)]["final_done"]) == want:
            return f, fin[mt.group(1)]
    return None


for t, N, txt in ((2, 3, "lift the bowl and place it back on the plate 3 times"), (0, 1, "pick up the bowl and place it back on the plate (once)")):
    a_, c_ = memb("A", t, False), memb("C", t, True)
    if a_ and c_:
        break
if a_ and c_:
    card("7 · Memory: counting and stopping", [f"LIBERO-Mem task: \"{txt}\". After each cycle the scene looks identical: only history tells how many are done.",
                                               "G0.5 (fine-tuned on these demos) can do the cycles, but it never knows when to stop.",
                                               "The memory module reads 8 frames spread over the whole episode plus the current one, counts completed cycles, and stops the robot at N."], secs=7)
    A, B = mp4_frames(a_[0]), mp4_frames(c_[0]); pc = c_[1].get("pred_counts", []); hs = c_[1].get("halt_step")
    n = max(len(A), len(B))
    for k in range(0, n, 2):
        im, d = base("7 · MEMORY B (LIBERO-Mem)", f"\"{txt}\"",
                     "120 held-out episodes: fine-tuned G0.5 alone 5.8% (it overshoots); + memory counter 19.2% (p = 0.003); same counter seeing only the current frame 4.2%; a perfect counter 41.7%. A zero-shot VLM counter (RoboBrain) gets 2.5%.")
        paste(im, A[min(k, len(A) - 1)], 70, 120, 420, "G0.5 alone: keeps going", RED, RED if k >= len(A) - 2 else None)
        paste(im, B[min(k, len(B) - 1)], 620, 120, 420, "with memory counter", GREEN, GREEN if k >= len(B) - 2 else None)
        step = k + 20; ci = (step - 1) // 4
        cnt = pc[min(ci, len(pc) - 1)] if pc else 0
        halted = hs is not None and step >= hs
        d.rounded_rectangle([1070, 140, 1250, 330], radius=14, fill=(255, 255, 255), outline=PURPLE, width=3)
        d.text((1086, 152), "memory count", font=F(16, True), fill=PURPLE)
        d.text((1100, 186), f"{cnt} / {N}", font=F(46, True), fill=INK)
        if halted:
            d.rounded_rectangle([1086, 262, 1234, 314], radius=10, fill=GREEN); d.text((1112, 272), "STOP", font=F(26, True), fill=(255, 255, 255))
        emit(im)

# ======================= 8. results =======================
im = Image.new("RGB", (W, H), (22, 30, 46)); d = ImageDraw.Draw(im)
d.text((70, 50), "What we measured (closed loop, paired episodes)", font=F(34, True), fill=(255, 255, 255))
rows = [("World model plans over frozen G0.5 (goal video, no instruction)", "2% random · 7% none", "77%"),
        ("High-level VLM (RoboBrain) from a single goal image", "24% world model, same image", "42%"),
        ("Memory module re-issues a withdrawn instruction", "40% no memory", "86%"),
        ("Memory counter on LIBERO-Mem repetition tasks", "5.8% alone · 4.2% no-memory counter", "19.2%"),
        ("Steering: switch-recovery fine-tune", "20% matched control", "28%"),
        ("Feeding LaWAM's predicted future into G0.5", "98.3% G0.5", "90.8% (hurts)")]
y = 130
for a1, a2, a3 in rows:
    d.text((70, y), a1, font=F(21), fill=(226, 232, 240)); d.text((760, y), a2, font=F(19), fill=(160, 180, 210)); d.text((1110, y), a3, font=F(24, True), fill=(120, 220, 160))
    y += 62
for j, line in enumerate(["Takeaways: small world models plan well with near-term subgoals; memory helps as an external module; steerability",
                          "must be trained into the policy; a general VLM chooses goals well but needs grounding to act as memory."]):
    d.text((70, 520 + 30 * j), line, font=F(19), fill=(226, 232, 240))
d.text((70, 640), "Code, write-up and figures: github.com/azizkhadraoui/g05", font=F(18, True), fill=(160, 180, 210))
emit(im, 8 * FPS)
writer.close()
print("wrote", OUT, n_frames[0], "frames", round(n_frames[0] / FPS, 1), "s")
