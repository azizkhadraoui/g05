"""Convert demo MP4s (agent view | wrist view) into captioned GIFs for the README (agent view only, 2x subsampled)."""
import glob, os, sys
import imageio.v2 as imageio
from PIL import Image, ImageDraw

D = "/export/home/kaziz/g05/results/demos"; OUT = f"{D}/gifs"; os.makedirs(OUT, exist_ok=True)
SPECS = [  # (glob, caption line 1, caption line 2, output prefix)
    ("mem2/no_memory/**/*.mp4", "Instruction removed at 2 s, no memory", "G0.5 alone: forgets the task", "mem2_no_memory"),
    ("mem2/memory/**/*.mp4", "Instruction removed at 2 s + memory module", "module re-issues the recognised task", "mem2_memory"),
    ("lmem/A_t0/**/*.mp4", "LIBERO-Mem: place the bowl back ONCE", "fine-tuned G0.5 alone (no memory)", "lmem_t0_alone"),
    ("lmem/C_t0/**/*.mp4", "LIBERO-Mem: place the bowl back ONCE", "+ memory module: counts cycles, stops at N", "lmem_t0_memory"),
    ("lmem/A_t2/**/*.mp4", "LIBERO-Mem: place the bowl back 3 TIMES", "fine-tuned G0.5 alone (no memory)", "lmem_t2_alone"),
    ("lmem/C_t2/**/*.mp4", "LIBERO-Mem: place the bowl back 3 TIMES", "+ memory module: counts cycles, stops at N", "lmem_t2_memory"),
    ("steer/switch_t*/**/*.mp4", "Command switched at 2 s", "released G0.5", "steer_switch"),
]
import json, re


def final_outcome(f):
    """LIBERO-Mem runs score success at the last step (trial_final), not at the client's early-stop flag."""
    d = os.path.dirname(f)
    while d.startswith(D) and not glob.glob(f"{d}/*_parallel_results.json"):
        d = os.path.dirname(d)
    js = glob.glob(f"{d}/*_parallel_results.json")
    m = re.search(r"episode(\d+)", os.path.basename(f))
    if js and m:
        fin = json.load(open(js[0]))["tasks"][0].get("trial_final")
        if fin and m.group(1) in fin:
            return "success" if fin[m.group(1)]["final_done"] else "failure"
    return "success" if "success" in os.path.basename(f) else "failure"


for pat, l1, l2, pre in SPECS:
    for k, f in enumerate(sorted(glob.glob(f"{D}/{pat}", recursive=True))[:6]):
        fr = imageio.mimread(f, memtest=False)
        res = final_outcome(f)
        frames = []
        for j, x in enumerate(fr[::2]):
            w = x.shape[1] // 2 if x.shape[1] >= 2 * x.shape[0] else x.shape[1]
            im = Image.fromarray(x[:, :w]).resize((288, 288))
            c = Image.new("RGB", (288, 332), (255, 255, 255)); c.paste(im, (0, 44)); d = ImageDraw.Draw(c)
            d.text((4, 2), l1, fill=(0, 0, 0)); d.text((4, 16), l2, fill=(0, 70, 160))
            if j == len(fr[::2]) - 1:
                d.text((4, 30), res.upper(), fill=(0, 140, 0) if res == "success" else (190, 0, 0))
            frames.append(c)
        frames += [frames[-1]] * 8
        o = f"{OUT}/{pre}_{k}_{res}.gif"
        frames[0].save(o, save_all=True, append_images=frames[1:], duration=100, loop=0, optimize=True)
        print(o, len(frames))
