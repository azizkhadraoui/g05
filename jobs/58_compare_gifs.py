"""Side-by-side comparison GIFs for the README (compressed: every 2nd frame, 200 px panels, 64-colour palette)."""
import os
from PIL import Image, ImageSequence

D = "/export/home/kaziz/g05/results/demos"; OUT = f"{D}/readme"; os.makedirs(OUT, exist_ok=True)
PAIRS = {
    "planner_compare": ["planner/blank/gifs/blank_task7_trial0_failure.gif", "planner/wm/gifs/wm_task7_trial0_success.gif"],
    "planner_three_tasks": ["planner/wm/gifs/wm_task1_trial0_success.gif", "planner/wm/gifs/wm_task2_trial0_success.gif", "planner/wm/gifs/wm_task7_trial1_success.gif"],
    "mem2_compare": ["gifs/mem2_no_memory_0_failure.gif", "gifs/mem2_memory_0_success.gif"],
    "libero_mem_compare": ["gifs/lmem_t0_alone_0_failure.gif", "gifs/lmem_t0_memory_0_success.gif"],
    "steer_compare": ["gifs/steer_switch_3_failure.gif", "gifs/steer_switch_0_success.gif"],
}
W = 200
for name, files in PAIRS.items():
    seqs = []
    for f in files:
        fr = [im.convert("RGB") for im in ImageSequence.Iterator(Image.open(f"{D}/{f}"))]
        fr = fr[::2]
        seqs.append([im.resize((W, int(im.height * W / im.width))) for im in fr])
    n = max(len(s) for s in seqs); h = max(s[0].height for s in seqs)
    frames = []
    for k in range(n):
        canvas = Image.new("RGB", (W * len(seqs) + 6 * (len(seqs) - 1), h), (255, 255, 255))
        for j, s in enumerate(seqs):
            canvas.paste(s[min(k, len(s) - 1)], (j * (W + 6), 0))
        frames.append(canvas.quantize(colors=64, method=Image.Quantize.MEDIANCUT))
    frames[0].save(f"{OUT}/{name}.gif", save_all=True, append_images=frames[1:], duration=160, loop=0, optimize=True)
    print(name, len(frames), round(os.path.getsize(f"{OUT}/{name}.gif") / 1e6, 2), "MB")
