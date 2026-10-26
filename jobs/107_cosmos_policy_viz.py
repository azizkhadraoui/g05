"""Cosmos 3 Edge-Policy-DROID on real DROID household episodes: generated video + generated actions, chained.

The policy mode of Cosmos 3 (cookbooks/cosmos3/generator/action/run_policy_with_diffusers.ipynb) takes one
conditioning image (wrist view on top, the two exterior views below, 640x540) and the task instruction, and generates
the next 16 actions together with the 17-frame video of executing them (15 fps). Chaining chunks (the last generated
frame conditions the next chunk) gives several seconds of the model's own imagined task execution.

Each output video: left = what the real robot did from the same frame (same composite), right = Cosmos's imagined
rollout. Generated actions (model-normalised DROID joint-position chunks) are saved next to it.
Inputs: data/droid_household clips + episodes.json (99_droid_to_robojepa.py). Output: results/cosmos_viz/
Env: COSMOS_DTYPE (fp16 | bf16 | fp32; T4/V100 have no native bf16), MODEL (checkpoint dir name)
"""
import argparse
import json
import os
import random
import time

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont, ImageOps

G = "/export/home/kaziz/g05"
CK = f"{G}/wm/checkpoints"
CLIPS = f"{G}/data/droid_household/clips"
META = f"{G}/data/droid_household_robojepa/episodes.json"
OUT = os.environ.get("COSMOS_OUT", f"{G}/results/cosmos_viz")
CAMS = {"wrist": "wrist_left", "exterior_1": "exterior_1_left", "exterior_2": "exterior_2_left"}
W, H = 640, 540
try:
    FONT = ImageFont.truetype("DejaVuSans-Bold.ttf", 18); FONT_S = ImageFont.truetype("DejaVuSans.ttf", 15)
except OSError:
    FONT = FONT_S = ImageFont.load_default()


def read_frames(path):
    import imageio.v2 as imageio
    r = imageio.get_reader(path)
    return [Image.fromarray(f) for f in r]


def concat(frames):
    """Wrist on top (full width), exterior 1 / 2 below: the policy's conditioning canvas."""
    can = Image.new("RGB", (W, H)); top = H // 2
    can.paste(ImageOps.fit(frames["wrist"], (W, top), method=Image.Resampling.BICUBIC), (0, 0))
    can.paste(ImageOps.fit(frames["exterior_1"], (W // 2, H - top), method=Image.Resampling.BICUBIC), (0, top))
    can.paste(ImageOps.fit(frames["exterior_2"], (W // 2, H - top), method=Image.Resampling.BICUBIC), (W // 2, top))
    return can


def frames_of(result):
    v = getattr(result, "video", None)
    if v is None:
        v = getattr(result, "frames", None)
    if isinstance(v, (list, tuple)) and v and isinstance(v[0], (list, tuple)):
        v = v[0]
    if isinstance(v, np.ndarray) and v.ndim == 5:
        v = v[0]
    out = []
    for f in v:
        if isinstance(f, Image.Image):
            out.append(f.convert("RGB"))
        else:
            a = np.asarray(f)
            a = (a * 255).clip(0, 255).astype(np.uint8) if a.dtype != np.uint8 else a
            out.append(Image.fromarray(a))
    return out


def label(im, title, sub, color):
    can = Image.new("RGB", (W, H + 56), (16, 18, 24)); can.paste(im.resize((W, H)), (0, 56))
    d = ImageDraw.Draw(can); d.text((10, 6), title, font=FONT, fill=color); d.text((10, 31), sub, font=FONT_S, fill=(210, 214, 222))
    return can


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("MODEL", "Cosmos3-Edge-Policy-DROID"))
    ap.add_argument("--per_category", type=int, default=2)
    ap.add_argument("--chunks", type=int, default=5)
    ap.add_argument("--start", type=int, default=15, help="conditioning frame index (15 fps)")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--fps", type=int, default=5, help="generation frame rate; the model card's canonical policy request uses 5 (17 frames = 3.2 s)")
    ap.add_argument("--guidance", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--example", action="store_true", help="NVIDIA's own DROID example episode (640x360 cameras) instead of ours")
    ap.add_argument("--episodes", default="", help="comma-separated episode names (e.g. ep054795); overrides the category sampling")
    ap.add_argument("--anchor", default="chain", choices=["chain", "real"],
                    help="chain: each chunk starts from the last generated frame (long-horizon imagination); "
                         "real: each chunk starts from the real frame at that moment (one-second predictions, no drift)")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    from diffusers import Cosmos3OmniPipeline, CosmosActionCondition
    from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[os.environ.get("COSMOS_DTYPE", "fp16")]
    t0 = time.time()
    pipe = Cosmos3OmniPipeline.from_pretrained(f"{CK}/{a.model}", torch_dtype=dtype, safety_checker=None, enable_safety_checker=False)
    if os.environ.get("OFFLOAD", "1") == "1":  # 16 GB T4: transformer and VAE take turns on the GPU
        pipe.enable_model_cpu_offload()
    else:
        pipe.to("cuda")
    for m in ("enable_tiling", "enable_slicing"):  # the Wan VAE's 3D convs need ~9 GB at 480p without tiling
        if hasattr(pipe.vae, m):
            getattr(pipe.vae, m)()
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=5.0, use_karras_sigmas=False)
    print(f"loaded {a.model} in {dtype} ({time.time() - t0:.0f}s), cuda {torch.cuda.memory_allocated() / 2**30:.1f} GiB", flush=True)

    meta = json.load(open(META)); rng = random.Random(a.seed)
    names = sorted(meta); rng.shuffle(names)
    want = {"fold_towel": a.per_category, "fold_cloth": a.per_category, "washer": a.per_category}
    picked = []
    for n in names:
        c = meta[n]["category"]
        need = a.start + round(15 / a.fps) * 16 * a.chunks + 1
        if want.get(c, 0) > 0 and meta[n]["T"] >= need:
            picked.append(n); want[c] -= 1
        if not any(want.values()):
            break
    if a.episodes:
        picked = [e for e in a.episodes.split(",") if e in meta]
    if a.example:  # the cookbook's DROID episode, at the resolution NVIDIA conditions on
        ex = f"{G}/wm/cosmos/cookbooks/cosmos3/generator/action/assets/droid_lerobot_example/videos"
        EXCAMS = {"wrist": "observation.image.wrist_image_left", "exterior_1": "observation.image.exterior_image_1_left",
                  "exterior_2": "observation.image.exterior_image_2_left"}
        meta = {"nvidia_example": {"episode_index": -1, "task": "Pick up the object and place it in the target container.",
                                   "category": "nvidia_example", "T": 10 ** 6}}
        picked = ["nvidia_example"]
    summary = []
    for n in picked:
        ep = meta[n]["episode_index"]; task = meta[n]["task"]
        if n == "nvidia_example":
            cams = {k: read_frames(f"{ex}/{v}/chunk-000/file-000.mp4") for k, v in EXCAMS.items()}
        else:
            cams = {k: read_frames(f"{CLIPS}/{ep:06d}/{v}.mp4") for k, v in CAMS.items()}
        T = min(len(f) for f in cams.values())
        stride = round(15 / a.fps)  # source clips are 15 fps; sample the real robot at the generation rate
        real = [concat({k: cams[k][t] for k in CAMS}) for t in range(a.start, min(T, a.start + stride * 16 * a.chunks + 1), stride)]
        cond = real[0]
        gen, actions, times = [cond], [], []
        g = torch.Generator(device="cuda").manual_seed(a.seed)
        for c in range(a.chunks):
            t1 = time.time()
            if a.anchor == "real":
                cond = real[min(16 * c, len(real) - 1)]
            res = pipe(prompt=task, action=CosmosActionCondition(mode="policy", chunk_size=16, domain_name="droid_lerobot",
                                                                 resolution_tier=480, image=cond, view_point="concat_view"),
                       fps=a.fps, num_inference_steps=a.steps, guidance_scale=a.guidance, use_system_prompt=False, generator=g)
            fr = frames_of(res)
            act = getattr(res, "action", None)
            actions.append(np.asarray(act[0].float().cpu() if torch.is_tensor(act[0]) else act[0]).tolist() if act is not None else None)
            gen += fr[1:]; cond = fr[-1]; times.append(time.time() - t1)
            arr = np.asarray(fr[-1], np.float32)
            print(f"{n} chunk {c}: {len(fr)} frames in {times[-1]:.0f}s, last-frame mean {arr.mean():.1f} std {arr.std():.1f}", flush=True)
        L = min(len(gen), len(real))
        out = []
        for i in range(L):
            left = label(real[i], "Real robot", f"t = {i / a.fps:.1f} s", (240, 240, 240))
            sub = "from the first frame + instruction only" if a.anchor == "chain" else f"next {16 / a.fps:.1f} s, from the real frame at {16 * min(i // 16, a.chunks - 1) / a.fps:.1f} s + instruction"
            right = label(gen[i], f"Cosmos 3 Edge imagines (chunk {min(i // 16 + 1, a.chunks)})", sub, (120, 200, 255))
            row = Image.new("RGB", (2 * W, H + 56 + 36), (8, 10, 14)); row.paste(left, (0, 0)); row.paste(right, (W, 0))
            ImageDraw.Draw(row).text((10, H + 62), f"“{task[:120]}”   ({meta[n]['category']}, DROID episode {ep})", font=FONT_S, fill=(230, 232, 240))
            out.append(np.asarray(row))
        path = f"{OUT}/{meta[n]['category']}_{n}{'_anchored' if a.anchor == 'real' else ''}.mp4"
        import imageio_ffmpeg
        w = imageio_ffmpeg.write_frames(path, out[0].shape[1::-1], fps=a.fps, quality=8, macro_block_size=1); w.send(None)
        for f in out:
            w.send(np.ascontiguousarray(f))
        w.close()
        json.dump({"episode": n, "task": task, "actions_normalised": actions, "chunk_seconds": times}, open(path[:-4] + "_actions.json", "w"))
        summary.append({"episode": n, "category": meta[n]["category"], "task": task, "video": path, "frames": L, "sec_per_chunk": float(np.mean(times))})
        print("wrote", path, flush=True)
    json.dump(summary, open(f"{OUT}/summary_{a.anchor}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
