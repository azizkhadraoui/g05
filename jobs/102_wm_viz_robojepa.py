"""Imagination videos: what RoboJEPA predicts for real DROID household clips (towel / cloth folding, washer, laundry).

RoboJEPA predicts V-JEPA 2.1 features, not pixels, and its pixel decoder is not released. Each imagined future frame is
therefore shown as the moment of the same real clip whose features match the prediction best ("looks like t = x s"),
next to the real frame it should match, plus a per-patch error heatmap. If the model imagines the right future, the
matched moment equals the true one; with another episode's actions it should drift.

Panels per prediction step h = 1..9 (0.8 .. 7.2 s after the last observed frame):
  real future | 22M (true actions) | 300M (true actions) | 300M (another episode's actions) | 300M error heatmap
Same data path, model loading and rollout as 100_wm_eval_robojepa.py. Output: results/wm_viz/<category>_<episode>.mp4
"""
import argparse
import importlib.util
import json
import os
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, "/export/home/kaziz/g05/jobs")
_spec = importlib.util.spec_from_file_location("wmeval", "/export/home/kaziz/g05/jobs/100_wm_eval_robojepa.py")
W = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(W)  # chdir + sys.path to the robo_jepa repo, E = their eval module
E = W.E
from app.robo_jepa.datasets.robot_video_dataset import RobotVideoDataset  # noqa: E402
from app.robo_jepa.transforms import make_transforms  # noqa: E402
from app.robo_jepa.world_model import WorldModel  # noqa: E402

DTYPE, DEV = W.DTYPE, "cuda"
OUT = "/export/home/kaziz/g05/results/wm_viz"
MEAN, STD = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
STEP_S = 0.8  # one model step at 1.25 Hz
try:
    FONT = ImageFont.truetype("DejaVuSans.ttf", 15); FONT_B = ImageFont.truetype("DejaVuSans-Bold.ttf", 17)
except OSError:
    FONT = FONT_B = ImageFont.load_default()


def to_img(rgb_ct):  # [C, H, W] normalised -> uint8 HWC
    x = rgb_ct.float().cpu().numpy().transpose(1, 2, 0)
    if x.min() < 0 or x.max() <= 1.5:
        x = (x * STD + MEAN) * (255.0 if x.max() <= 5 else 1.0)
    return np.clip(x, 0, 255).astype(np.uint8)


def load_wm(size, encoder):
    ck = f"{W.CKPT}/wm-{size}/{W.STEPS[size]}.pth.tar"
    tmp = E.load_dir(W.CKPT, size, ck, f"{W.CKPT}/vjepa2_1_vitG_384.pt")
    try:
        wm = WorldModel.load_pretrained(tmp, load_decoder=False, load_encoder=encoder is None, device="cpu")
    finally:
        import shutil
        shutil.rmtree(tmp)
    if encoder is None:
        encoder = wm.encoder.to(DTYPE).to(DEV)
    wm.encoder = encoder
    wm.model.to(DTYPE)
    wm = wm.to(DEV)
    wm.eval(); wm.predictor_window_steps = 1; wm.rollout_stop_gradient = True
    return wm, encoder


def rollout(wm, item, suffix_actions):
    """Predicted features for target frames 2..10 (last observed frame 0), shape [9, V, H, W, C]. The history uses the
    clip's own actions; the future uses suffix_actions (the clip's own, or another episode's), with poses integrated."""
    to = lambda x: x.to(DEV, dtype=DTYPE)  # noqa: E731
    video_f, action, state, extr, dt = item["video_f"], to(item["action"]), to(item["state"]), to(item["extr"]), to(item["dt"])
    fut = to(suffix_actions)[:, 1:1 + W.ROLL]
    views, emb_ids = item["views"], torch.tensor([E.embodiment_id(wm, item["emb"])], device=DEV)
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=DTYPE):
        predicted, _, pred_states, _ = wm.forward(video_f, action, state, extr, view_map=views, dt=dt, embodiment_ids=emb_ids)
        hist = wm.select_history(video_f, action, predicted, states=state, predicted_states=pred_states, camera_extrinsics=extr,
                                 dt=dt, prefix_len=1, suffix_len=1)
        sp = W.integrate(state[:, 1], fut, emb_ids[:, None], list(wm.model.embodiment_list)).to(DTYPE)
        _, st = wm.rollout(hist, actions_suffix=fut, features_suffix=None, states_suffix=sp, dt=dt[:, 2:],
                           action_noise=0.0, suffix_len=1, run_backward=False, rollout_steps=W.ROLL, view_map=views, embodiment_ids=emb_ids)
    return st.features[0, -W.ROLL:].float()


def panel(img, title, sub, color=(255, 255, 255)):
    im = Image.fromarray(img).resize((320, 240)); can = Image.new("RGB", (320, 290), (20, 24, 32)); can.paste(im, (0, 50))
    d = ImageDraw.Draw(can); d.text((8, 6), title, font=FONT_B, fill=color); d.text((8, 28), sub, font=FONT, fill=(200, 205, 215))
    return can


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per_category", type=int, default=2)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    meta = json.load(open(f"{W.DATA}/episodes.json"))
    ds = RobotVideoDataset(
        dataset_name="DROID_objectstore", data_path=None, cameras=E.CAMERAS, view_map=E.VIEW_MAP, embodiment_name=E.EMBODIMENT,
        raw_fps=E.RAW_FPS, fps=E.FPS, pose_dim=E.POSE_DIM, gripper_dim=E.GRIPPER_DIM, joints_dim=E.JOINTS_DIM,
        intermediate_frames_fps=E.INTERMEDIATE_FRAMES_FPS, s3_profile=None, s3_prefix=W.DATA.rstrip("/") + "/", pad_size=E.PAD_SIZE,
        frames_per_clip=E.FRAMES_PER_CLIP, min_length_to_drop=E.MIN_LENGTH_TO_DROP, min_size=1, transform=make_transforms(**E.EVAL_AUGS),
        read_intermediate_frames=False, action_source="state_delta", filter_samples_func=None, filter_samples_func_kwargs=None)
    paths = [os.path.basename(str(p).rstrip("/")) for p in ds.samples["path"]]
    rng = random.Random(a.seed)
    order = list(range(len(ds))); rng.shuffle(order)
    want = {"fold_towel": a.per_category, "fold_cloth": a.per_category, "washer": a.per_category, "laundry_other": 1}
    picked = []
    for idx in order:
        cat = meta.get(paths[idx], {}).get("category")
        if want.get(cat, 0) > 0 and meta[paths[idx]]["T"] >= 200:
            picked.append(idx); want[cat] -= 1
        if not any(want.values()):
            break
    wm22, enc = load_wm("22m", None)
    items = []
    for idx in picked:
        (rgb, _, action, state, extr, pad, views, dt, start, traj, emb, _i) = ds[idx]
        video = rgb.unsqueeze(0).to(DEV, dtype=DTYPE)
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=DTYPE):
            import einops
            b, _, v, t = video.shape[:4]
            h_, w_ = (s // enc.vit.patch_size for s in video.shape[-2:])
            flat = einops.repeat(einops.rearrange(video, "b c v t h w -> (b t v) c h w"), "b c h w -> b c t h w", t=1)
            feats = einops.rearrange(enc(flat), "(b t v) (h w) c -> b t v h w c", b=b, t=t, v=v, h=h_, w=w_)
        video_f = F.layer_norm(feats.float(), (feats.size(-1),)).to(DTYPE)
        if video_f.shape[1] == action.shape[0] + 1:
            video_f = video_f[:, :-1]
        name = os.path.basename(str(traj).rstrip("/"))
        items.append({"name": name, "task": meta[name]["task"], "cat": meta[name]["category"], "rgb": rgb, "video_f": video_f,
                      "action": action.unsqueeze(0), "state": state.unsqueeze(0), "extr": extr.unsqueeze(0), "dt": dt.unsqueeze(0),
                      "views": views.unsqueeze(0).to(DEV), "emb": emb if isinstance(emb, str) else emb[0], "valid": int(pad.sum())})
    print("clips:", [(it["cat"], it["name"], it["task"][:50]) for it in items], flush=True)
    preds = {}
    for it in items:
        preds[(it["name"], "22m")] = rollout(wm22, it, it["action"])
    del wm22; torch.cuda.empty_cache()
    wm300, _ = load_wm("300m", enc)
    for k, it in enumerate(items):
        preds[(it["name"], "300m")] = rollout(wm300, it, it["action"])
        other = items[(k + 1) % len(items)]  # another household episode's actions, poses integrated from them
        preds[(it["name"], "300m_wrong")] = rollout(wm300, it, other["action"])
    summary = []
    for it in items:
        real = it["video_f"][0].float()  # [T, V, H, W, C]
        T = min(real.shape[0], it["valid"])
        frames, rows = [], []
        for h in range(W.ROLL):
            tgt = 2 + h  # target frame index; last observed frame is 0
            if tgt >= T:
                break
            cols, info = [panel(to_img(it["rgb"][:, 0, tgt]), f"Real  t = {tgt * STEP_S:.1f} s", f"{it['cat']}")], {}
            for key, label in (("22m", "RoboJEPA 22M"), ("300m", "RoboJEPA 300M"), ("300m_wrong", "300M, wrong actions")):
                p = preds[(it["name"], key)][h]
                d = (real[:T] - p[None]).abs().mean(dim=(1, 2, 3, 4))  # match against every moment of the clip
                m = int(d.argmin()); info[key] = m
                ok = m == tgt
                cols.append(panel(to_img(it["rgb"][:, 0, m]), label, f"looks like t = {m * STEP_S:.1f} s " + ("(correct)" if ok else f"(true {tgt * STEP_S:.1f})"),
                                  (120, 220, 150) if ok else (255, 190, 120)))
            err = (real[tgt, 0] - preds[(it["name"], "300m")][h][0]).abs().mean(-1).cpu().numpy()  # [H, W] patches, view 0
            err = (err - err.min()) / (np.ptp(err) + 1e-6)
            heat = np.array(Image.fromarray((err * 255).astype(np.uint8)).resize((320, 240), Image.BILINEAR)).astype(np.float32) / 255
            base = np.array(Image.fromarray(to_img(it["rgb"][:, 0, tgt])).resize((320, 240))).astype(np.float32)
            over = np.clip(base * 0.55 + np.stack([heat * 255, heat * 60, heat * 20], -1) * 0.8, 0, 255).astype(np.uint8)
            cols.append(panel(over, "300M error map", "bright = where prediction is off"))
            row = Image.new("RGB", (320 * len(cols), 290 + 40), (12, 14, 20))
            for c, im in enumerate(cols):
                row.paste(im, (320 * c, 0))
            ImageDraw.Draw(row).text((10, 296), f"“{it['task'][:110]}”   predicting {(h + 1) * STEP_S + STEP_S:.1f} s after the last observed frame",
                                     font=FONT, fill=(230, 232, 240))
            frames += [np.asarray(row)] * 6  # 0.8 s of the clip per step, shown for 1.2 s at 5 fps
            rows.append({"target": tgt, **info})
        path = f"{OUT}/{it['cat']}_{it['name']}.mp4"
        import imageio_ffmpeg
        w = imageio_ffmpeg.write_frames(path, frames[0].shape[1::-1], fps=5, quality=8, macro_block_size=1); w.send(None)
        for f in frames:
            w.send(np.ascontiguousarray(f))
        w.close()
        acc = {k: float(np.mean([r[k] == r["target"] for r in rows])) for k in ("22m", "300m", "300m_wrong")}
        summary.append({"clip": it["name"], "category": it["cat"], "task": it["task"], "video": path, "matched_moment_correct": acc, "steps": rows})
        print(path, acc, flush=True)
    json.dump(summary, open(f"{OUT}/summary.json", "w"), indent=1)


if __name__ == "__main__":
    main()
