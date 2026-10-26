"""Household world-model evaluation, RoboJEPA on DROID towel-folding / laundry episodes (zero-shot).

Data: 99_droid_to_robojepa.py output (1.4k DROID episodes whose instruction is folding a towel/cloth or laundry/washer),
read with RoboJEPA's own RobotVideoDataset exactly as its DROID-holdout eval does (scripts/evals/eval_droid_holdout.py:
15 fps raw, 1.25 Hz model steps = 0.8 s, exterior + wrist views, state-delta actions), fp16 on T4/V100.
Per model size and per category (fold_towel / fold_cloth / washer / laundry_other), with the last observed frame t:
  standard     their imagination error: latent L1 of the autoregressive rollout at steps h = 1..9 (target frame t+1+h),
               fed the dataset actions and the dataset's future poses (the published metric)
  integrated   same, but the pose token of each predicted frame is integrated from the actions (compute_new_pose),
               so the future trajectory enters only through the actions
  shuffled     actions (and integrated poses) taken from another clip of the batch: a world model that ignores
               actions scores the same as 'integrated'; responsiveness = shuffled / integrated
  persistence  'nothing moves' baseline: L1 between frame t and frame t+1+h (same loss function)
  planning     from frame t, 10 candidate 3-step action sequences (the true one + 9 from other clips), each rolled
               out with its own integrated poses; correct if the true sequence's imagined frame t+4 is closest to the
               real frame t+4 (chance 10%)
Output: results/wm_eval/robojepa_<size>.json
"""
import argparse
import importlib.util
import json
import os
import random
import sys
import time
from collections import defaultdict

import einops
import numpy as np
import torch
import torch.nn.functional as F

RJ = "/export/home/kaziz/g05/wm/robo_jepa"
sys.path.insert(0, RJ)
os.chdir(RJ)
_spec = importlib.util.spec_from_file_location("eval_droid_holdout", f"{RJ}/scripts/evals/eval_droid_holdout.py")
E = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(E)
from app.robo_jepa.datasets.robot_video_dataset import RobotVideoDataset  # noqa: E402
from app.robo_jepa.transforms import make_transforms  # noqa: E402
from app.robo_jepa.world_model import WorldModel, compute_new_pose  # noqa: E402

DTYPE = torch.float16  # T4 / V100: no native bf16
E.DTYPE = DTYPE
DATA = "/export/home/kaziz/g05/data/droid_household_robojepa"
CKPT = "/export/home/kaziz/g05/wm/checkpoints/robojepa"
STEPS = {"22m": "cooldown_22m_150k", "300m": "cooldown_300m_300k", "1b": "cooldown_1b_300k", "8b": "cooldown_8b_300k",
         "droid-8b": "world_model"}
ROLL = 9  # standard rollout steps (as published); integrated / shuffled use the same
STRATIFY = False


def encode(encoder, n_clips, seed, device):
    ds = RobotVideoDataset(
        dataset_name="DROID_objectstore", data_path=None, cameras=E.CAMERAS, view_map=E.VIEW_MAP, embodiment_name=E.EMBODIMENT,
        raw_fps=E.RAW_FPS, fps=E.FPS, pose_dim=E.POSE_DIM, gripper_dim=E.GRIPPER_DIM, joints_dim=E.JOINTS_DIM,
        intermediate_frames_fps=E.INTERMEDIATE_FRAMES_FPS, s3_profile=None, s3_prefix=DATA.rstrip("/") + "/", pad_size=E.PAD_SIZE,
        frames_per_clip=E.FRAMES_PER_CLIP, min_length_to_drop=E.MIN_LENGTH_TO_DROP, min_size=1, transform=make_transforms(**E.EVAL_AUGS),
        read_intermediate_frames=False, action_source="state_delta", filter_samples_func=None, filter_samples_func_kwargs=None)
    meta = json.load(open(f"{DATA}/episodes.json"))
    print(f"dataset: {len(ds.samples)} episodes, {len(ds)} items", flush=True)
    E.seed_all(seed)
    order = torch.randperm(len(ds), generator=torch.Generator().manual_seed(seed)).tolist()
    if STRATIFY:  # equal clips per category (towel / cloth / washer are rare among DROID's laundry episodes)
        paths = [os.path.basename(str(p).rstrip("/")) for p in ds.samples["path"]] if hasattr(ds.samples, "__getitem__") else []
        per = defaultdict(list)
        for idx in order:
            cat = meta.get(paths[idx % len(paths)], {}).get("category", "unknown") if paths else "unknown"
            per[cat].append(idx)
        k = n_clips // max(len(per), 1)
        order = [i for c in sorted(per) for i in per[c][:k]]
        print("stratified:", {c: min(len(v), k) for c, v in per.items()}, flush=True)
    else:
        order = order[:n_clips]
    clips, t0 = [], time.time()
    for i, idx in enumerate(order):
        (rgb, _, action, state, extr, pad, views, dt, start, traj, emb, _index) = ds[idx]
        video = rgb.unsqueeze(0).to(device, dtype=DTYPE)
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=DTYPE):
            b, _, v, t = video.shape[:4]
            h_, w_ = (s // encoder.vit.patch_size for s in video.shape[-2:])
            flat = einops.repeat(einops.rearrange(video, "b c v t h w -> (b t v) c h w"), "b c h w -> b c t h w", t=1)
            feats = einops.rearrange(encoder(flat.to(DTYPE)), "(b t v) (h w) c -> b t v h w c", b=b, t=t, v=v, h=h_, w=w_)
        name = os.path.basename(str(traj).rstrip("/"))
        clips.append({"features": feats.to(DTYPE).cpu(), "action": action.unsqueeze(0), "state": state.unsqueeze(0),
                      "extrinsics": extr.unsqueeze(0), "padding_mask": pad.unsqueeze(0), "view_list": views.unsqueeze(0),
                      "dt": dt.unsqueeze(0), "embodiment": [emb] if isinstance(emb, str) else list(emb),
                      "id": f"{name}@{int(start)}", "category": meta.get(name, {}).get("category", "unknown")})
        if (i + 1) % 64 == 0:
            print(f"  encoded {i + 1}/{len(order)} ({time.time() - t0:.0f}s)", flush=True)
    return clips


def integrate(state_now, actions, emb_ids, emb_list):
    """Poses after each action, from state_now [B, S] and actions [B, H, A] -> [B, H, S] (RoboJEPA's own integrator)."""
    S = state_now.shape[-1]
    p, out = state_now[:, None].float(), []
    for h in range(actions.shape[1]):
        new = compute_new_pose(p, actions[:, h:h + 1].float(), emb_ids, emb_list)
        new = F.pad(new, (0, S - new.shape[-1])).to(p.device)
        out.append(new)
        p = new
    return torch.cat(out, 1)


def l1(pred, target):
    return (pred.float() - target.float()).abs().mean(dim=tuple(range(1, pred.dim())))  # per batch element


def evaluate(wm, clips, device, batch_size, starts_every, rng):
    res = defaultdict(lambda: defaultdict(list))  # category -> metric -> list of per-element arrays
    emb_list = list(wm.model.embodiment_list)
    pool = [(c["action"][0], c["state"][0], int(c["padding_mask"].sum())) for c in clips]
    groups = defaultdict(list)
    for c in clips:
        groups[(int(c["padding_mask"].sum()), c["category"])].append(c)
    for (_vl, cat), group in sorted(groups.items()):
        for gi in range(0, len(group), batch_size):
            chunk = group[gi:gi + batch_size]
            B = len(chunk)
            cat_t = lambda k: torch.cat([c[k] for c in chunk])  # noqa: E731
            to = lambda x: x.to(device, dtype=DTYPE)  # noqa: E731
            action, state, extr, dt = to(cat_t("action")), to(cat_t("state")), to(cat_t("extrinsics")), to(cat_t("dt"))
            pmask, views = to(cat_t("padding_mask")), cat_t("view_list").to(device)
            emb_ids = torch.tensor([E.embodiment_id(wm, e) for c in chunk for e in c["embodiment"]], device=device)
            video_f = cat_t("features").to(device)
            if wm.encoder_layernorm:
                video_f = F.layer_norm(video_f.float(), (video_f.size(-1),)).to(DTYPE)
            if video_f.shape[1] == action.shape[1] + 1:
                video_f, pmask = video_f[:, :-1], pmask[:, :-1]
            total = int(pmask.sum(1).min().item())
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=DTYPE):
                predicted, _, pred_states, _ = wm.forward(video_f, action, state, extr, view_map=views, dt=dt, embodiment_ids=emb_ids)
                for t in range(0, total - ROLL - 1, starts_every):  # last target frame t + 1 + ROLL stays valid
                    hist = wm.select_history(video_f, action, predicted, states=state, predicted_states=pred_states,
                                             camera_extrinsics=extr, dt=dt, prefix_len=t + 1, suffix_len=1)
                    common = dict(dt=dt[:, t + 2:], action_noise=0.0, suffix_len=1, run_backward=False, rollout_steps=ROLL,
                                  view_map=views, embodiment_ids=emb_ids)
                    tgt = video_f[:, t + 2:t + 2 + ROLL]
                    acts = action[:, t + 1:t + 1 + ROLL]
                    per = {}
                    # standard (published): dataset poses
                    _, st = wm.rollout(hist, actions_suffix=acts, features_suffix=None, states_suffix=state[:, t + 2:], **common)
                    per["standard"] = torch.stack([l1(st.features[:, -ROLL + h], tgt[:, h]) for h in range(ROLL)], 1)
                    # integrated poses from the true actions
                    ip = integrate(state[:, t + 1], acts, emb_ids[:, None], emb_list).to(DTYPE)
                    _, st = wm.rollout(hist, actions_suffix=acts, features_suffix=None, states_suffix=ip, **common)
                    per["integrated"] = torch.stack([l1(st.features[:, -ROLL + h], tgt[:, h]) for h in range(ROLL)], 1)
                    # shuffled: another clip's actions (from the pool, any start), poses integrated from them
                    sa = []
                    for _ in range(B):
                        a, _s, vl = pool[rng.randrange(len(pool))]
                        s0 = rng.randrange(0, max(1, vl - ROLL - 1))
                        sa.append(a[s0:s0 + ROLL])
                    sa = to(torch.stack(sa))
                    sp = integrate(state[:, t + 1], sa, emb_ids[:, None], emb_list).to(DTYPE)
                    _, st = wm.rollout(hist, actions_suffix=sa, features_suffix=None, states_suffix=sp, **common)
                    per["shuffled"] = torch.stack([l1(st.features[:, -ROLL + h], tgt[:, h]) for h in range(ROLL)], 1)
                    per["persistence"] = torch.stack([l1(video_f[:, t], tgt[:, h]) for h in range(ROLL)], 1)
                    for k, v in per.items():
                        res[cat][k].append(v.float().cpu().numpy())
                    if t == 0:  # planning: 10 candidates x 3 steps, per clip
                        for i in range(B):
                            hi = wm.select_history(video_f[i:i + 1], action[i:i + 1], predicted[i:i + 1], states=state[i:i + 1],
                                                   predicted_states=pred_states[i:i + 1] if pred_states is not None else None,
                                                   camera_extrinsics=extr[i:i + 1], dt=dt[i:i + 1], prefix_len=t + 1, suffix_len=1)
                            hi = hi.repeat(10)
                            cands = [action[i, t + 1:t + 4]]
                            while len(cands) < 10:
                                a, _s, vl = pool[rng.randrange(len(pool))]
                                s0 = rng.randrange(0, max(1, vl - 4))
                                cands.append(to(a[s0:s0 + 3]))
                            ca = torch.stack(cands)
                            cp = integrate(state[i:i + 1, t + 1].repeat(10, 1), ca, emb_ids[i:i + 1, None].repeat(10, 1), emb_list).to(DTYPE)
                            _, st = wm.rollout(hi, actions_suffix=ca, features_suffix=None, states_suffix=cp, dt=dt[i:i + 1, t + 2:].repeat(10, 1),
                                               action_noise=0.0, suffix_len=1, run_backward=False, rollout_steps=3,
                                               view_map=views[i:i + 1].repeat(10, 1) if views.dim() > 1 else views, embodiment_ids=emb_ids[i:i + 1].repeat(10))
                            d = l1(st.features[:, -1], video_f[i:i + 1, t + 4].repeat(10, *[1] * (video_f.dim() - 2)))
                            res[cat]["planning_correct"].append(np.array([[float(int(d.argmin()) == 0)]]))
    out = {}
    for cat, m in res.items():
        out[cat] = {k: np.concatenate(v, 0).mean(0).tolist() for k, v in m.items()}
        out[cat]["n_windows"] = int(np.concatenate(m["standard"], 0).shape[0])
    allm = defaultdict(list)
    for m in res.values():
        for k, v in m.items():
            allm[k] += v
    out["all"] = {k: np.concatenate(v, 0).mean(0).tolist() for k, v in allm.items()}
    out["all"]["n_windows"] = int(np.concatenate(allm["standard"], 0).shape[0])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="22m,300m,1b")
    ap.add_argument("--n_clips", type=int, default=192)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--starts_every", type=int, default=3)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", default="/export/home/kaziz/g05/results/wm_eval")
    ap.add_argument("--stratify", action="store_true", help="equal clips per category")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    global STRATIFY
    STRATIFY = a.stratify
    os.makedirs(a.out, exist_ok=True)
    device = "cuda"
    encoder, clips = None, None
    for size in a.sizes.split(","):
        t0 = time.time()
        tmp = E.load_dir(CKPT, size, f"{CKPT}/wm-{size}/{STEPS[size]}.pth.tar", f"{CKPT}/vjepa2_1_vitG_384.pt")
        try:
            # load on the CPU and cast to fp16 before moving: the 8B predictor is ~32 GB in fp32
            wm = WorldModel.load_pretrained(tmp, load_decoder=False, load_encoder=encoder is None, device="cpu")
        finally:
            import shutil
            shutil.rmtree(tmp)
        if encoder is None:
            encoder = wm.encoder.to(DTYPE).to(device)
            clips = encode(encoder, a.n_clips, a.seed, device)
        else:
            wm.encoder = encoder
        wm.model.to(DTYPE)
        wm = wm.to(device)
        for obj in (wm, getattr(wm, "model", None)):
            if obj is not None and isinstance(getattr(obj, "device", None), (str, torch.device)):
                obj.device = torch.device(device)
        wm.eval(); wm.predictor_window_steps = 1; wm.rollout_stop_gradient = True
        res = evaluate(wm, clips, device, a.batch_size, a.starts_every, random.Random(a.seed))
        al = res["all"]
        summary = {k: round(float(np.mean(al[k])), 4) for k in ("standard", "integrated", "shuffled", "persistence")}
        summary["responsiveness"] = round(float(np.mean(al["shuffled"]) / np.mean(al["integrated"])), 3)
        summary["vs_persistence"] = round(float(np.mean(al["integrated"]) / np.mean(al["persistence"])), 3)
        summary["planning_top1"] = round(float(al["planning_correct"][0]), 3)
        json.dump({"size": size, "checkpoint": STEPS[size], "n_clips": len(clips), "summary": summary, "results": res,
                   "categories": {c: sum(1 for x in clips if x["category"] == c) for c in set(x["category"] for x in clips)},
                   "seconds": time.time() - t0}, open(f"{a.out}/robojepa_{size}{a.tag}.json", "w"), indent=1)
        print(f"robojepa {size}: {summary} ({time.time() - t0:.0f}s)", flush=True)
        del wm
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
