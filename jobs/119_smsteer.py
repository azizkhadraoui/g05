"""SuperMimic-style steering of frozen G0.5 (Liao et al., "SuperMimic: Scaling Humanoid Whole-Body Control with One
Differentiable World Model", CoRL 2026): learn a differentiable world model from simulator transitions and improve the
policy by backpropagating its imagined return through that model, instead of imitating recorded successes (DSBC).

Setting: the LIBERO-Goal command-switch probe (96_frs_steer.sh). The steered policy is the DSBC noise policy
(109_dsbc_train.py): pooled DINOv3 features of the agent view + proprio state + commanded task -> starting noise z of the
frozen G0.5 flow sampler, applied during the 2 s after the command switch. G0.5's weights never change.

  fit      world model f(s_t, feat_t, a_t) -> s_{t+10}: proprio state after the executed 10-step chunk a_t (G0.5's normalised
           action space, valid dims), fitted on k-step autoregressive windows of real recorded transitions (SuperMimic
           eq. 1; k <= 3 inside the 2 s window), and critic Q(feat_t, s, task) = probability that the episode succeeds,
           fitted on the recorded outcomes (it bootstraps beyond the imagined chunk, SuperMimic eq. 2)
  improve  maximise log Q(feat_t, f(s_t, feat_t, a_t), task) with a_t = G0.5_sampler(z = pi(feat_t, s_t, task) | obs_t):
           analytic gradients through the critic, the world model and G0.5's own 10-step Euler sampler (weights frozen;
           the vision-language prefix runs without gradients, exactly as in inference) back to the noise policy, from
           the DSBC policy, with a trust region ||z - z_dsbc||^2 against exploiting model error
  The improved policy is saved in the DSBC checkpoint format: 94_frs_serve.py's "dsbc" mode runs it (DSBC_CKPT).
  policy   SuperMimic proper: the same objective trains G0.5's own weights (the last --train_layers layers, final norm
           and output projection of the action expert, in fp32) from standard-normal noise, with an anchor
           ||chunk - chunk_G0.5||^2 to the original policy's chunk for the same noise; saved as an action-expert state
           dict that 94_frs_serve.py loads with EXPERT_CKPT (no noise policy at inference)
Imagination horizon: one 10-step chunk. G0.5 conditions on camera images, which a state world model cannot imagine, so
every imagined chunk starts from a real observation and the critic carries the return beyond it.
Usage: python 119_smsteer.py fit --sets prefix:results_dir,... --out models.pt
       python 119_smsteer.py improve --models models.pt --init dsbc/policy.pt --out policy_sm.pt
"""
import argparse
import glob
import importlib.util
import json
import os
import re
import sys
import time
import types

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

G = os.environ.get("G05_ROOT", "/export/home/kaziz/g05")
JOBS = f"{G}/jobs"
REC = f"{G}/results/steer_frs/dsbc_records"
CHUNK = 10  # executed steps per request (server --action_steps)


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, f"{JOBS}/{file}")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


DS = _load("dsbc", "109_dsbc_train.py")  # NoisePolicy, TASKS, FEAT, dino_fn


def task_idx(task):
    t = " ".join(str(task).lower().split())
    return DS.TASKS.index(t) if t in DS.TASKS else len(DS.TASKS)


def parse_sets(s):
    return [x.split(":", 1) for x in s.split(",")]


def load_episodes(sets, rec_dir=REC):
    """Recorded window requests grouped by episode (prefix, target, trial), with the episode outcome."""
    succ, eps = {}, {}
    for prefix, results in sets:
        for f in glob.glob(f"{results}/{prefix}_t*/*_parallel_results.json"):
            t = int(re.search(r"_t(\d+)/", f).group(1)); task = json.load(open(f))["tasks"][0]
            off = (task.get("frs") or {}).get("init_offset", 0)
            for k, v in (task.get("trial_successes") or {}).items():
                succ[(prefix, t, int(k) + off)] = bool(v)
        for f in glob.glob(f"{rec_dir}/{prefix}_t*_tr*_s*.npz"):
            m = re.search(r"_t(\d+)_tr(\d+)_s(\d+)\.npz$", f); key = (prefix, int(m.group(1)), int(m.group(2)))
            if key in succ:
                eps.setdefault(key, []).append((int(m.group(3)), f))
    return [{"key": k, "success": succ[k], "recs": sorted(v)} for k, v in sorted(eps.items())]


def read_record(f):
    d = np.load(f, allow_pickle=True)
    keys = sorted(k for k in d.files if k.startswith("state_"))
    return {"s": np.concatenate([d[k].reshape(-1) for k in keys]).astype(np.float32), "x": d["x"].astype(np.float32),
            "pad": d["pad"], "image": d["image"], "wrist": d["wrist"], "task": str(d["task"]),
            "state": {k[6:]: d[k] for k in keys}, "z": d["z"].astype(np.float32)}


class WorldModel(nn.Module):
    """(s_t, a_t, feat_t) -> normalised s_{t+10} - s_t."""

    def __init__(self, sdim, adim, w=512):
        super().__init__()
        self.f = nn.Sequential(nn.LayerNorm(DS.FEAT), nn.Linear(DS.FEAT, 256), nn.GELU())
        self.net = nn.Sequential(nn.Linear(sdim + adim + 256, w), nn.GELU(), nn.Linear(w, w), nn.GELU(), nn.Linear(w, sdim))

    def forward(self, sn, an, f):
        return self.net(torch.cat([sn, an, self.f(f)], -1))


class Critic(nn.Module):
    """(feat, s, task) -> logit of eventual task success."""

    def __init__(self, sdim, w=512):
        super().__init__()
        self.f = nn.Sequential(nn.LayerNorm(DS.FEAT), nn.Linear(DS.FEAT, 256), nn.GELU())
        self.t = nn.Embedding(len(DS.TASKS) + 1, 64)
        self.net = nn.Sequential(nn.Linear(256 + sdim + 64, w), nn.GELU(), nn.Dropout(0.1), nn.Linear(w, w), nn.GELU(), nn.Linear(w, 1))

    def forward(self, f, sn, t):
        return self.net(torch.cat([self.f(f), sn, self.t(t)], -1)).squeeze(-1)


def auc(y, p):
    y, p = np.asarray(y), np.asarray(p); pos, neg = p[y == 1], p[y == 0]
    return float((pos[:, None] > neg[None]).mean() + 0.5 * (pos[:, None] == neg[None]).mean()) if len(pos) and len(neg) else float("nan")


def fit(a):
    dev = "cuda"
    eps = load_episodes(parse_sets(a.sets))
    feat_fn = DS.dino_fn(dev)
    rows = []
    for e in eps:
        for step, f in e["recs"]:
            r = read_record(f)
            rows.append({"ep": e["key"], "step": step, "y": float(e["success"]), "f": feat_fn(r["image"]), "s": r["s"], "x": r["x"],
                         "pad": r["pad"], "task": task_idx(r["task"])})
    pad = next((r["pad"].astype(bool) for r in rows if r["pad"].size), None)
    keep = np.ones(rows[0]["x"].shape[1], bool) if pad is None else ~pad
    Fm = torch.stack([r["f"] for r in rows]).float()
    Sm = torch.tensor(np.stack([r["s"] for r in rows]))
    Am = torch.tensor(np.stack([r["x"][:CHUNK][:, keep].reshape(-1) for r in rows]))
    Tm = torch.tensor([r["task"] for r in rows]); Ym = torch.tensor([r["y"] for r in rows])
    index = {(r["ep"], r["step"]): i for i, r in enumerate(rows)}
    nxt = torch.tensor([index.get((r["ep"], r["step"] + CHUNK), -1) for r in rows])
    has = nxt >= 0
    s_mean, s_std = Sm.mean(0), Sm.std(0) + 1e-4
    a_mean, a_std = Am.mean(0), Am.std(0) + 1e-4
    dS = Sm[nxt[has]] - Sm[has]
    d_mean, d_std = dS.mean(0), dS.std(0) + 1e-4
    keys = sorted({r["ep"] for r in rows}); rng = np.random.default_rng(0)
    val_eps = {keys[i] for i in rng.permutation(len(keys))[: max(1, len(keys) // 10)]}
    is_val = torch.tensor([r["ep"] in val_eps for r in rows])
    print(f"{len(eps)} episodes ({sum(e['success'] for e in eps)} successful), {len(rows)} records, {int(has.sum())} transitions, "
          f"state {Sm.shape[1]}-D, chunk {Am.shape[1]}-D; val episodes {len(val_eps)}", flush=True)
    N = lambda x, m, s: (x - m) / s  # noqa: E731
    Fd, Sd, Ad, Td, Yd, nd = Fm.to(dev), Sm.to(dev), Am.to(dev), Tm.to(dev), Ym.to(dev), nxt.to(dev)
    st = {k: v.to(dev) for k, v in dict(s_mean=s_mean, s_std=s_std, a_mean=a_mean, a_std=a_std, d_mean=d_mean, d_std=d_std).items()}

    # world model: k-step autoregressive windows (SuperMimic eq. 1), seeded at a logged state, rolled on logged chunks
    wm = WorldModel(Sm.shape[1], Am.shape[1]).to(dev)
    opt = torch.optim.AdamW(wm.parameters(), lr=1e-3, weight_decay=1e-4)
    starts = (has & ~is_val).nonzero().view(-1).to(dev); vstarts = (has & is_val).nonzero().view(-1).to(dev)

    def rollout_loss(idx, k_max=3):
        s_hat, i, loss, n, err = Sd[idx], idx, 0.0, 0, []
        for _k in range(k_max):
            ok = nd[i] >= 0
            if not ok.any():
                break
            i, s_hat = i[ok], s_hat[ok]
            d = wm(N(s_hat, st["s_mean"], st["s_std"]), N(Ad[i], st["a_mean"], st["a_std"]), Fd[i])
            s_hat = s_hat + d * st["d_std"] + st["d_mean"]
            j = nd[i]
            loss = loss + (N(s_hat, 0, st["d_std"]) - N(Sd[j], 0, st["d_std"])).pow(2).mean(); n += 1
            err.append((s_hat - Sd[j]).abs().mean().item())
            i = j
        return loss / max(n, 1), err

    for ep in range(a.epochs):
        wm.train()
        for b in starts[torch.randperm(len(starts), device=dev)].split(128):
            loss, _ = rollout_loss(b)
            opt.zero_grad(); loss.backward(); opt.step()
        if ep % 100 == 0 or ep == a.epochs - 1:
            wm.eval()
            with torch.no_grad():
                _, verr = rollout_loss(vstarts)
                base = (Sd[nd[vstarts]] - Sd[vstarts]).abs().mean().item()
            print(f"wm epoch {ep}: val |s_hat - s| per AR step {[round(x, 4) for x in verr]} (no-motion baseline, 1 step: {base:.4f})", flush=True)

    # critic: (feat_t, s, task) -> success, on (s_t) and (s_{t+10}) of every record (the policy step scores s_{t+10})
    q = Critic(Sm.shape[1]).to(dev)
    qopt = torch.optim.AdamW(q.parameters(), lr=3e-4, weight_decay=1e-3)
    pairs_i = torch.cat([torch.arange(len(rows)), has.nonzero().view(-1)]).to(dev)
    pairs_s = torch.cat([torch.arange(len(rows)), nxt[has]]).to(dev)
    pv = is_val.to(dev)[pairs_i]
    tr, va = (~pv).nonzero().view(-1), pv.nonzero().view(-1)
    best = (float("inf"), -1, None)  # early stopping on val BCE: an overconfident critic is what the policy step exploits
    for ep in range(a.epochs):
        q.train()
        for b in tr[torch.randperm(len(tr), device=dev)].split(128):
            i, j = pairs_i[b], pairs_s[b]
            loss = F.binary_cross_entropy_with_logits(q(Fd[i], N(Sd[j], st["s_mean"], st["s_std"]), Td[i]), Yd[i])
            qopt.zero_grad(); loss.backward(); qopt.step()
        if ep % 5 == 0 or ep == a.epochs - 1:
            q.eval()
            with torch.no_grad():
                i, j = pairs_i[va], pairs_s[va]
                p = torch.sigmoid(q(Fd[i], N(Sd[j], st["s_mean"], st["s_std"]), Td[i]))
                vl = F.binary_cross_entropy(p, Yd[i]).item()
            if a.critic_es and vl < best[0]:
                best = (vl, ep, {k: v.clone() for k, v in q.state_dict().items()})
            if ep % 100 == 0 or ep == a.epochs - 1:
                print(f"critic epoch {ep}: val BCE {vl:.3f} AUC {auc(Yd[i].cpu().numpy(), p.cpu().numpy()):.3f} "
                      f"(base rate {Yd[i].mean().item():.2f})", flush=True)
    if a.critic_es:
        q.load_state_dict(best[2])
        print(f"critic: kept epoch {best[1]} (val BCE {best[0]:.3f})", flush=True)
    torch.save({"wm": wm.state_dict(), "q": q.state_dict(), "sdim": Sm.shape[1], "adim": Am.shape[1], "keep": keep.tolist(),
                "stats": {k: v.cpu() for k, v in st.items()}, "sets": a.sets, "episodes": len(eps), "records": len(rows)}, a.out)
    print("saved", a.out, flush=True)


GRAD = {}


def grad_infer(self, model, attention_mask, pixel_values, past_key_values, action_dim_is_pad=None,
               position_ids_override=None, embodiment_types=None, **kwargs):
    """FMHelper.infer from the noise in GRAD["z"], with autograd on through the 10 Euler steps (G0.5's weights are
    frozen, so the gradient reaches only the noise); the differentiable chunk goes to GRAD["x"]."""
    first = next(iter(pixel_values.values()))
    bsz, device, dtype = first.size(0), first.device, first.dtype
    pad = action_dim_is_pad.bool().unsqueeze(1) if (not self.zero_pad_action_target and action_dim_is_pad is not None) else None
    action_mask, action_pos = model.build_action_mask_and_position_ids(
        attention_mask, action_len=self.horizon_steps, position_ids_prefix=position_ids_override,
        split_index=attention_mask.size(1), dtype=dtype, action_causal=self.action_causal)
    if hasattr(model, "_build_prefix_action_kv"):
        ae_kv = {"kv_cache": model._build_prefix_action_kv(past_key_values, past_key_values.num_items())}
    else:
        ae_kv = {"past_key_values": past_key_values}

    def vel(a, t):
        with torch.autocast(device.type, enabled=False):
            emb = model.action_expert.embed(a.float())
            tc = model.action_expert.encode_time(t)
        h = model.action_expert(inputs_embeds=emb, attention_mask=action_mask, position_ids=action_pos, time_cond=tc,
                                attn_implementation=model.attn_implementation, mixture_name="action", **ae_kv)
        return model.action_expert.decode(h).float()

    n = self.num_inference_steps
    with torch.set_grad_enabled(GRAD.get("grad", True)):
        x = GRAD["z"].to(device=device, dtype=torch.float32)
        if pad is not None:
            x = x.masked_fill(pad, 0.0)
        t = torch.ones(bsz, device=device, dtype=dtype)
        for _ in range(n):
            x = x - (1.0 / n) * vel(x, t)
            t = t - 1.0 / n
            if pad is not None:
                x = x.masked_fill(pad, 0.0)
        GRAD["x"] = x
    out = x.detach().to(dtype)
    return out.clamp(-self.final_action_clip_value, self.final_action_clip_value) if self.final_action_clip_value is not None else out


def improve(a):
    sys.argv = [sys.argv[0], "--seed", "7"]
    F94 = _load("frs_serve", "94_frs_serve.py")  # G0.5 server module: T4/V100 fp16 setup, patched FMHelper, cwd = GalaxeaVLA
    S = F94.S
    cfg = S.load_config_from_run_dir(S.find_run_dir(a.ckpt_path), a.ckpt_path, ["eval_embodiment=libero"])
    if "embodiment_datasets" in cfg.data:
        S.filter_embodiment(cfg, "libero")
    policy, processor = S.setup(cfg, device="cuda")
    policy.requires_grad_(False); policy.eval()
    fm = policy.model.fm_helper; fm.infer = types.MethodType(grad_infer, fm)
    inf = S.PolicyInferencer(policy, processor, device="cuda")
    dev = "cuda"

    M = torch.load(a.models, map_location="cpu", weights_only=False)
    wm = WorldModel(M["sdim"], M["adim"]).to(dev); wm.load_state_dict(M["wm"]); wm.eval().requires_grad_(False)
    q = Critic(M["sdim"]).to(dev); q.load_state_dict(M["q"]); q.eval().requires_grad_(False)
    st = {k: v.to(dev) for k, v in M["stats"].items()}; keep = torch.tensor(M["keep"], device=dev)
    ck = torch.load(a.init, map_location="cpu", weights_only=False)
    pol = DS.NoisePolicy(ck["H"], ck["D"], ck["sdim"]).to(dev); pol.load_state_dict(ck["model"])
    ref = DS.NoisePolicy(ck["H"], ck["D"], ck["sdim"]).to(dev); ref.load_state_dict(ck["model"]); ref.eval().requires_grad_(False)
    p_smean, p_sstd = torch.tensor(ck["s_mean"], device=dev), torch.tensor(ck["s_std"], device=dev)
    zpad = torch.zeros(ck["D"], dtype=torch.bool, device=dev) if ck.get("pad") is None else torch.tensor(ck["pad"], device=dev).bool()
    feat_fn = DS.dino_fn(dev)

    eps = load_episodes(parse_sets(a.sets))
    recs, pad_id = [], None
    for e in eps:
        for step, f in e["recs"]:
            r = read_record(f)
            ro = {"images": {"image": r["image"], "wrist_image": r["wrist"]}, "state": r["state"], "task": r["task"]}
            S._parse_task_and_plan(ro)  # the server's own request path (94_frs_serve._process_batch)
            prep = inf._prepare(S.build_obs_dict(ro, processor))
            pad_id = prep.sub_processor.pad_token_id if pad_id is None else pad_id
            recs.append({"sample": prep.sample, "f": feat_fn(r["image"]).float(),
                         "s": torch.from_numpy(r["s"]), "task": task_idx(r["task"]), "ep": e["key"]})
    keys = sorted({r["ep"] for r in recs}); rng = np.random.default_rng(1)
    val_eps = {keys[i] for i in rng.permutation(len(keys))[: max(1, len(keys) // 10)]}
    tr = [i for i, r in enumerate(recs) if r["ep"] not in val_eps]; va = [i for i, r in enumerate(recs) if r["ep"] in val_eps]
    print(f"{len(recs)} window requests from {len(eps)} episodes (val {len(va)}); G0.5 frozen, noise policy from {a.init}", flush=True)
    N = lambda x, m, s: (x - m) / s  # noqa: E731

    def imagine(idx, net):
        """Noise from `net` -> G0.5 chunk (differentiable) -> world model -> critic logit, for the records idx."""
        f = torch.stack([recs[i]["f"] for i in idx]).to(dev); s = torch.stack([recs[i]["s"] for i in idx]).to(dev)
        t = torch.tensor([recs[i]["task"] for i in idx], device=dev)
        z = net(f, N(s, p_smean, p_sstd), t).float()
        z = z.masked_fill(zpad, 0.0)
        GRAD["z"] = z
        inf.forward_batch(inf._collate([dict(recs[i]["sample"]) for i in idx], padding_input_id=pad_id))
        x = GRAD.pop("x")
        act = x[:, :CHUNK][..., keep].reshape(len(idx), -1)
        s_next = s + wm(N(s, st["s_mean"], st["s_std"]), N(act, st["a_mean"], st["a_std"]), f) * st["d_std"] + st["d_mean"]
        return q(f, N(s_next, st["s_mean"], st["s_std"]), t), z

    def evaluate():
        out = []
        GRAD["grad"] = False
        with torch.no_grad():
            for b in range(0, len(va), a.batch):
                idx = va[b:b + a.batch]
                lp, _ = imagine(idx, pol); lr, _ = imagine(idx, ref)
                out.append((torch.sigmoid(lp).mean().item(), torch.sigmoid(lr).mean().item(), len(idx)))
        GRAD["grad"] = True
        n = sum(o[2] for o in out)
        return sum(o[0] * o[2] for o in out) / n, sum(o[1] * o[2] for o in out) / n

    opt = torch.optim.Adam(pol.parameters(), lr=a.lr)
    p0, r0 = evaluate()
    print(f"start: imagined success, val windows: policy {p0:.3f} (DSBC {r0:.3f})", flush=True)
    t0 = time.time()
    for it in range(1, a.iters + 1):
        idx = list(rng.choice(tr, a.batch, replace=False))
        logit, z = imagine(idx, pol)
        with torch.no_grad():
            z_ref = ref(torch.stack([recs[i]["f"] for i in idx]).to(dev),
                        N(torch.stack([recs[i]["s"] for i in idx]).to(dev), p_smean, p_sstd),
                        torch.tensor([recs[i]["task"] for i in idx], device=dev)).float().masked_fill(zpad, 0.0)
        trust = (z - z_ref)[..., ~zpad].pow(2).mean()
        loss = -F.logsigmoid(logit).mean() + a.beta * trust
        opt.zero_grad(); loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(pol.parameters(), 1.0)
        opt.step()
        if it % 25 == 0 or it == a.iters:
            pv, rv = evaluate()
            print(f"iter {it}: train imagined success {torch.sigmoid(logit).mean().item():.3f}, trust {trust.item():.3f}, |grad| {gn:.3f}; "
                  f"val imagined success {pv:.3f} (DSBC {rv:.3f}) [{time.time() - t0:.0f}s]", flush=True)
    torch.save({**{k: v for k, v in ck.items() if k != "model"}, "model": pol.state_dict(), "smsteer": {
        "models": a.models, "init": a.init, "iters": a.iters, "beta": a.beta, "lr": a.lr}}, a.out)
    print("saved", a.out, flush=True)


def train_policy(a):
    """SuperMimic-style training of the low-level policy itself (see the module docstring, `policy`)."""
    sys.argv = [sys.argv[0], "--seed", "7"]
    F94 = _load("frs_serve", "94_frs_serve.py")
    S = F94.S
    cfg = S.load_config_from_run_dir(S.find_run_dir(a.ckpt_path), a.ckpt_path, ["eval_embodiment=libero"])
    if "embodiment_datasets" in cfg.data:
        S.filter_embodiment(cfg, "libero")
    policy, processor = S.setup(cfg, device="cuda")
    policy.requires_grad_(False); policy.eval()
    fm = policy.model.fm_helper; fm.infer = types.MethodType(grad_infer, fm)
    inf = S.PolicyInferencer(policy, processor, device="cuda")
    dev = "cuda"
    expert = next(m for n, m in policy.named_modules() if n.endswith("action_expert"))
    train_mods = [expert.output_proj, expert.norm] + list(expert.layers[-a.train_layers:])
    for m in train_mods:
        m.float(); m.requires_grad_(True)
    params = [p for m in train_mods for p in m.parameters()]
    print(f"training {sum(p.numel() for p in params) / 1e6:.1f}M of the action expert's "
          f"{sum(p.numel() for p in expert.parameters()) / 1e6:.0f}M parameters (last {a.train_layers} layers, norm, output)", flush=True)

    M = torch.load(a.models, map_location="cpu", weights_only=False)
    wm = WorldModel(M["sdim"], M["adim"]).to(dev); wm.load_state_dict(M["wm"]); wm.eval().requires_grad_(False)
    q = Critic(M["sdim"]).to(dev); q.load_state_dict(M["q"]); q.eval().requires_grad_(False)
    st = {k: v.to(dev) for k, v in M["stats"].items()}; keep = torch.tensor(M["keep"], device=dev)
    ck = torch.load(a.init, map_location="cpu", weights_only=False)  # only for the noise shape and padded dims
    H, D = ck["H"], ck["D"]
    zpad = torch.zeros(D, dtype=torch.bool, device=dev) if ck.get("pad") is None else torch.tensor(ck["pad"], device=dev).bool()
    feat_fn = DS.dino_fn(dev)
    eps = load_episodes(parse_sets(a.sets))
    recs, pad_id = [], None
    gen = torch.Generator().manual_seed(0)
    for e in eps:
        for step, f in e["recs"]:
            r = read_record(f)
            ro = {"images": {"image": r["image"], "wrist_image": r["wrist"]}, "state": r["state"], "task": r["task"]}
            S._parse_task_and_plan(ro)
            prep = inf._prepare(S.build_obs_dict(ro, processor))
            pad_id = prep.sub_processor.pad_token_id if pad_id is None else pad_id
            recs.append({"sample": prep.sample, "f": feat_fn(r["image"]).float(), "s": torch.from_numpy(r["s"]),
                         "task": task_idx(r["task"]), "ep": e["key"], "z0": torch.randn(H, D, generator=gen)})
    keys = sorted({r["ep"] for r in recs}); rng = np.random.default_rng(1)
    val_eps = {keys[i] for i in rng.permutation(len(keys))[: max(1, len(keys) // 10)]}
    tr = [i for i, r in enumerate(recs) if r["ep"] not in val_eps]; va = [i for i, r in enumerate(recs) if r["ep"] in val_eps]
    N = lambda x, m, s: (x - m) / s  # noqa: E731

    def chunk(idx):  # G0.5 (current weights) from each record's fixed standard-normal noise
        GRAD["z"] = torch.stack([recs[i]["z0"] for i in idx]).to(dev).masked_fill(zpad, 0.0)
        inf.forward_batch(inf._collate([dict(recs[i]["sample"]) for i in idx], padding_input_id=pad_id))
        return GRAD.pop("x")

    def score(idx, x):
        f = torch.stack([recs[i]["f"] for i in idx]).to(dev); s = torch.stack([recs[i]["s"] for i in idx]).to(dev)
        t = torch.tensor([recs[i]["task"] for i in idx], device=dev)
        act = x[:, :CHUNK][..., keep].reshape(len(idx), -1)
        s_next = s + wm(N(s, st["s_mean"], st["s_std"]), N(act, st["a_mean"], st["a_std"]), f) * st["d_std"] + st["d_mean"]
        return q(f, N(s_next, st["s_mean"], st["s_std"]), t)

    GRAD["grad"] = False  # the original policy's chunks: anchor targets and the imagined-success baseline
    with torch.no_grad():
        for b in range(0, len(recs), a.batch):
            idx = list(range(b, min(b + a.batch, len(recs))))
            for i, x in zip(idx, chunk(idx)):
                recs[i]["x0"] = x.detach().float().cpu()

    def evaluate():
        GRAD["grad"] = False; out = []
        with torch.no_grad():
            for b in range(0, len(va), a.batch):
                idx = va[b:b + a.batch]
                x0 = torch.stack([recs[i]["x0"] for i in idx]).to(dev)
                out.append((torch.sigmoid(score(idx, chunk(idx))).mean().item(), torch.sigmoid(score(idx, x0)).mean().item(), len(idx)))
        GRAD["grad"] = True
        n = sum(o[2] for o in out)
        return sum(o[0] * o[2] for o in out) / n, sum(o[1] * o[2] for o in out) / n

    opt = torch.optim.Adam(params, lr=a.lr)
    p0, r0 = evaluate()
    print(f"{len(recs)} window requests from {len(eps)} episodes (val {len(va)}). start: imagined success {p0:.3f} (G0.5 {r0:.3f})", flush=True)
    t0 = time.time()
    for it in range(1, a.iters + 1):
        idx = list(rng.choice(tr, a.batch, replace=False))
        x = chunk(idx); logit = score(idx, x)
        x0 = torch.stack([recs[i]["x0"] for i in idx]).to(dev)
        anchor = (x - x0)[..., ~zpad].pow(2).mean()
        loss = -F.logsigmoid(logit).mean() + a.beta * anchor
        opt.zero_grad(); loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if it % 25 == 0 or it == a.iters:
            pv, rv = evaluate()
            print(f"iter {it}: train imagined success {torch.sigmoid(logit).mean().item():.3f}, anchor {anchor.item():.4f}, |grad| {gn:.3f}; "
                  f"val imagined success {pv:.3f} (G0.5 {rv:.3f}) [{time.time() - t0:.0f}s]", flush=True)
    torch.save({k: v.detach().cpu() for k, v in expert.state_dict().items()}, a.out)
    print("saved", a.out, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["fit", "improve", "policy"])
    ap.add_argument("--sets", default=f"frs_ref_rec:{G}/results/steer_frs/frs_screen")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=600)
    ap.add_argument("--critic_es", action="store_true", help="keep the critic epoch with the best val BCE")
    ap.add_argument("--models", default=f"{G}/results/steer_frs/smsteer/models.pt")
    ap.add_argument("--init", default=f"{G}/results/steer_frs/dsbc/policy.pt")
    ap.add_argument("--ckpt_path", default="checkpoints/g05-libero/model.pt")
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--beta", type=float, default=0.05)
    ap.add_argument("--train_layers", type=int, default=2, help="policy: trained action-expert layers (from the top)")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    a.out = os.path.abspath(a.out)
    {"fit": fit, "improve": improve, "policy": train_policy}[a.cmd](a)


if __name__ == "__main__":
    main()
