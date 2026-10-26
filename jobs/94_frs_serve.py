"""G0.5 batched policy server with Flow Reversal Steering (FRS) and related reference-action conditions.
No edits to GalaxeaVLA or the other servers. Same CLI as scripts/serve_policy_batched.py (+ --seed S).

Per-request fields a client may add to raw_obs (requests without them are plain G0.5):
  ref_action  float32 [H, 7]   coarse reference chunk in the dataset's raw action units (ee_pose 6 + gripper, 1 = open)
  frs_mode    "frs" | "sde" | "ref"
      frs: integrate the policy's flow backward from the reference (t: 0 -> 1, frs_back Euler steps) to get a noise
           initialisation, then integrate forward as usual (10 steps). Exact integration would return the reference;
           the refinement comes from the finite-step mismatch and from conditioning on the current scene + command.
      sde: partial noising: x = (1 - t0) * ref + t0 * noise, integrate forward from t0 (SDEdit-style baseline)
      ref: return the reference itself (no policy), i.e. "execute the rough action directly"
  frs_back    int    backward steps for frs (default 5)
  frs_ts      float  frs: invert only up to t = ts (< 1 keeps more of the reference; default 1 = all the way to noise),
                     then integrate forward from ts
  sde_t0      float  noise level for sde (default 0.6)
  guide_lam   float  reference guidance during the forward pass: v <- v + lam * (x0_hat - ref), where
                     x0_hat = x_t - t * v is the current estimate of the final action (pulls each step toward the
                     reference while the policy keeps it on its action manifold); 0 = off. Works with any mode.
  frs_mode "exact"   invert the reference EXACTLY through G0.5's own 10-step Euler sampler (each step solved backward by
                     fixed-point iteration), then run the sampler forward from that noise (reproduces the reference)
  frs_mode "noise"   start the sampler from the noise in z_init (model space [H, D])
  frs_mode "dsbc"    start the sampler from the noise predicted by the DSBC noise policy (DSBC_CKPT, 109_dsbc_train.py)
  rec_tag     str    record this request (images, state, task, executed model-space chunk and its exact noise) to REC_DIR
  cfg_neg/cfg_w      classifier-free guidance on the flow velocity against a negative prompt (e.g. the old command):
                     v <- v_neg + w * (v - v_neg); the negative row shares the start point and schedule
The reference is mapped into the model's normalised, merged action space with the policy's own processor (the same
path dataset actions take in training), so normalisation and padding match exactly.
Env: DSBC_CKPT (noise policy), EXPERT_CKPT (fine-tuned action expert), HYBRID_CKPT [+ HYBRID_BRANCH, default current]
(LaWAM adapter as in serve_policy_hybrid.py, e.g. the LIBERO-Mem model), REC_DIR (where tagged requests are saved).
"""
import os
import sys
import time
import types

JOBS = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.join(os.environ.get("G05_ROOT", "/export/home/kaziz/g05"), "GalaxeaVLA")


def pop_arg(name, default=None):
    if name in sys.argv:
        i = sys.argv.index(name)
        val = sys.argv[i + 1]
        del sys.argv[i:i + 2]
        return val
    return default


SEED = int(pop_arg("--seed", "7"))
sys.path.insert(0, JOBS)
sys.path.insert(0, PROJECT)
os.chdir(PROJECT)
import numpy as np  # noqa: E402
import torch  # noqa: E402
import t4_fp16  # noqa: E402,F401  (bf16 -> fp16 autocast on T4/V100; no-op on H200)
import scripts.serve_policy_batched as S  # noqa: E402
from g05.models.g05.inferencer import resolve_processor  # noqa: E402


class FRS:
    rows = {}  # batch row -> {"mode", "ref" [H, D] model space, "kb", "t0", "ts", "lam"}
    pairs = []  # (positive row, negative row, w) for classifier-free guidance
    rec = {}  # batch row -> tag, for requests to record (exact noise of the executed chunk is computed after sampling)
    rec_out = {}
    inv_err = None
    dsbc = None  # (noise policy, feature fn) once loaded
    diag = []  # per frs row: (|z|/sqrt(n) of the inverted start, |out - ref| / |ref|) for the log


def disable_flash_attn():
    """FlashAttention kernels need sm80+: route G0.5's vision tower and action codec to their SDPA paths."""
    for _name, _mod in list(sys.modules.items()):
        if "g05" in _name:
            for _attr in ("_flash_attn_varlen", "flash_attn_func"):
                if getattr(_mod, _attr, None) is not None:
                    setattr(_mod, _attr, None)


def frs_infer(self, model, attention_mask, pixel_values, past_key_values, action_dim_is_pad=None,
              position_ids_override=None, embodiment_types=None, **kwargs):
    """FMHelper.infer (pi_convention: t = 1 noise -> t = 0 data) with per-row start points and step sizes."""
    first_image = next(iter(pixel_values.values()))
    bsz, device, dtype = first_image.size(0), first_image.device, first_image.dtype
    vlm_kv = past_key_values
    pad = action_dim_is_pad.bool().unsqueeze(1) if (not self.zero_pad_action_target and action_dim_is_pad is not None) else None
    dummy = torch.zeros(bsz, self.horizon_steps, self.action_dim, device=device, dtype=dtype)
    x = self._sample_noise(dummy, dtype, embodiment_types)
    if pad is not None:
        x.masked_fill_(pad, 0.0)
    action_mask, action_pos = model.build_action_mask_and_position_ids(
        attention_mask, action_len=self.horizon_steps, position_ids_prefix=position_ids_override,
        split_index=attention_mask.size(1), dtype=dtype, action_causal=self.action_causal)
    if hasattr(model, "_build_prefix_action_kv"):
        ae_kv_kwargs = {"kv_cache": model._build_prefix_action_kv(vlm_kv, vlm_kv.num_items())}
    else:
        ae_kv_kwargs = {"past_key_values": vlm_kv}
    assert self.time_convention == "pi_convention", self.time_convention

    def vel(a, t):
        with torch.autocast(device.type, enabled=False):
            emb = model.action_expert.embed(a.float())
            tc = model.action_expert.encode_time(t)
        h = model.action_expert(inputs_embeds=emb, attention_mask=action_mask, position_ids=action_pos, time_cond=tc,
                                attn_implementation=model.attn_implementation, mixture_name="action", **ae_kv_kwargs)
        return model.action_expert.decode(h).to(a.dtype)

    rows = {i: r for i, r in FRS.rows.items() if i < bsz}
    pairs = [(i, j, w) for i, j, w in FRS.pairs if i < bsz and j < bsz]

    def gvel(a, t):  # velocity with classifier-free guidance on paired rows (negative rows follow the guided value)
        v = vel(a, t)
        if pairs:
            v = v.clone()
            for i, j, w in pairs:
                g = v[j] + w * (v[i] - v[j]); v[i] = g; v[j] = g
        return v

    t_start = torch.ones(bsz, device=device, dtype=dtype)
    keep = torch.zeros(bsz, dtype=torch.bool, device=device)  # rows returned as the reference itself
    refs = torch.zeros_like(x); lam = torch.zeros(bsz, device=device, dtype=dtype)
    for i, r in rows.items():
        ref = r["ref"].to(device=device, dtype=dtype); refs[i] = ref; lam[i] = r["lam"]
        if r["mode"] == "sde":
            x[i] = (1.0 - r["t0"]) * ref + r["t0"] * x[i]
            t_start[i] = r["t0"]
        elif r["mode"] in ("frs", "ref"):
            x[i] = ref
            keep[i] = r["mode"] == "ref"
    for i, j, _w in pairs:  # the negative row starts where its positive row starts
        x[j] = x[i]; t_start[j] = t_start[i]
    frs_rows = [i for i, r in rows.items() if r["mode"] == "frs"]
    frs_rows += [j for i, j, _w in pairs if i in frs_rows]
    if frs_rows:  # backward: data (t = 0) -> t = ts, x_{t+dt} = x_t + dt * v(x_t, t); only frs rows move
        kb = max(rows[i]["kb"] for i in frs_rows if i in rows)
        ts = torch.ones(bsz, device=device, dtype=dtype)
        for i in frs_rows:
            ts[i] = rows[i]["ts"] if i in rows else rows[next(p for p, q, _ in pairs if q == i)]["ts"]
        sel = torch.zeros(bsz, dtype=torch.bool, device=device); sel[frs_rows] = True
        t = torch.zeros(bsz, device=device, dtype=dtype)
        for _ in range(kb):
            v = gvel(x, t)
            x = torch.where(sel[:, None, None], x + (ts / kb)[:, None, None] * v, x)
            t = torch.where(sel, t + ts / kb, t)
            if pad is not None:
                x.masked_fill_(pad, 0.0)
        t_start = torch.where(sel, ts, t_start)
    exact_rows = [i for i, r in rows.items() if r["mode"] == "exact"]
    n_steps = self.num_inference_steps

    def invert_exact(target, sel, iters=int(os.environ.get("INV_ITERS", "30"))):
        """Noise z with forward(z) = target for G0.5's explicit Euler sampler: x_{k+1} = x_k - dt v(x_k, t_k), t_k = 1 - k dt.
        Each step is solved backward, x_k = x_{k+1} + dt v(x_k, t_k), by fixed-point iteration (only rows in sel move)."""
        dtk = 1.0 / n_steps
        xk1 = target.clone()
        for k in range(n_steps - 1, -1, -1):
            tk = torch.full((bsz,), 1.0 - k * dtk, device=device, dtype=dtype)
            xk = xk1 + dtk * gvel(xk1, tk)
            for _ in range(iters):
                xk = torch.where(sel[:, None, None], xk1 + dtk * gvel(xk, tk), xk1)
            if pad is not None:
                xk.masked_fill_(pad, 0.0)
            xk1 = xk
        return xk1

    if exact_rows:
        sel = torch.zeros(bsz, dtype=torch.bool, device=device); sel[exact_rows] = True
        x = torch.where(sel[:, None, None], invert_exact(refs, sel), x)
    for i, r in rows.items():  # noise given by the client or by the DSBC noise policy
        if r["mode"] in ("noise", "dsbc") and r.get("z") is not None:
            x[i] = r["z"].to(device=device, dtype=dtype); t_start[i] = 1.0
    z_used = x.clone()
    z_norm = {i: float(x[i].float().norm() / max(x[i].numel(), 1) ** 0.5) for i in frs_rows + exact_rows if i in rows}
    ref_x = x.clone()
    n = self.num_inference_steps
    dt = t_start / n
    t = t_start.clone()
    guided = bool((lam > 0).any())
    for _ in range(n):  # forward: x_{t-dt} = x_t - dt * v(x_t, t), per-row dt
        v = gvel(x, t)
        if guided:  # pull the predicted final action x0_hat = x_t - t v toward the reference
            x0_hat = x - t[:, None, None] * v
            v = v + lam[:, None, None] * (x0_hat - refs)
        x = x - dt[:, None, None] * v
        t = t - dt
        if pad is not None:
            x.masked_fill_(pad, 0.0)
    x = torch.where(keep[:, None, None], ref_x, x)
    if FRS.rec:  # exact noise of each executed chunk (ref rows: the reference; policy rows: the noise actually used)
        rec_rows = [i for i in FRS.rec if i < bsz]
        need = [i for i in rec_rows if bool(keep[i]) or rows.get(i, {}).get("mode") in ("frs", "sde")]
        z_exec = z_used.clone()
        if need:
            sel = torch.zeros(bsz, dtype=torch.bool, device=device); sel[need] = True
            z_exec = torch.where(sel[:, None, None], invert_exact(x, sel), z_exec)
        row_err = {i: 0.0 for i in rec_rows}  # policy rows: the noise is the one actually used, error 0
        if need:  # verify: the sampler run forward from the exact noise must give back the executed chunk
            xr = z_exec.clone(); tt = torch.ones(bsz, device=device, dtype=dtype); dd = 1.0 / n_steps
            for _ in range(n_steps):
                xr = xr - dd * gvel(xr, tt); tt = tt - dd
                if pad is not None:
                    xr.masked_fill_(pad, 0.0)
            for i in need:
                row_err[i] = float(((xr[i] - x[i]).float().norm() / (x[i].float().norm() + 1e-6)).item())
            FRS.inv_err = max(row_err[i] for i in need)
        FRS.rec_out = {i: (z_exec[i].float().cpu().numpy(), x[i].float().cpu().numpy(),
                           (pad[i, 0].cpu().numpy() if pad is not None else None), row_err[i]) for i in rec_rows}
    FRS.diag = [(z_norm.get(i, float("nan")), float((x[i] - refs[i]).float().norm() / (refs[i].float().norm() + 1e-6)))
                for i, r in rows.items() if r["mode"] != "ref"]
    if self.final_action_clip_value is not None:
        x = torch.clamp(x, -self.final_action_clip_value, self.final_action_clip_value)
    return x


def dsbc_noise(ro):
    """Noise for one request from the DSBC noise policy (109_dsbc_train.py): pooled DINOv3 features of the agent view, the
    proprioceptive state and the commanded task -> starting noise [H, D] for the frozen G0.5 sampler."""
    if FRS.dsbc is None:
        import importlib.util
        spec = importlib.util.spec_from_file_location("dsbc", os.path.join(JOBS, "109_dsbc_train.py"))
        D = importlib.util.module_from_spec(spec); spec.loader.exec_module(D)
        FRS.dsbc = D.load_policy(os.environ["DSBC_CKPT"], device="cuda")
    pol, feat_fn = FRS.dsbc
    img = np.asarray(ro["images"]["image"], np.uint8)
    state = np.concatenate([np.asarray(v, np.float32).reshape(-1) for _, v in sorted(ro["state"].items())])
    return pol(feat_fn(img), state, str(ro["task"]))  # the checkpoint maps the command to its vocabulary


_orig_setup = S.setup


def frs_setup(cfg, device="cuda"):
    torch.manual_seed(SEED)
    policy, processor = _orig_setup(cfg, device=device)
    if t4_fp16.T4_MODE:
        disable_flash_attn()
        policy.half()
        torch.cuda.empty_cache()

        def _cast_in(mod, args):  # fp32 activations created outside autocast -> match the fp16 weights
            if args and torch.is_tensor(args[0]) and args[0].is_floating_point() and args[0].dtype != mod.weight.dtype:
                return (args[0].to(mod.weight.dtype),) + tuple(args[1:])
        for _m in policy.modules():
            if isinstance(_m, (torch.nn.Linear, torch.nn.Conv2d, torch.nn.Conv3d, torch.nn.LayerNorm)) and getattr(_m, "weight", None) is not None:
                _m.register_forward_pre_hook(_cast_in)
    if os.environ.get("EXPERT_CKPT"):  # fine-tuned action expert (e.g. a PrefPI iteration, 121_prefpi_train.py)
        expert = next(m for n, m in policy.named_modules() if n.endswith("action_expert"))
        dt = next(expert.parameters()).dtype
        sd = torch.load(os.environ["EXPERT_CKPT"], map_location="cpu")
        expert.load_state_dict({k: v.to(dt) for k, v in sd.items()}, strict=True)
        print(f"[frs] loaded fine-tuned action expert from {os.environ['EXPERT_CKPT']}", flush=True)
    if os.environ.get("HYBRID_CKPT"):  # LaWAM adapter, as serve_policy_hybrid.py (e.g. the LIBERO-Mem model LMEMSTEP_ex_v100)
        import lawam_hybrid as H
        branch = os.environ.get("HYBRID_BRANCH", "current")
        hybrid = H.build_hybrid("adapter", branch, ckpt=os.environ["HYBRID_CKPT"], device=device)
        hybrid.eval()
        H.attach(policy, hybrid)
        if branch == "current" and t4_fp16.T4_MODE:  # current-frame adapter only needs DINOv3: keep the rest of LaWM on CPU
            for _n, _c in hybrid.lawm.named_children():
                if _n != "vision_encoder":
                    _c.to("cpu")
            torch.cuda.empty_cache()
        print(f"[hybrid] attached: ckpt={os.environ['HYBRID_CKPT']} branch={branch}", flush=True)
    fm = policy.model.fm_helper
    fm.infer = types.MethodType(frs_infer, fm)
    print(f"[frs] FRS sampler installed (steps={fm.num_inference_steps}, horizon={fm.horizon_steps}, dim={fm.action_dim})", flush=True)
    return policy, processor


def ref_to_model(od, ref, processor):
    """Raw-unit reference chunk -> normalised, merged model action [H, D] via the processor's training-time path."""
    import copy
    p = resolve_processor(processor, od)
    d = copy.deepcopy(od)
    H = next(iter(d["action"].values())).shape[0]
    r = np.asarray(ref, np.float32)
    r = np.concatenate([r, np.repeat(r[-1:], max(0, H - len(r)), 0)])[:H]
    act, o = {}, 0
    for m in p.shape_meta["action"]:
        act[m["key"]] = torch.from_numpy(r[:, o:o + m["raw_shape"]].copy()); o += m["raw_shape"]
    d["action"] = act
    d["action_is_pad"] = torch.zeros(H, dtype=torch.bool)
    return p.preprocess(d)["action"].float()


def _process_batch(self, requests):
    outcomes, valid, obs_dicts, extras = [], [], [], []
    for req in requests:
        ro = req.raw_obs
        ex = {"ref": ro.pop("ref_action", None), "mode": ro.pop("frs_mode", None), "kb": int(ro.pop("frs_back", 5)),
              "t0": float(ro.pop("sde_t0", 0.6)), "ts": float(ro.pop("frs_ts", 1.0)), "lam": float(ro.pop("guide_lam", 0.0)),
              "neg": ro.pop("cfg_neg", None), "w": float(ro.pop("cfg_w", 1.0)), "tag": ro.pop("rec_tag", None),
              "z": ro.pop("z_init", None)}
        if ex["mode"] == "dsbc":
            try:
                ex["z_model"] = dsbc_noise(ro)
            except Exception as exc:  # noqa: BLE001
                S.logger.exception("dsbc noise failed: %s", exc)
        elif ex["mode"] == "noise" and ex["z"] is not None:
            ex["z_model"] = torch.as_tensor(np.asarray(ex["z"], np.float32))
        try:
            S._parse_task_and_plan(ro)
            od = S.build_obs_dict(ro, self.processor)
            if ex["ref"] is not None and ex["mode"]:
                ex["ref_model"] = ref_to_model(od, ex["ref"], self.processor)
            elif ex["mode"] in ("noise", "dsbc") and "z_model" in ex:
                ex["ref_model"] = torch.zeros_like(ex["z_model"])  # placeholder so the row is configured
            if ex["tag"] is not None:
                ex["rec_obs"] = {"image": np.asarray(ro["images"]["image"], np.uint8), "wrist": np.asarray(ro["images"]["wrist_image"], np.uint8),
                                 "state": {k: np.asarray(v, np.float32) for k, v in ro["state"].items()}, "task": str(ro["task"])}
        except Exception as exc:
            outcomes.append(S.RequestOutcome(request=req, error=exc))
            continue
        valid.append(req); obs_dicts.append(od); extras.append(ex)
    if not obs_dicts:
        return outcomes
    n = len(obs_dicts)
    pairs = []
    for i in range(n):  # classifier-free guidance: one extra row per request with the negative prompt
        ex = extras[i]
        if ex["neg"] is not None and ex["w"] != 1.0:
            ro = dict(valid[i].raw_obs); ro["task"] = str(ex["neg"])
            S._parse_task_and_plan(ro)
            obs_dicts.append(S.build_obs_dict(ro, self.processor)); pairs.append((i, len(obs_dicts) - 1, ex["w"]))
    FRS.rows = {i: {"mode": ex["mode"], "ref": ex["ref_model"], "kb": ex["kb"], "t0": ex["t0"], "ts": ex["ts"], "lam": ex["lam"],
                    "z": ex.get("z_model")} for i, ex in enumerate(extras) if "ref_model" in ex}
    FRS.rec = {i: ex["tag"] for i, ex in enumerate(extras) if ex["tag"] is not None}
    FRS.rec_out, FRS.inv_err = {}, None
    FRS.pairs = pairs
    t0 = time.monotonic()
    try:
        actions, _timing = self.inferencer.infer_with_timing(obs_dicts)
    except Exception as exc:
        S.logger.exception("Dynamic batch inference failed")
        outcomes.extend(S.RequestOutcome(request=r, error=exc) for r in valid)
        return outcomes
    finally:
        n_ref = len(FRS.rows)
        FRS.rows, FRS.pairs = {}, []
    if FRS.rec_out:  # save recorded requests
        rd = os.environ.get("REC_DIR")
        if rd:
            os.makedirs(rd, exist_ok=True)
            for i, (z, xo, padm, err) in FRS.rec_out.items():
                ro = extras[i]["rec_obs"]
                np.savez_compressed(f"{rd}/{extras[i]['tag']}.npz", z=z, x=xo, pad=padm if padm is not None else np.zeros(0), inv_err=err,
                                    image=ro["image"], wrist=ro["wrist"], task=ro["task"],
                                    **{f"state_{k}": v for k, v in ro["state"].items()})
        if FRS.inv_err is not None:
            S.logger.info("exact inversion check: max |forward(z) - x| / |x| = %.2e", FRS.inv_err)
    FRS.rec, FRS.rec_out = {}, {}
    if FRS.diag:
        zs = [d[0] for d in FRS.diag if d[0] == d[0]]
        S.logger.info("frs diag: inverted-noise rms %s (Gaussian = 1.0); |out-ref|/|ref| mean %.3f",
                      f"{np.mean(zs):.3f}" if zs else "n/a", float(np.mean([d[1] for d in FRS.diag])))
        FRS.diag = []
    for i, (req, action) in enumerate(zip(valid, actions[:n])):
        outcomes.append(S.RequestOutcome(request=req, action=action))
        ex = extras[i]
        if ex.get("mode") == "ref" and "ref_model" in ex:  # round-trip check: raw ref -> model space -> postprocess
            pk = next(k for k in action if k.endswith("ee_pose")); gk = next((k for k in action if k.endswith("gripper")), None)
            out = np.concatenate([np.asarray(action[pk], np.float32).reshape(-1, 6),
                                  np.asarray(action[gk], np.float32).reshape(-1, 1)[:, :1] if gk else np.zeros((len(np.asarray(action[pk]).reshape(-1, 6)), 1), np.float32)], 1)
            ref = np.asarray(ex["ref"], np.float32)[:len(out)]
            S.logger.info("ref round-trip: max|out-ref| pose=%.4f gripper=%.4f (ref pose range %.3f)", float(np.abs(out[:, :6] - ref[:, :6]).max()),
                          float(np.abs(out[:, 6] - ref[:, 6]).max()), float(np.abs(ref[:, :6]).max()))
    S.logger.info("Batch forward: batch_size=%d (%d with reference, %d cfg rows), infer=%.1fms", n, n_ref, len(pairs), (time.monotonic() - t0) * 1000.0)
    return outcomes


S.setup = frs_setup
S.DynamicBatcher._process_batch = _process_batch

if __name__ == "__main__":
    S.main()
