from __future__ import annotations
"""Flow Reversal Steering client: gc_steer_client.py (command-switch probe on LIBERO-Goal) plus a coarse reference
action attached to every replan inside --frs_window, for 94_frs_serve.py. GalaxeaVLA itself is not modified.
  --frs_mode none|frs|sde|ref  what the server does with the reference (see 94_frs_serve.py); none = plain G0.5
  --frs_window LO:HI        policy steps [LO, HI) that carry a reference (default 40:80 = the 2 s after the switch)
  --frs_back K / --sde_t0 T backward Euler steps (frs) / noise level (sde)
  --frs_targets PATH        93_frs_targets.py output: per-task contact point, approach speed, action->metre scale
  The reference is a straight-line approach from the current end-effector position toward 4 cm above the commanded
  task's mean contact point (a stand-in for a human pointing at the target), gripper open, rotation zero.
  --init_task_id N          initial states from task N of the same suite (LIBERO-Goal: identical scene, different command)
  --prompt_schedule SPEC    instruction timeline over policy action steps, e.g. "0:target,40:blank" or "0:task3,40:target".
                            Items: target | task<i> (language of task i in the suite) | blank.
  --goal_schedule SPEC      goal timeline for the goal adapter, same syntax ("none" = never). Each segment restarts its
                            goal video: the goal at policy step s of a segment starting at k is the reference
                            --goal_ahead steps ahead, i.e. frame (s - k + goal_ahead) of the task's goal video.
  --goal_source demo|gen    demo: mean pooled DINOv3 features of the task's training demos, every 0.4 s (as in
                            49_wm_planner.py); gen: the subgoal generator (gc_subgoal.py) applied to the current frame.
  --cfg_neg_schedule SPEC   negative prompt for classifier-free guidance on the flow velocity ("none" = no guidance;
                            "40:task3" = from step 40 on, the old command task3 is the negative), scale --cfg_w.
Success is always the env's own check for --task_id (the commanded target).
"""

import argparse
import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import rootutils

import sys as _sys0
_sys0.path.insert(0, "/export/home/kaziz/g05/jobs")
import t4_fp16  # noqa: E402,F401
rootutils.setup_root("/export/home/kaziz/g05/GalaxeaVLA/experiments/libero/eval_libero_parallel.py", indicator=".python-version", pythonpath=True)

from tqdm import tqdm

from g05.utils.checkpoint.ckpt_utils import find_run_dir, load_config_from_run_dir
from g05.utils.config.config_resolvers import register_default_resolvers
from g05.utils.eval.eval_utils import filter_embodiment

from experiments.libero.libero_eval_utils import (
    LIBERO_DUMMY_ACTION,
    LIBERO_ENV_RESOLUTION as DEFAULT_ENV_RESOLUTION,
    LiberoGripperCommandState,
    action_dict_gripper_scalar,
    extract_libero_images,
    single_action_to_libero_action,
    build_libero_raw_obs,
    quat2axisangle,
    ensure_libero_config,
    ensure_libero_import,
    get_max_steps,
    save_rollout_video,
)
from experiments.libero.libero_vector_env import LiberoAsyncVectorEnv
from scripts.utils.policy_ws_client import MultiPolicyWSClient

register_default_resolvers()

logger = logging.getLogger(__name__)


def _json_default(obj: Any):
    if hasattr(obj, "item"):
        return obj.item()
    if hasattr(obj, "tolist"):
        return obj.tolist()
    return str(obj)


def _cfg_get(cfg, path: str, default=None):
    if cfg is None:
        return default
    cur = cfg
    for key in path.split("."):
        if cur is None or key not in cur:
            return default
        cur = cur[key]
    return cur


def _make_env_fn(task, resolution: int, seed: int | None):
    ensure_libero_config()
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    import pathlib

    task_bddl_file = (
        pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    )

    def _fn():
        env = OffScreenRenderEnv(
            bddl_file_name=task_bddl_file,
            camera_heights=resolution,
            camera_widths=resolution,
        )
        env.seed(seed)
        return env

    return _fn


def _make_dummy_env_fn(task, resolution: int, seed: int | None):
    ensure_libero_config()
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    import pathlib

    task_bddl_file = (
        pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    )

    def _fn():
        env = OffScreenRenderEnv(
            bddl_file_name=task_bddl_file,
            camera_heights=resolution,
            camera_widths=resolution,
            has_renderer=False,
        )
        env.seed(seed)
        return env

    return _fn


RECORD_DIR = None  # set from --record_dir
RECORD_FROM_STEP = 0


def _rec_state(obs):
    import numpy as _np
    return _np.concatenate([_np.asarray(obs["robot0_eef_pos"], _np.float32), quat2axisangle(obs["robot0_eef_quat"]).astype(_np.float32),
                            _np.asarray(obs["robot0_gripper_qpos"], _np.float32).reshape(-1)[:2]]).astype(_np.float32)


def _obj_z(obs):
    """Height of every object LIBERO reports a position for (PrefPI's carry-height preference)."""
    import numpy as _np
    out = {}
    for k, v in obs.items():
        if k.endswith("_pos") and not k.startswith("robot"):
            v = _np.asarray(v).reshape(-1)
            if v.size == 3:
                out[k[:-4]] = float(v[2])
    return out


PREF = {"prefix": "", "w": 1.0}  # PrefPI: preference-conditioned prompt prefix and its CFG weight against the plain prompt


def _rec_action(env_action):
    import numpy as _np
    a = _np.asarray(env_action, _np.float32).copy()
    a[-1] = (1.0 - a[-1]) / 2.0  # LIBERO env gripper (-1 open / +1 close) -> G0.5 dataset convention (1 open / 0 close)
    return a


MEM = None  # MEM-2: {"lam", "dec", "cond", "names", "allowed", "tau"} when --memory_decoder is given
MEMORY_TOKEN = "__MEMORY__"


def _mem_load(path, tau):
    import sys as _sys
    import torch as _t
    _sys.path.insert(0, "/export/home/kaziz/g05/jobs")
    import lawam_hybrid as _H
    from mem_decoder import Decoder as _D
    ck = _t.load(path, map_location="cpu", weights_only=False)
    dec = _D(ck["n_frames"], ck["n_tasks"])
    dec.load_state_dict(ck["model"]); dec.eval().cuda()
    suite_id = ck["suite_id"]
    goal = ck["suites"].index("libero_goal")
    allowed = _t.tensor([sid == goal for sid in suite_id]).cuda()
    return {"lam": _H.load_lawm(), "H": _H, "dec": dec, "cond": ck["cond"], "names": ck["task_names"], "allowed": allowed, "tau": tau}


def _mem_feats(images_hwc_list):
    """uint8 HxWx3 frames (policy view) -> pooled DINOv3 features [n, 64, 768] fp16 on GPU (same as 22_cache_feats.py)."""
    import numpy as _np
    import torch as _t
    import torch.nn.functional as _F
    x = _t.from_numpy(_np.stack(images_hwc_list)).permute(0, 3, 1, 2).cuda()
    f = MEM["H"].dino_from_uint8(MEM["lam"], x)
    b = f.shape[0]
    f = f.view(b, 16, 16, 768).permute(0, 3, 1, 2)
    return _F.avg_pool2d(f, 2).permute(0, 2, 3, 1).reshape(b, 64, 768).half()


def _mem_select(buf, snaps, cur):
    """Frame list for the decoder, identical selection rules to 29_mem_intent.py / 27_snapshot_memory.py."""
    t = len(buf)
    c = MEM["cond"]
    if c == "current":
        return [cur]
    if c == "first":
        return [buf[0] if t else cur, cur]
    if c == "offsets":
        return [buf[max(t - d, 0)] if t else cur for d in (160, 80, 40, 20)] + [cur]
    written = [k for k in snaps if k < t]
    if not written:
        return [cur] * 5
    sel = [written[0]] + written[-3:]
    while len(sel) < 4:
        sel.insert(1, sel[1] if len(sel) > 1 else sel[0])
    return [buf[k] for k in sorted(sel[:4])] + [cur]


def _mem_infer(buf, snaps, cur):
    import torch as _t
    x = _t.stack(_mem_select(buf, snaps, cur)).float()[None]
    with _t.no_grad():
        lg = MEM["dec"](x).float()[0].masked_fill(~MEM["allowed"], -1e9)
    return MEM["names"][int(lg.argmax())]


def _parse_schedule(spec: str, target: str, suite) -> list:
    segs = []
    for item in spec.split(","):
        k, src = item.split(":")
        if src == "target":
            text = target
        elif src == "blank":
            text = ""
        elif src == "none":
            text = None
        elif src == "memory":
            text = MEMORY_TOKEN
        elif src.startswith("task"):
            text = suite.get_task(int(src[4:])).language
        else:
            raise ValueError(src)
        segs.append((int(k), text))
    return sorted(segs, key=lambda x: x[0])


def make_prompt_fn(spec: str, target: str, suite) -> Any:
    segs = _parse_schedule(spec, target, suite)

    def fn(step: int) -> str:
        text = segs[0][1]
        for k, t in segs:
            if step >= k:
                text = t
        return text

    return fn


def make_segment_fn(spec: str, target: str, suite) -> Any:
    """step -> (text or None, start step of the current segment); "none" or steps before the first item -> None."""
    segs = [] if spec in ("none", "") else _parse_schedule(spec, target, suite)

    def fn(step: int):
        cur = (None, 0)
        for k, t in segs:
            if step >= k:
                cur = (t, k)
        return cur

    return fn


FRSC = None  # {"mode", "lo", "hi", "kb", "t0", "targets"} when --frs_mode != none


def _nrm(s):
    return " ".join(str(s).lower().replace("_", " ").split())


INIT_OFFSET = 0
FRS_ARRIVED = set()  # (env, trial) pairs whose arm reached the target: the reference stops for good


def frs_arrived(env: int, trial: int, obs, task: str) -> bool:
    """--frs_until D > 0: stop attaching references once the end effector is within D m of the target point."""
    if FRSC["until"] <= 0:
        return False
    if (env, trial) in FRS_ARRIVED:
        return True
    target = target_point(task, trial=trial)
    if float(np.linalg.norm(np.asarray(obs["robot0_eef_pos"], np.float64) - target)) < FRSC["until"]:
        FRS_ARRIVED.add((env, trial))
        return True
    return False


def target_point(task: str, env: int = -1, trial: int = -1):
    """Contact point (+4 cm above) for the task, plus a fixed random offset per (env, trial) when --frs_target_noise > 0
    (simulates the 2-5 cm error of a human click or a VLM pointing at the target)."""
    tg = FRSC["targets"]
    p = np.asarray(tg["tasks"][_nrm(task)]["point"], np.float64) + np.array([0.0, 0.0, 0.04])
    sig = FRSC.get("noise", 0.0)
    if sig > 0:
        rng = np.random.default_rng(abs(hash((_nrm(task), int(trial), FRSC.get("noise_seed", 0)))) % (2 ** 32))
        p = p + rng.normal(0.0, sig, 3)
    return p


def coarse_ref(obs, task: str, horizon: int = 32, trial: int = -1):
    """Straight-line approach chunk [horizon, 7] in raw action units toward 4 cm above the task's contact point."""
    tg = FRSC["targets"]
    T = tg["tasks"][_nrm(task)]
    k = np.asarray(tg["k"], np.float64)
    target = target_point(task, trial=trial)
    pos = np.asarray(obs["robot0_eef_pos"], np.float64).copy()
    out = np.zeros((horizon, 7), np.float32)
    out[:, 6] = tg["gripper_open"]
    for h in range(horizon):
        units = (target - pos) / k
        n = float(np.linalg.norm(units))
        a = units / n * min(T["speed"], n) if n > 1e-6 else np.zeros(3)
        out[h, :3] = a
        pos = pos + a * k
    return out


GC = None  # {"refs": {lang: [T,64,768] fp16}, "ahead", "source", "gen", "lam", "task_ids", "w"} (goal / CFG runs)


def _goal_refs(langs):
    """Goal videos: per LIBERO task language, the mean pooled DINOv3 features of its training demos every 8 frames
    (0.4 s), each demo clamped to its last frame (identical to 49_wm_planner.py --goal_type timed)."""
    import glob as _glob
    import numpy as _np
    import pandas as _pd
    G_ = "/export/home/kaziz/g05"
    C = f"{G_}/results/feat_cache"; DS = f"{G_}/data/libero_merged_g05grip"
    m = _np.load(f"{C}/meta.npz")
    d = _pd.concat([_pd.read_parquet(f, columns=["index", "task_index"]) for f in sorted(_glob.glob(f"{DS}/data/*/*.parquet"))]).sort_values("index")
    tidx = d["task_index"].values
    tp = _pd.read_parquet(f"{DS}/meta/tasks.parquet").reset_index()
    tcol = [c for c in tp.columns if tp[c].dtype == object][0]
    norm = lambda s: " ".join(str(s).lower().replace("_", " ").split())  # noqa: E731
    name2idx = {norm(r[tcol]): int(r["task_index"]) for _, r in tp.iterrows()}
    feats = _np.load(f"{C}/feats.npy", mmap_mode="r")
    fr, L = m["frame_index"], m["ep_length"]
    refs, task_ids = {}, {}
    for lang in langs:
        task_ids[lang] = name2idx[norm(lang)]
        rows_t = _np.nonzero(tidx == task_ids[lang])[0]
        starts = rows_t[fr[rows_t] == 0]
        T_ = int(L[starts].max())
        refs[lang] = _np.stack([_np.asarray(feats[_np.sort(starts + _np.minimum(f_, L[starts] - 1))], dtype=_np.float32).mean(0)
                                for f_ in range(0, T_, 8)]).astype(_np.float16)
        logger.info("goal video '%s': %d demos, %d reference points", lang, len(starts), len(refs[lang]))
    return refs, task_ids


def _gc_load(args, langs):
    import sys as _sys
    _sys.path.insert(0, "/export/home/kaziz/g05/jobs")
    st = {"ahead": args.goal_ahead, "source": args.goal_source, "w": args.cfg_w, "gen": None, "lam": None}
    st["refs"], st["task_ids"] = _goal_refs(langs)
    if args.goal_source == "gen":
        import torch as _t
        import lawam_hybrid as _H
        from gc_subgoal import load_generator
        st["gen"] = load_generator(args.subgoal_ckpt, "cuda")
        st["lam"], st["H"] = _H.load_lawm(), _H
        for _n, _c in st["lam"].named_children():
            if _n != "vision_encoder":
                _c.to("cpu")
        _t.cuda.empty_cache()
    return st


def _goal_feats(env_ids, texts, starts, steps, obs_list):
    """Goal features [64,768] fp16 (numpy) per env, or None where the goal schedule says none."""
    out = {}
    need_gen = [i for i, t in zip(env_ids, texts) if t is not None and GC["source"] == "gen"]
    gen_feats = {}
    if need_gen:
        import numpy as _np
        import torch as _t
        import torch.nn.functional as _F
        x = _t.from_numpy(_np.stack([extract_libero_images(obs_list[i])["image"] for i in need_gen])).permute(0, 3, 1, 2).cuda()
        f = GC["H"].dino_from_uint8(GC["lam"], x)
        b = f.shape[0]
        u = _F.avg_pool2d(f.view(b, 16, 16, 768).permute(0, 3, 1, 2), 2).permute(0, 2, 3, 1).reshape(b, 64, 768)
        tids = [GC["task_ids"][t] for i, t in zip(env_ids, texts) if i in need_gen]
        g = GC["gen"].predict(u, tids)
        gen_feats = {i: g[j].half().cpu().numpy() for j, i in enumerate(need_gen)}
    for i, t, k, s in zip(env_ids, texts, starts, steps):
        if t is None:
            out[i] = None
        elif GC["source"] == "gen":
            out[i] = gen_feats[i]
        else:
            ref = GC["refs"][t]
            out[i] = ref[min((s - k + GC["ahead"]) // 8, len(ref) - 1)]
    return out


async def run_parallel_task(
    task_id: int,
    task_description: str,
    initial_states: list,
    num_trials: int,
    max_steps: int,
    num_steps_wait: int,
    num_parallel: int,
    client: MultiPolicyWSClient,
    vec_env: LiberoAsyncVectorEnv,
    embodiment_type: str | None,
    save_videos: bool = False,
    output_dir: Path = Path("outputs/libero_parallel_eval"),
    prompt_fn=None,
    goal_fn=None,
    cfg_fn=None,
) -> dict[str, Any]:
    """Run parallel evaluation.

    Logic per iteration:
      1. If all active envs are in dummy phase, step them with dummy actions.
      2. Otherwise, for each ready env:
         - If cache has actions -> consume locally (instant).
         - If needs recompute -> send request to server.
      3. Server decides when to batch-infer (max_batch_size OR max_wait_ms).
         Client just awaits the response.
      4. Step all envs that have actions.
    """
    trial_successes: dict[int, bool] = {}
    trial_infer_calls: dict[int, int] = {}
    trial_replay_images: dict[int, list] = {}
    completed = 0
    next_trial_idx = 0

    env_trial_id: list[int] = [0] * num_parallel
    env_step_count: list[int] = [0] * num_parallel
    env_action_step: list[int] = [0] * num_parallel
    env_infer_calls: list[int] = [0] * num_parallel
    env_replay: list[list] = [[] for _ in range(num_parallel)]
    env_rec: list[list] = [[] for _ in range(num_parallel)]
    env_mbuf: list[list] = [[] for _ in range(num_parallel)]
    env_msnap: list[list] = [[] for _ in range(num_parallel)]
    env_minf: list = [None] * num_parallel
    trial_inferred: dict[int, str] = {}

    def finish_episode(env_idx: int, success: bool):
        nonlocal next_trial_idx, completed
        tid = env_trial_id[env_idx]
        if tid < 0 or tid in trial_successes:
            return None
        trial_successes[tid] = success
        if env_minf[env_idx] is not None:
            trial_inferred[tid] = env_minf[env_idx]
        env_mbuf[env_idx], env_msnap[env_idx], env_minf[env_idx] = [], [], None
        if RECORD_DIR is not None and success and env_rec[env_idx]:
            import numpy as _np
            recs = [r for r in env_rec[env_idx] if r[0] >= RECORD_FROM_STEP]
            if len(recs) >= 8:
                Path(RECORD_DIR).mkdir(parents=True, exist_ok=True)
                objz = {f"objz_{k}": _np.array([r[6].get(k, _np.nan) for r in recs], _np.float32) for k in recs[0][6]}
                _np.savez_compressed(
                    f"{RECORD_DIR}/task{task_id}_trial{tid}.npz",
                    step=_np.array([r[0] for r in recs]), image=_np.stack([r[1] for r in recs]),
                    wrist_image=_np.stack([r[2] for r in recs]), state=_np.stack([r[3] for r in recs]),
                    action=_np.stack([r[4] for r in recs]), prompt=_np.array([r[5] for r in recs]),
                    task_description=task_description, **objz)
        env_rec[env_idx] = []
        trial_infer_calls[tid] = env_infer_calls[env_idx]
        if save_videos:
            trial_replay_images[tid] = list(env_replay[env_idx])
        completed += 1
        env_step_count[env_idx] = 0
        env_action_step[env_idx] = 0
        env_infer_calls[env_idx] = 0
        env_replay[env_idx] = []
        client.clear_chunk(env_idx)
        if next_trial_idx < num_trials:
            _, state = assign_trial(env_idx)
            return state
        else:
            env_trial_id[env_idx] = -1
            return None

    def assign_trial(env_idx: int) -> tuple[int, Any]:
        nonlocal next_trial_idx
        trial_id = next_trial_idx
        state_idx = (trial_id + INIT_OFFSET) % len(initial_states)  # --init_offset: held-out initial states
        next_trial_idx += 1
        env_trial_id[env_idx] = trial_id
        env_rec[env_idx] = []
        env_mbuf[env_idx], env_msnap[env_idx], env_minf[env_idx] = [], [], None
        env_step_count[env_idx] = 0
        env_action_step[env_idx] = 0
        env_infer_calls[env_idx] = 0
        env_replay[env_idx] = []
        return trial_id, initial_states[state_idx]

    init_states = []
    for i in range(num_parallel):
        _, state = assign_trial(i)
        init_states.append(state)

    await asyncio.to_thread(vec_env.reset)
    obs_batch = await asyncio.to_thread(vec_env.set_init_state_each, init_states)
    per_env_obs: list[dict | None] = list(obs_batch)
    gripper_states = [LiberoGripperCommandState.from_obs(obs) for obs in per_env_obs]

    await client.reset_all()

    total_steps = max_steps + num_steps_wait
    global_progress = tqdm(
        total=num_trials * total_steps,
        desc=f"Task {task_id}",
        leave=False,
    )

    while completed < num_trials:
        if next_trial_idx >= num_trials and all(env_trial_id[i] < 0 for i in range(num_parallel)):
            break

        # --- Handle episodes that exceeded max steps ---
        for i in range(num_parallel):
            if env_trial_id[i] >= 0 and env_step_count[i] >= total_steps:
                await client.reset_connection(i)
                new_state = finish_episode(i, False)
                if new_state is not None:
                    per_env_obs[i] = await asyncio.to_thread(
                        vec_env.set_init_state_single, i, new_state
                    )
                    gripper_states[i].reset_from_obs(per_env_obs[i])

        active_envs = [i for i in range(num_parallel) if env_trial_id[i] >= 0]
        if not active_envs:
            continue

        # --- Phase: dummy steps (all envs still in warmup) ---
        all_in_dummy = all(env_step_count[i] < num_steps_wait for i in active_envs)
        if all_in_dummy:
            active_mask = [False] * num_parallel
            actions = [LIBERO_DUMMY_ACTION] * num_parallel
            for i in active_envs:
                active_mask[i] = True
                gripper_states[i].set_env_command(LIBERO_DUMMY_ACTION[-1])
                if env_step_count[i] == 0:
                    logger.info(
                        "env %d trial %d warmup gripper env_command=%.1f",
                        i,
                        env_trial_id[i],
                        LIBERO_DUMMY_ACTION[-1],
                    )
            obs_batch, _, _, infos = await asyncio.to_thread(
                vec_env.step_selective, active_mask, actions
            )
            for i in active_envs:
                if obs_batch[i] is not None:
                    per_env_obs[i] = obs_batch[i]
                env_step_count[i] += 1
                global_progress.update(1)
                if infos[i].get("_done"):
                    await client.reset_connection(i)
                    client.clear_chunk(i)
                    new_state = finish_episode(i, True)
                    if new_state is not None:
                        per_env_obs[i] = await asyncio.to_thread(
                            vec_env.set_init_state_single, i, new_state
                        )
                        gripper_states[i].reset_from_obs(per_env_obs[i])
            continue

        # --- Phase: normal action steps ---
        # First, let all cached envs consume their remaining actions quickly
        # so they also enter recompute state before we send requests to server.
        # This maximizes the batch size on the server side.
        dummy_envs = [i for i in active_envs if env_step_count[i] < num_steps_wait]
        ready_envs = [i for i in active_envs if env_step_count[i] >= num_steps_wait]

        # Drain cache: step all envs that have cached actions until they
        # exhaust their cache or hit max_steps. This is fast (no server call).
        while True:
            cached_envs = [
                i
                for i in ready_envs
                if env_trial_id[i] >= 0
                and env_step_count[i] < total_steps
                and not client.needs_recompute(i)
            ]
            if not cached_envs:
                break

            # Step cached envs + dummy envs together
            active_mask = [False] * num_parallel
            actions = [LIBERO_DUMMY_ACTION] * num_parallel

            for i in dummy_envs:
                if env_trial_id[i] >= 0 and env_step_count[i] < num_steps_wait:
                    active_mask[i] = True
                    gripper_states[i].set_env_command(LIBERO_DUMMY_ACTION[-1])

            for i in cached_envs:
                action = client.get_cached_action(i)
                if action is None:
                    continue
                actions[i] = single_action_to_libero_action(
                    action,
                    obs=per_env_obs[i],
                    gripper_state=gripper_states[i],
                )
                if env_action_step[i] == 0:
                    logger.info(
                        "env %d trial %d first cached policy_gripper=%s env_command=%.1f",
                        i,
                        env_trial_id[i],
                        action_dict_gripper_scalar(action),
                        actions[i][-1],
                    )
                active_mask[i] = True

            if MEM is not None:
                _envs = [i for i in cached_envs if active_mask[i] and actions[i] is not LIBERO_DUMMY_ACTION]
                if _envs:
                    _f = _mem_feats([extract_libero_images(per_env_obs[i])["image"] for i in _envs])
                    for _j, i in enumerate(_envs):
                        buf = env_mbuf[i]
                        if not buf:
                            env_msnap[i].append(0)
                        elif float(((_f[_j].float() - buf[env_msnap[i][-1]].float()) ** 2).mean()) > MEM["tau"]:
                            env_msnap[i].append(len(buf))
                        buf.append(_f[_j])
            if RECORD_DIR is not None:
                for i in cached_envs:
                    if active_mask[i] and actions[i] is not LIBERO_DUMMY_ACTION:
                        imgs = extract_libero_images(per_env_obs[i])
                        env_rec[i].append((env_action_step[i], imgs["image"], imgs["wrist_image"], _rec_state(per_env_obs[i]),
                                           _rec_action(actions[i]), prompt_fn(env_action_step[i]) if prompt_fn else task_description,
                                           _obj_z(per_env_obs[i])))
            if not any(active_mask):
                break

            obs_batch, _, _, infos = await asyncio.to_thread(
                vec_env.step_selective, active_mask, actions
            )

            for i in range(num_parallel):
                if not active_mask[i]:
                    continue
                if save_videos and env_step_count[i] >= num_steps_wait:
                    images_hwc = extract_libero_images(per_env_obs[i])
                    env_replay[i].append(images_hwc)
                if obs_batch[i] is not None:
                    per_env_obs[i] = obs_batch[i]
                env_step_count[i] += 1
                if env_step_count[i] > num_steps_wait:
                    env_action_step[i] += 1
                global_progress.update(1)
                if infos[i].get("_done"):
                    await client.reset_connection(i)
                    client.clear_chunk(i)
                    new_state = finish_episode(i, True)
                    if new_state is not None:
                        per_env_obs[i] = await asyncio.to_thread(
                            vec_env.set_init_state_single, i, new_state
                        )
                        gripper_states[i].reset_from_obs(per_env_obs[i])

            # Update ready_envs for next drain iteration
            active_envs = [i for i in range(num_parallel) if env_trial_id[i] >= 0]
            dummy_envs = [i for i in active_envs if env_step_count[i] < num_steps_wait]
            ready_envs = [i for i in active_envs if env_step_count[i] >= num_steps_wait]

        # Now all ready envs should need recompute (cache exhausted).
        # Collect and send all recompute requests at once.
        recompute_envs = [
            i
            for i in range(num_parallel)
            if env_trial_id[i] >= 0
            and env_step_count[i] >= num_steps_wait
            and env_step_count[i] < total_steps
            and client.needs_recompute(i)
        ]

        if not recompute_envs:
            # Nothing to do (all envs are in dummy or finished)
            await asyncio.sleep(0)
            continue

        n_dummy = len(
            [
                i
                for i in range(num_parallel)
                if env_trial_id[i] >= 0 and env_step_count[i] < num_steps_wait
            ]
        )
        n_finished = len([i for i in range(num_parallel) if env_trial_id[i] < 0])
        logger.info(
            "Sending recompute: %d envs (dummy=%d, finished=%d, total_parallel=%d)",
            len(recompute_envs),
            n_dummy,
            n_finished,
            num_parallel,
        )

        # Send all recompute requests to server concurrently.
        # Server will batch them and return together.
        raw_obs_list = []
        current_steps_list = []
        _goals = {}
        if goal_fn is not None:
            _segs = [goal_fn(env_action_step[i]) for i in recompute_envs]
            _goals = _goal_feats(recompute_envs, [g[0] for g in _segs], [g[1] for g in _segs],
                                 [env_action_step[i] for i in recompute_envs], per_env_obs)
        for i in recompute_envs:
            _txt = prompt_fn(env_action_step[i]) if prompt_fn else task_description
            if _txt == MEMORY_TOKEN:
                if env_minf[i] is None:  # retrieve once, when the instruction is lost
                    _cur = _mem_feats([extract_libero_images(per_env_obs[i])["image"]])[0]
                    env_minf[i] = _mem_infer(env_mbuf[i], env_msnap[i], _cur)
                    logger.info("env %d trial %d memory inferred: %r (target %r)", i, env_trial_id[i], env_minf[i], task_description)
                _txt = env_minf[i]
            _plain = _txt
            if PREF["prefix"]:  # PrefPI: the preference-conditioned branch is the instruction behind a "[cfg]" prefix
                _txt = PREF["prefix"] + _txt
            raw_obs, _ = build_libero_raw_obs(
                obs=per_env_obs[i],
                task_description=_txt,
                embodiment_type=embodiment_type,
            )
            if _goals.get(i) is not None:
                raw_obs["goal_feat"] = _goals[i]
            _neg = cfg_fn(env_action_step[i])[0] if cfg_fn is not None else None
            if _neg is not None and GC["w"] != 1.0:
                raw_obs["cfg_neg"], raw_obs["cfg_w"] = _neg, float(GC["w"])
            if FRSC is not None and FRSC["lo"] <= env_action_step[i] < FRSC["hi"] and not frs_arrived(i, env_trial_id[i], per_env_obs[i], task_description):
                if FRSC["mode"] != "dsbc":  # DSBC gets no reference: the noise policy alone steers the frozen sampler
                    raw_obs["ref_action"] = coarse_ref(per_env_obs[i], task_description, trial=env_trial_id[i])
                if FRSC.get("rec"):
                    raw_obs["rec_tag"] = f"{FRSC['rec']}_t{task_id}_tr{env_trial_id[i] + INIT_OFFSET}_s{env_action_step[i]}"
                raw_obs["frs_mode"], raw_obs["frs_back"], raw_obs["sde_t0"] = FRSC["mode"], FRSC["kb"], FRSC["t0"]
                raw_obs["frs_ts"], raw_obs["guide_lam"] = FRSC["ts"], FRSC["lam"]
                if FRSC["cfg"] != "none" and FRSC["cfg_w"] != 1.0:  # guidance away from the old command (or a blank one)
                    raw_obs["cfg_neg"] = (prompt_fn(0) if prompt_fn else "") if FRSC["cfg"] == "old" else ""
                    raw_obs["cfg_w"] = FRSC["cfg_w"]
            if PREF["prefix"] and PREF["w"] != 1.0:  # PrefPI: v = v_plain + w (v_pref - v_plain)
                raw_obs["cfg_neg"], raw_obs["cfg_w"] = _plain, PREF["w"]
            raw_obs_list.append(raw_obs)
            current_steps_list.append(env_action_step[i])

        await client.infer_batch(recompute_envs, raw_obs_list, current_steps_list)
        for i in recompute_envs:
            env_infer_calls[i] += 1

        # After recompute, step all recompute envs once (consume first action
        # from the fresh chunk) + dummy envs.
        active_mask = [False] * num_parallel
        actions = [LIBERO_DUMMY_ACTION] * num_parallel

        for i in dummy_envs:
            if env_trial_id[i] >= 0 and env_step_count[i] < num_steps_wait:
                active_mask[i] = True
                gripper_states[i].set_env_command(LIBERO_DUMMY_ACTION[-1])

        for i in recompute_envs:
            action = client.get_cached_action(i)
            if action is None:
                logger.warning("env %d has no cached action after infer, skipping step", i)
                continue
            actions[i] = single_action_to_libero_action(
                action,
                obs=per_env_obs[i],
                gripper_state=gripper_states[i],
            )
            if env_action_step[i] == 0:
                logger.info(
                    "env %d trial %d first recompute policy_gripper=%s env_command=%.1f",
                    i,
                    env_trial_id[i],
                    action_dict_gripper_scalar(action),
                    actions[i][-1],
                )
            active_mask[i] = True

            if MEM is not None:
                _envs = [i for i in recompute_envs if active_mask[i] and actions[i] is not LIBERO_DUMMY_ACTION]
                if _envs:
                    _f = _mem_feats([extract_libero_images(per_env_obs[i])["image"] for i in _envs])
                    for _j, i in enumerate(_envs):
                        buf = env_mbuf[i]
                        if not buf:
                            env_msnap[i].append(0)
                        elif float(((_f[_j].float() - buf[env_msnap[i][-1]].float()) ** 2).mean()) > MEM["tau"]:
                            env_msnap[i].append(len(buf))
                        buf.append(_f[_j])
            if RECORD_DIR is not None:
                for i in recompute_envs:
                    if active_mask[i] and actions[i] is not LIBERO_DUMMY_ACTION:
                        imgs = extract_libero_images(per_env_obs[i])
                        env_rec[i].append((env_action_step[i], imgs["image"], imgs["wrist_image"], _rec_state(per_env_obs[i]),
                                           _rec_action(actions[i]), prompt_fn(env_action_step[i]) if prompt_fn else task_description,
                                           _obj_z(per_env_obs[i])))
        if not any(active_mask):
            await asyncio.sleep(0)
            continue

        obs_batch, _, _, infos = await asyncio.to_thread(
            vec_env.step_selective, active_mask, actions
        )

        for i in range(num_parallel):
            if not active_mask[i]:
                continue
            if save_videos and env_step_count[i] >= num_steps_wait:
                images_hwc = extract_libero_images(per_env_obs[i])
                env_replay[i].append(images_hwc)
            if obs_batch[i] is not None:
                per_env_obs[i] = obs_batch[i]
            env_step_count[i] += 1
            if env_step_count[i] > num_steps_wait:
                env_action_step[i] += 1
            global_progress.update(1)
            if infos[i].get("_done"):
                await client.reset_connection(i)
                client.clear_chunk(i)
                new_state = finish_episode(i, True)
                if new_state is not None:
                    per_env_obs[i] = await asyncio.to_thread(
                        vec_env.set_init_state_single, i, new_state
                    )
                    gripper_states[i].reset_from_obs(per_env_obs[i])

    global_progress.close()

    successes = sum(1 for v in trial_successes.values() if v)
    task_result = {
        "task_id": task_id,
        "task_description": task_description,
        "successes": successes,
        "total_episodes": num_trials,
        "success_rate": successes / num_trials if num_trials > 0 else 0.0,
        "infer_calls": sum(trial_infer_calls.values()),
        "trial_successes": {int(k): bool(v) for k, v in sorted(trial_successes.items())},  # g05: per-trial outcomes for paired analysis
        "trial_inferred": {int(k): v for k, v in sorted(trial_inferred.items())},
    }

    if save_videos:
        for tid, success in trial_successes.items():
            images = trial_replay_images.get(tid, [])
            if images:
                save_rollout_video(
                    output_dir / "videos",
                    task_description,
                    tid,
                    success,
                    images,
                    fps=20,
                )

    return task_result


async def evaluate(args, cfg, embodiment_type: str | None) -> dict[str, Any]:
    global RECORD_DIR, RECORD_FROM_STEP, MEM, GC, FRSC, INIT_OFFSET
    RECORD_DIR, RECORD_FROM_STEP = args.record_dir, args.record_from
    INIT_OFFSET = args.init_offset
    PREF["prefix"], PREF["w"] = args.prompt_prefix, args.cfg_plain_w
    if args.frs_mode != "none":
        lo, hi = (int(x) for x in args.frs_window.split(":"))
        FRSC = {"mode": args.frs_mode, "lo": lo, "hi": hi, "kb": args.frs_back, "t0": args.sde_t0, "until": args.frs_until,
                "ts": args.frs_ts, "lam": args.guide_lam, "cfg": args.frs_cfg, "cfg_w": args.frs_cfg_w, "noise": args.frs_target_noise,
                "rec": args.frs_rec,
                "targets": json.load(open(args.frs_targets))}
        logger.info("FRS: mode=%s window=[%d,%d) back=%d t0=%.2f", args.frs_mode, lo, hi, args.frs_back, args.sde_t0)
    if args.memory_decoder:
        MEM = _mem_load(args.memory_decoder, args.memory_tau)
    if args.goal_schedule != "none" or args.cfg_neg_schedule != "none":
        ensure_libero_import()
        ensure_libero_config()
        from libero.libero import benchmark as _bm
        _suite = _bm.get_benchmark_dict()[args.task_suite_name]()
        GC = _gc_load(args, [_suite.get_task(k).language for k in range(_suite.n_tasks)])
    ensure_libero_import()
    ensure_libero_config()
    from libero.libero import benchmark

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    max_steps = get_max_steps(args.task_suite_name)

    if args.task_id is None:
        task_ids = list(range(task_suite.n_tasks))
    else:
        task_ids = [args.task_id]

    summary: dict[str, Any] = {
        "server_uri": args.server_uri,
        "ckpt_path": args.ckpt_path,
        "task_suite_name": args.task_suite_name,
        "task_ids": task_ids,
        "num_trials_per_task": args.num_trials_per_task,
        "num_parallel": args.num_parallel,
        "num_steps_wait": args.num_steps_wait,
        "embodiment_type": embodiment_type,
        "tasks": [],
        "total_episodes": 0,
        "total_successes": 0,
        "total_infer_calls": 0,
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    async with MultiPolicyWSClient(
        args.server_uri, args.num_parallel, timeout_s=args.timeout_s
    ) as client:
        logger.info(
            "Connected to policy server %s with %d parallel connections",
            args.server_uri,
            args.num_parallel,
        )
        summary["server_metadata"] = client.metadata

        for task_id in tqdm(task_ids, desc="Tasks"):
            task = task_suite.get_task(task_id)
            init_tid = args.init_task_id if args.init_task_id is not None else task_id
            initial_states = task_suite.get_task_init_states(init_tid)
            task_description = task.language
            if len(initial_states) < args.num_trials_per_task:
                logger.warning(
                    "Task %s only has %d initial states, cycling to %d trials",
                    task_id,
                    len(initial_states),
                    args.num_trials_per_task,
                )

            env_fn = _make_env_fn(task, args.env_resolution, args.seed)
            dummy_fn = _make_dummy_env_fn(task, args.env_resolution, args.seed)

            vec_env = await asyncio.to_thread(
                LiberoAsyncVectorEnv,
                env_fns=[env_fn] * args.num_parallel,
                dummy_env_fn=dummy_fn,
            )

            try:
                task_result = await run_parallel_task(
                    task_id=task_id,
                    task_description=task_description,
                    initial_states=initial_states,
                    num_trials=args.num_trials_per_task,
                    max_steps=max_steps,
                    num_steps_wait=args.num_steps_wait,
                    num_parallel=args.num_parallel,
                    client=client,
                    vec_env=vec_env,
                    embodiment_type=embodiment_type,
                    save_videos=args.save_videos,
                    output_dir=args.output_dir,
                    prompt_fn=make_prompt_fn(args.prompt_schedule, task_description, task_suite),
                    goal_fn=make_segment_fn(args.goal_schedule, task_description, task_suite) if GC is not None else None,
                    cfg_fn=make_segment_fn(args.cfg_neg_schedule, task_description, task_suite) if GC is not None else None,
                )
                task_result["init_task_id"] = init_tid
                task_result["prompt_schedule"] = args.prompt_schedule
                task_result["goal_schedule"] = args.goal_schedule
                task_result["goal_source"] = args.goal_source
                task_result["cfg_neg_schedule"] = args.cfg_neg_schedule
                task_result["cfg_w"] = args.cfg_w
                task_result["frs"] = {"mode": args.frs_mode, "window": args.frs_window, "back": args.frs_back, "t0": args.sde_t0, "until": args.frs_until,
                                      "ts": args.frs_ts, "lam": args.guide_lam, "cfg": args.frs_cfg, "cfg_w": args.frs_cfg_w, "target_noise": args.frs_target_noise, "init_offset": args.init_offset}
            finally:
                await asyncio.to_thread(vec_env.close)

            summary["tasks"].append(task_result)
            summary["total_successes"] += task_result["successes"]
            summary["total_episodes"] += task_result["total_episodes"]
            summary["total_infer_calls"] += task_result["infer_calls"]
            logger.info(
                "Task %s success rate: %.3f (%d/%d)",
                task_id,
                task_result["success_rate"],
                task_result["successes"],
                task_result["total_episodes"],
            )

    summary["success_rate"] = (
        summary["total_successes"] / summary["total_episodes"] if summary["total_episodes"] else 0.0
    )
    summary["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    return summary


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate remote policy server on LIBERO (parallel)"
    )
    parser.add_argument("--ckpt_path", default=None)
    parser.add_argument("--server_uri", default="ws://127.0.0.1:8765")
    parser.add_argument("--task_suite_name", default="libero_10")
    parser.add_argument("--task_id", type=int, default=None)
    parser.add_argument("--num_trials_per_task", type=int, default=50)
    parser.add_argument("--num_parallel", type=int, default=5)
    parser.add_argument("--num_steps_wait", type=int, default=10)
    parser.add_argument("--embodiment_type", default="libero")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--timeout_s", type=float, default=300.0)
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/libero_parallel_eval"))
    parser.add_argument("--save_videos", action="store_true")
    parser.add_argument("--env_resolution", type=int, default=DEFAULT_ENV_RESOLUTION)
    parser.add_argument("--log_level", default="INFO")
    parser.add_argument("--init_task_id", type=int, default=None)
    parser.add_argument("--prompt_schedule", default="0:target")
    parser.add_argument("--record_dir", default=None, help="save successful episodes (policy steps >= --record_from)")
    parser.add_argument("--record_from", type=int, default=0)
    parser.add_argument("--memory_decoder", default=None, help="MEM-2: decoder .pt from 29_mem_intent.py")
    parser.add_argument("--memory_tau", type=float, default=0.11319, help="snapshot write threshold (27_snapshot_memory.py)")
    parser.add_argument("--goal_schedule", default="none")
    parser.add_argument("--goal_ahead", type=int, default=32, help="goal = goal video this many policy steps ahead (1.6 s)")
    parser.add_argument("--goal_source", default="demo", choices=["demo", "gen"])
    parser.add_argument("--subgoal_ckpt", default="/export/home/kaziz/g05/results/subgoal_gen/gen.pt")
    parser.add_argument("--cfg_neg_schedule", default="none")
    parser.add_argument("--cfg_w", type=float, default=1.0)
    parser.add_argument("--frs_mode", default="none", choices=["none", "frs", "sde", "ref", "exact", "dsbc"])
    parser.add_argument("--frs_window", default="40:80")
    parser.add_argument("--frs_back", type=int, default=5)
    parser.add_argument("--sde_t0", type=float, default=0.6)
    parser.add_argument("--frs_targets", default="/export/home/kaziz/g05/results/frs/targets.json")
    parser.add_argument("--frs_until", type=float, default=0.0, help="stop the reference once within this many metres of the target (0 = off)")
    parser.add_argument("--frs_ts", type=float, default=1.0, help="frs: invert only up to t = ts (1 = to pure noise)")
    parser.add_argument("--guide_lam", type=float, default=0.0, help="reference guidance strength in the forward pass (0 = off)")
    parser.add_argument("--frs_cfg", default="none", choices=["none", "old", "blank"], help="CFG negative prompt inside the window")
    parser.add_argument("--frs_cfg_w", type=float, default=2.0)
    parser.add_argument("--frs_target_noise", type=float, default=0.0, help="std (m) of a fixed per-trial error on the target point")
    parser.add_argument("--frs_rec", default="", help="record window requests on the server under this tag prefix (REC_DIR)")
    parser.add_argument("--init_offset", type=int, default=0, help="trial k uses initial state k + offset (held-out states)")
    parser.add_argument("--prompt_prefix", default="", help="PrefPI: prefix of the preference-conditioned prompt, e.g. '[cfg] '")
    parser.add_argument("--cfg_plain_w", type=float, default=1.0, help="PrefPI: CFG weight of the prefixed prompt against the plain one")
    args, remaining = parser.parse_known_args()
    return args, [item for item in remaining if "=" in item]


def main():
    args, overrides = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    cfg = None
    run_dir = None
    embodiment_type = args.embodiment_type
    if args.ckpt_path is not None:
        run_dir = find_run_dir(args.ckpt_path)
        cfg = load_config_from_run_dir(run_dir, args.ckpt_path, overrides)

        if cfg.get("eval_embodiment") and "embodiment_datasets" in cfg.data:
            filter_embodiment(cfg, cfg.eval_embodiment)

        args.task_suite_name = _cfg_get(cfg, "libero.task_suite_name", args.task_suite_name)
        args.num_trials_per_task = _cfg_get(
            cfg,
            "libero.num_trials_per_task",
            _cfg_get(cfg, "EVALUATION.num_trials", args.num_trials_per_task),
        )
        args.num_steps_wait = _cfg_get(cfg, "libero.num_steps_wait", args.num_steps_wait)
        args.seed = args.seed if args.seed is not None else cfg.get("seed")
        if cfg.data.get("embodiment_datasets"):
            embodiment_type = cfg.data.embodiment_datasets.get(args.embodiment_type, {}).get(
                "embodiment_type", embodiment_type
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if run_dir is not None:
        logger.info("Resolved run dir: %s", run_dir)
    logger.info("Parallel envs: %d", args.num_parallel)

    summary = asyncio.run(evaluate(args, cfg, embodiment_type))
    summary_path = args.output_dir / f"{args.task_suite_name}_parallel_results.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=_json_default)

    logger.info(
        "Total success rate: %.3f (%d/%d)",
        summary["success_rate"],
        summary["total_successes"],
        summary["total_episodes"],
    )
    logger.info("Results saved to %s", summary_path)


if __name__ == "__main__":
    main()
