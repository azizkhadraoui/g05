from __future__ import annotations
"""g05 steering/memory client: a copy of GalaxeaVLA's experiments/libero/eval_libero_parallel.py with two additions
(protocol S02 and M02-style probes; GalaxeaVLA itself is not modified):
  --init_task_id N        initial states from task N of the same suite (LIBERO-Goal: identical scene, different command)
  --prompt_schedule SPEC  instruction timeline over policy action steps, e.g. "0:target,40:blank" or "0:task3,40:target".
                          Items: target | task<i> (language of task i in the suite) | blank.
Success is always the env's own check for --task_id (the commanded target).
"""

import argparse
import asyncio
import re
import json
import logging
import time
from pathlib import Path
from typing import Any
import os
import sys
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
        return _lmem_patch(env)

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


LM_SUB: dict = {}  # LIBERO-Mem: env index -> max satisfied subgoals in the running episode
LM_GOAL = [None]
TRIAL_LMSUB: dict = {}
LM_DONE: dict = {}  # env index -> benchmark done at the latest step
TRIAL_LMDONE: dict = {}
HALT_CFG = None  # memory module: {"mode", "n", "dec", "cond", "lam", "H"}
HBUF: dict = {}; HSTEP: dict = {}; HALT: dict = {}; HCNT: dict = {}; HPRED: dict = {}
TRIAL_HALT: dict = {}


def _halt_load(mode, dec_path, n_target):
    import torch as _t
    sys.path.insert(0, "/export/home/kaziz/g05/jobs")
    import lawam_hybrid as _H
    from mem_decoder import Decoder as _D
    cfg = {"mode": mode, "n": n_target}
    if mode in ("vlm", "vlm_current"):  # RoboBrain counts completed cycles from frames (second GPU)
        from robobrain import RoboBrain
        cfg.update(vlm=RoboBrain(VLM_ARGS["path"], device=VLM_ARGS["device"]), obj=VLM_ARGS["obj"], every=VLM_ARGS["every"])
        return cfg
    if mode in ("memory", "current"):
        ck = _t.load(dec_path, map_location="cpu", weights_only=False)
        d = _D(ck["n_frames"], 9); d.load_state_dict(ck["model"]); d.eval().cuda()
        lam = _H.load_lawm(device="cpu")  # only the DINOv3 encoder goes to the GPU (shares it with the policy server)
        lam.vision_encoder.to("cuda")
        cfg.update(dec=d, cond=ck["cond"], lam=lam, H=_H)
    return cfg


VLM_ARGS: dict = {}


def _halt_update_vlm(obs_list, active):
    """1 Hz: RoboBrain answers 'how many cycles so far?' from 8 frames over the episode + now (or now only); halt at N."""
    from robobrain import count_prompt, first_int
    q = []
    for i, ob in enumerate(obs_list):
        if ob is None or not active[i] or HALT.get(i):
            continue
        HSTEP[i] = HSTEP.get(i, 0) + 1
        if (HSTEP[i] - 1) % 4 == 0:
            HBUF.setdefault(i, []).append(extract_libero_images(ob)["image"])
        if HSTEP[i] > 20 and HSTEP[i] % HALT_CFG["every"] == 0 and HBUF.get(i):
            q.append(i)
    if not q:
        return
    multi = HALT_CFG["mode"] == "vlm"
    batch = []
    for i in q:
        buf = HBUF[i]; t = len(buf) - 1
        ims = [buf[(t * j) // 8] for j in range(8)] + [buf[t]] if multi else [buf[t]]
        batch.append((ims, count_prompt(HALT_CFG["obj"], len(ims), "multi" if multi else "current")))
    for i, ans in zip(q, HALT_CFG["vlm"].ask(batch, max_new_tokens=6)):
        pred = first_int(ans, 0, 50)
        HPRED.setdefault(i, []).append(-1 if pred is None else pred)
        if pred is not None and pred >= HALT_CFG["n"]:
            HALT[i] = True; TRIAL_HALT[i] = HSTEP[i]


def _halt_frames(buf):
    import torch as _t
    c = HALT_CFG["cond"]
    t = len(buf) - 1
    if c == "current":
        idx = [t]
    elif c == "uniform8":
        idx = [(t * j) // 8 for j in range(8)] + [t]
    elif c == "offsets":
        idx = [max(t - k, 0) for k in (40, 20, 10, 5)] + [t]
    else:
        raise ValueError(c)
    return _t.stack([buf[k] for k in idx]).float()[None]


def _halt_update(obs_list, active):
    """5 Hz: append features, predict completed cycles, halt env once the count reaches N (2 consecutive frames)."""
    import torch as _t
    import torch.nn.functional as _F
    if HALT_CFG is None or HALT_CFG["mode"] == "none":
        return
    if HALT_CFG["mode"] in ("vlm", "vlm_current"):
        return _halt_update_vlm(obs_list, active)
    newf = []
    for i, ob in enumerate(obs_list):
        if ob is None or not active[i] or HALT.get(i):
            continue
        HSTEP[i] = HSTEP.get(i, 0) + 1
        if HALT_CFG["mode"] == "oracle":
            continue
        if (HSTEP[i] - 1) % 4 == 0:
            newf.append(i)
    if HALT_CFG["mode"] == "oracle":
        for i in list(HSTEP):
            if LM_SUB.get(i, 0) >= HALT_CFG["n"] and not HALT.get(i):
                HALT[i] = True; TRIAL_HALT[i] = HSTEP[i]
        return
    if not newf:
        return
    x = _t.from_numpy(np.stack([extract_libero_images(obs_list[i])["image"] for i in newf])).permute(0, 3, 1, 2).cuda()
    with _t.no_grad():
        f = HALT_CFG["H"].dino_from_uint8(HALT_CFG["lam"], x)
        b = f.shape[0]
        f = _F.avg_pool2d(f.view(b, 16, 16, 768).permute(0, 3, 1, 2), 2).permute(0, 2, 3, 1).reshape(b, 64, 768).half()
    for j, i in enumerate(newf):
        HBUF.setdefault(i, []).append(f[j])
        with _t.no_grad(), _t.autocast("cuda", dtype=_t.float16):
            pred = int(HALT_CFG["dec"](_halt_frames(HBUF[i])).float().argmax(-1).item())
        HPRED.setdefault(i, []).append(pred)
        HCNT[i] = HCNT.get(i, 0) + 1 if pred >= HALT_CFG["n"] else 0
        if HCNT[i] >= 2:
            HALT[i] = True; TRIAL_HALT[i] = HSTEP[i]


def _halt_reset(i):
    for d in (HBUF, HSTEP, HALT, HCNT, HPRED, LM_DONE, TRIAL_HALT):
        d.pop(i, None)


def _lmem_patch(env):
    """LIBERO-Mem's subgoal-aware success: advance its sequence counter every step (as in its mem_5/mem_6 scripts),
    reset it at every new initial state, and report progress in info."""
    P = env.env
    if not hasattr(P, "_satisfied_subgoals"):
        return env
    gs = P.parsed_problem["goal_state"]
    goal_len = P.get_goal_sequence_len() if hasattr(P, "get_goal_sequence_len") else len(gs[0]) - 1
    _orig_step, _orig_set = env.step, env.set_init_state

    def _reset():
        P.reset_subgoal_progress()
        if hasattr(P, "_overshot"):
            P._overshot = False

    def _step(a):
        P._check_success(inc=True)
        obs, r, done, info = _orig_step(a)
        info = dict(info or {})
        info["lm_sub"] = len(P._satisfied_subgoals)
        info["lm_goal"] = goal_len
        info["lm_done"] = bool(done)
        if os.environ.get("LMEM_FULL") == "1":
            done = False  # run to the step budget; success = benchmark done at the final step (overshoot fails)
        return obs, r, done, info

    def _set(s):
        out = _orig_set(s)
        _reset()
        return out

    env.step, env.set_init_state = _step, _set
    _reset()
    return env


RECORD_DIR = None  # set from --record_dir
RECORD_FROM_STEP = 0


def _rec_state(obs):
    import numpy as _np
    return _np.concatenate([_np.asarray(obs["robot0_eef_pos"], _np.float32), quat2axisangle(obs["robot0_eef_quat"]).astype(_np.float32),
                            _np.asarray(obs["robot0_gripper_qpos"], _np.float32).reshape(-1)[:2]]).astype(_np.float32)


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


def make_prompt_fn(spec: str, target: str, suite) -> Any:
    segs = []
    for item in spec.split(","):
        k, src = item.split(":")
        if src == "target":
            text = target
        elif src == "blank":
            text = ""
        elif src == "memory":
            text = MEMORY_TOKEN
        elif src.startswith("task"):
            text = suite.get_task(int(src[4:])).language
        else:
            raise ValueError(src)
        segs.append((int(k), text))
    segs.sort()

    def fn(step: int) -> str:
        text = segs[0][1]
        for k, t in segs:
            if step >= k:
                text = t
        return text

    return fn


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
        TRIAL_LMSUB[tid] = LM_SUB.pop(env_idx, 0)
        TRIAL_LMDONE[tid] = {"final_done": bool(LM_DONE.get(env_idx, False)), "halt_step": TRIAL_HALT.get(env_idx),
                             "pred_counts": HPRED.get(env_idx, [])[::5]}
        _halt_reset(env_idx)
        if env_minf[env_idx] is not None:
            trial_inferred[tid] = env_minf[env_idx]
        env_mbuf[env_idx], env_msnap[env_idx], env_minf[env_idx] = [], [], None
        if RECORD_DIR is not None and success and env_rec[env_idx]:
            import numpy as _np
            recs = [r for r in env_rec[env_idx] if r[0] >= RECORD_FROM_STEP]
            if len(recs) >= 8:
                Path(RECORD_DIR).mkdir(parents=True, exist_ok=True)
                _np.savez_compressed(
                    f"{RECORD_DIR}/task{task_id}_trial{tid}.npz",
                    step=_np.array([r[0] for r in recs]), image=_np.stack([r[1] for r in recs]),
                    wrist_image=_np.stack([r[2] for r in recs]), state=_np.stack([r[3] for r in recs]),
                    action=_np.stack([r[4] for r in recs]), prompt=_np.array([r[5] for r in recs]),
                    task_description=task_description)
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
        state_idx = trial_id % len(initial_states)
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
                                           _rec_action(actions[i]), prompt_fn(env_action_step[i]) if prompt_fn else task_description))
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
        for i in recompute_envs:
            _txt = prompt_fn(env_action_step[i]) if prompt_fn else task_description
            if _txt == MEMORY_TOKEN:
                if env_minf[i] is None:  # retrieve once, when the instruction is lost
                    _cur = _mem_feats([extract_libero_images(per_env_obs[i])["image"]])[0]
                    env_minf[i] = _mem_infer(env_mbuf[i], env_msnap[i], _cur)
                    logger.info("env %d trial %d memory inferred: %r (target %r)", i, env_trial_id[i], env_minf[i], task_description)
                _txt = env_minf[i]
            raw_obs, _ = build_libero_raw_obs(
                obs=per_env_obs[i],
                task_description=_txt,
                embodiment_type=embodiment_type,
            )
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
                                           _rec_action(actions[i]), prompt_fn(env_action_step[i]) if prompt_fn else task_description))
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
        "trial_subgoals": {int(k): v for k, v in sorted(TRIAL_LMSUB.items())},
        "trial_final": {int(k): v for k, v in sorted(TRIAL_LMDONE.items())},
        "goal_len": LM_GOAL[0],
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
    global RECORD_DIR, RECORD_FROM_STEP, MEM
    RECORD_DIR, RECORD_FROM_STEP = args.record_dir, args.record_from
    global HALT_CFG
    if args.memory_decoder:
        MEM = _mem_load(args.memory_decoder, args.memory_tau)
    if args.halt_mode != "none":
        VLM_ARGS.update(path=args.count_vlm, device=args.vlm_device, obj=args.count_object, every=args.vlm_every)
        HALT_CFG = _halt_load(args.halt_mode, args.count_decoder, args.n_target)
    ensure_libero_import()
    ensure_libero_config()
    from libero.libero import benchmark

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    max_steps = args.max_steps if args.max_steps else get_max_steps(args.task_suite_name)

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
            initial_states = task_suite.get_task_init_states(init_tid)[args.init_offset:]
            task_description = re.sub(r"^\d+\s+", "", task.language)  # LIBERO-Mem names carry a task number
            LM_SUB.clear(); TRIAL_LMSUB.clear(); TRIAL_LMDONE.clear()
            for _i in range(args.num_parallel):
                _halt_reset(_i)
            if args.prompt_text:
                task_description = args.prompt_text
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
            _orig_ss = vec_env.step_selective

            def _ss(mask, acts, _orig_ss=_orig_ss):
                acts = [LIBERO_DUMMY_ACTION if (HALT.get(_i) and mask[_i]) else _a for _i, _a in enumerate(acts)]
                out = _orig_ss(mask, acts)
                for _i, _inf in enumerate(out[3]):
                    if "lm_sub" in _inf:
                        LM_SUB[_i] = max(LM_SUB.get(_i, 0), _inf["lm_sub"])
                        LM_GOAL[0] = _inf["lm_goal"]
                        LM_DONE[_i] = _inf["lm_done"]
                _halt_update(out[0], mask)
                return out

            vec_env.step_selective = _ss

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
                )
                task_result["init_task_id"] = init_tid
                task_result["prompt_schedule"] = args.prompt_schedule
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
    parser.add_argument("--max_steps", type=int, default=None, help="override the per-suite step budget (LIBERO-Mem)")
    parser.add_argument("--init_offset", type=int, default=0, help="first initial state to use (LIBERO-Mem held-out: 100)")
    parser.add_argument("--prompt_text", default=None, help="instruction given to the policy instead of the task's own")
    parser.add_argument("--halt_mode", default="none", choices=["none", "memory", "current", "oracle", "vlm", "vlm_current"])
    parser.add_argument("--count_vlm", default=None, help="RoboBrain checkpoint dir (halt_mode vlm / vlm_current)")
    parser.add_argument("--count_object", default="bowl", choices=["bowl", "bottle"])
    parser.add_argument("--vlm_every", type=int, default=20, help="query RoboBrain every N env steps")
    parser.add_argument("--vlm_device", default="cuda:1")
    parser.add_argument("--count_decoder", default=None)
    parser.add_argument("--n_target", type=int, default=1)
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
