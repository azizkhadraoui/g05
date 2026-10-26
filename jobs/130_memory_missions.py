"""Memory as a household robot needs it: interrupt-and-resume and standing instructions, on the LIBERO-Goal kitchen with the
full stack (planner VLM + frozen G0.5), with three kinds of planner memory (--arm):

  event       the conversation so far + the robot's own activity log: the plan it made, which skill it worked on and for how
              long, interruptions, rest-pose moves, and a camera check by the VLM of whether each skill looks finished.
              Nothing in the log comes from the simulator's success check.
  transcript  the conversation so far (every user utterance), no activity log: a long-context VLM with no robot memory
  view        the current camera image and the latest utterance only (no memory)
  rules       event memory + rule memory: when a request arrives the planner names the skills it forbids ("don't open any
              drawers today"); those skills are logged as a rule and removed from the skill list of every later decision

Episodes (every arm: same scenes, same start states, same utterances; all skills are executed by G0.5, so visible progress
is available to every arm):
  interrupt  "First A, then B, and after that C." -> A runs; B runs for --interrupt_at steps; "Wait, X first." -> X runs;
             the arm parks; "OK, carry on with what you were doing." -> up to 3 planner decisions, each executed.
             Scored: the first decision after "carry on" is the first unfinished planned skill (B, or C if B finished before
             the interruption); no finished skill is redone; A, X, B and C all finished at the end.
  standing   "<standing instruction> Now, first T1, then T2." -> T1, T2 run; the arm parks; a later request that the
             standing instruction constrains ("Put the bowl away somewhere." after "keep the stove clear today").
             Scored: the decision violates the instruction / is an allowed way to fulfil the request.
The simulator's goal checks end a skill (held 5 steps) and score the episode; they are never shown to the planner.
Output: results/memory_real/<arm>/episodes.jsonl (+ summary.json)
"""
import argparse, asyncio, json, os, re, sys
from pathlib import Path
import numpy as np

G = os.environ.get("G05_ROOT", "/export/home/kaziz/g05"); JOBS = f"{G}/jobs"
sys.path.insert(0, JOBS); sys.path.insert(0, f"{G}/GalaxeaVLA"); os.chdir(f"{G}/GalaxeaVLA")
import t4_fp16  # noqa: E402,F401
from experiments.libero.libero_eval_utils import (  # noqa: E402
    LIBERO_DUMMY_ACTION, LiberoGripperCommandState, build_libero_raw_obs, ensure_libero_config, extract_libero_images,
    single_action_to_libero_action)
from experiments.libero.libero_vector_env import LiberoAsyncVectorEnv  # noqa: E402
from scripts.utils.policy_ws_client import MultiPolicyWSClient  # noqa: E402

# LIBERO-Goal task indices: 0 open middle drawer, 1 bowl on stove, 2 wine bottle on cabinet, 3 top drawer + bowl inside,
# 4 bowl on cabinet, 5 push plate, 6 cream cheese in bowl, 7 turn on stove, 8 bowl on plate, 9 wine bottle on rack
INTERRUPT = [(2, 0, 7, 8), (5, 1, 9, 0), (7, 9, 0, 6), (6, 2, 5, 7), (0, 8, 2, 7), (9, 7, 6, 5), (4, 7, 0, 9), (3, 9, 7, 5)]  # A, B, X, C
STANDING = [  # (instruction, later request, allowed skills, forbidden skills, distractor pairs)
    ("Please keep the stove clear today: don't put anything on it.", "Put the bowl away somewhere.", {3, 4, 8}, {1}, [(2, 0), (9, 5)]),
    ("Don't touch the wine bottle today, it's a gift.", "Put something on top of the cabinet.", {4}, {2, 9}, [(0, 7), (5, 8)]),
    ("Please don't open any of the drawers today.", "Put the bowl away somewhere.", {1, 4, 8}, {0, 3}, [(9, 7), (2, 5)]),
]

p = argparse.ArgumentParser()
p.add_argument("--server_uri", required=True)
p.add_argument("--arm", default="event", choices=["event", "rules", "transcript", "view"])
p.add_argument("--inits", type=int, default=5)
p.add_argument("--kinds", default="interrupt,standing")
p.add_argument("--num_parallel", type=int, default=3)
p.add_argument("--vlm", default=f"{G}/models/RoboBrain2.5-4B")
p.add_argument("--vlm_device", default="cuda:1")
p.add_argument("--budget", type=int, default=300, help="policy steps per skill")
p.add_argument("--interrupt_at", type=int, default=30)
p.add_argument("--output_dir", default=f"{G}/results/memory_real")
p.add_argument("--tag", default="")
a = p.parse_args()
OUT = f"{a.output_dir}/{a.tag or a.arm}"; os.makedirs(OUT, exist_ok=True)
EXEC, HZ = 10, 20.0
EVENT = a.arm in ("event", "rules")  # rules = event memory + standing instructions stored as rules that remove forbidden skills


def make_env_fn(bddl, goal_bddls, seed):
    def _fn():
        from libero.libero.envs import OffScreenRenderEnv
        import libero.libero.envs.bddl_utils as BDDLUtils
        env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256, horizon=6000); env.seed(seed)  # missions exceed 1000 steps
        P_ = env.env; goals = [BDDLUtils.robosuite_parse_problem(g)["goal_state"] for g in goal_bddls]; _orig = env.step

        def _step(act):
            obs, r, done, info = _orig(act); info = dict(info or {})
            info["goal_sat"] = [bool(all(P_._eval_predicate(s) for s in gs)) for gs in goals]
            return obs, r, False, info
        env.step = _step
        return env
    return _fn


async def main():
    ensure_libero_config()
    from libero.libero import benchmark, get_libero_path
    from robosuite.utils import transform_utils as TU
    from robobrain import RoboBrain
    suite = benchmark.get_benchmark_dict()["libero_goal"]()
    langs = [suite.get_task(i).language for i in range(suite.n_tasks)]
    bddls = [str(Path(get_libero_path("bddl_files")) / suite.get_task(i).problem_folder / suite.get_task(i).bddl_file) for i in range(suite.n_tasks)]
    rb = RoboBrain(a.vlm, device=a.vlm_device)
    skills = "\n".join(f"{k + 1}. {c}" for k, c in enumerate(langs))
    img = lambda k: extract_libero_images(obs[k])["image"]  # noqa: E731

    def ints(ans):
        return [int(x) for x in re.findall(r"\d+", ans)]

    def parse_plan(k, utt, n):
        q = (f"You control a robot arm in a kitchen. The user says: \"{utt}\"\nThe robot can run these skills:\n{skills}\n"
             f"Which skills carry out the request, in order? Answer with exactly {n} skill numbers separated by commas.")
        ans = rb.ask([([img(k)], q)], max_new_tokens=16)[0]
        return [x - 1 for x in ints(ans) if 1 <= x <= len(langs)][:n], ans

    def context(conv, log, utt):
        if a.arm == "view":
            return f"The user says: \"{utt}\""
        s = "Conversation so far (oldest first):\n" + "\n".join(f"- User: \"{u}\"" for u in conv)
        if EVENT:
            s += "\nRobot activity log (oldest first):\n" + "\n".join(f"- {line}" for line in log)
        return s + f"\nThe user now says: \"{utt}\""

    def decide(k, conv, log, utt, rules=()):  # -> (skill index or None for "nothing left to do", answer)
        menu = [i for i in range(len(langs)) if i not in rules]  # rule memory: forbidden skills are not offered
        q = (f"You control a robot arm in a kitchen.\n{context(conv, log, utt)}\nThe robot can run these skills:\n"
             + "\n".join(f"{n + 1}. {langs[i]}" for n, i in enumerate(menu))
             + "\nWhich single skill should the robot do now? If nothing is left to do, answer 0. Answer with one skill number.")
        ans = rb.ask([([img(k)], q)], max_new_tokens=8)[0]
        xs = ints(ans)
        return (menu[xs[0] - 1] if xs and 1 <= xs[0] <= len(menu) else None), ans

    def extract_rules(k, utt, log):  # rules arm: skills the user forbids, stored when the request is heard
        if a.arm != "rules":
            return set(), None
        q = (f"You control a robot arm in a kitchen. The user says: \"{utt}\"\nThe robot can run these skills:\n{skills}\n"
             "Does the user forbid any of these skills for today? Answer with the numbers of the forbidden skills separated by "
             "commas, or 0 if none.")
        ans = rb.ask([([img(k)], q)], max_new_tokens=24)[0]
        rules = {x - 1 for x in ints(ans) if 1 <= x <= len(langs)}
        if rules:
            log.append("Rule from the user: do not " + "; do not ".join(langs[i] for i in sorted(rules)) + ".")
        return rules, ans

    def note(k, log, skill, res, interrupted=False):  # the robot's own record of a skill (event arm): no simulator signal
        if not EVENT:
            return
        q = f"Look at the kitchen. Has the robot arm finished this task: \"{langs[skill]}\"? Answer yes or no."
        ans = rb.ask([([img(k)], q)], max_new_tokens=4)[0].strip().lower()
        chk = "finished" if ans.startswith("yes") else "not finished"
        why = "stopped because the user interrupted" if interrupted else "then stopped"
        log.append(f"Worked on \"{langs[skill]}\" for {res['steps'] / HZ:.0f} s, {why}; camera check: {chk}.")

    def ep_interrupt(k, mi):
        A, B, X, C = INTERRUPT[mi]; conv, log, done, decisions = [], [], [], []
        u0 = f"First {langs[A]}, then {langs[B]}, and after that {langs[C]}."
        plan, pans = parse_plan(k, u0, 3); conv.append(u0)
        if EVENT:
            log.append("Made a plan from the request: " + "; ".join(f"{i + 1}) {langs[s]}" for i, s in enumerate(plan)) + ".")
        rules, rans = extract_rules(k, u0, log)
        rec = dict(kind="interrupt", mission=mi, plan=[langs[s] for s in plan], plan_answer=pans, plan_ok=plan == [A, B, C], runs=[],
                   rules=[langs[i] for i in sorted(rules)], rules_answer=rans)

        def run_note(res, skill, interrupted=False):
            rec["runs"].append(dict(skill=langs[skill], steps=res["steps"], finished=res["completed"], interrupted=interrupted))
            if res["completed"]:
                done.append(skill)
        for i, (s, lim) in enumerate([(plan[0] if plan else None, a.budget), (plan[1] if len(plan) > 1 else None, a.interrupt_at)]):
            if s is None:
                continue
            res = yield ("run", s, lim); run_note(res, s, interrupted=(i == 1 and not res["completed"]))
            note(k, log, s, res, interrupted=(i == 1 and not res["completed"]))
        u1 = f"Wait, {langs[X]} first."
        x_choice, x_ans = decide(k, conv, log, u1, rules); conv.append(u1)
        rec["interrupt_decision"] = dict(answer=x_ans, choice=None if x_choice is None else langs[x_choice], correct=x_choice == X)
        if x_choice is not None:
            res = yield ("run", x_choice, a.budget); run_note(res, x_choice); note(k, log, x_choice, res)
        yield ("home",)
        if EVENT:
            log.append("Moved the arm back to its rest position and waited.")
        u2 = "OK, carry on with what you were doing."
        expected = next((s for s in (B, C) if s not in done), None)
        rec["resume_expected"] = None if expected is None else langs[expected]
        conv_now = list(conv)
        for _ in range(3):
            done_before = list(done)
            ch, ans = decide(k, conv_now, log, u2, rules)
            want = next((s for s in (B, C) if s not in done_before), None)  # first unfinished planned skill (scoring only)
            decisions.append(dict(answer=ans, choice=None if ch is None else langs[ch], redo=ch is not None and ch in done_before,
                                  correct=ch == want))
            if ch is None:
                break
            res = yield ("run", ch, a.budget); run_note(res, ch); note(k, log, ch, res)
        conv.append(u2)
        rec["resume_decisions"] = decisions
        first = decisions[0]["choice"] if decisions else None
        rec["resume_ok"] = first == (None if expected is None else langs[expected])
        rec["redo"] = any(d["redo"] for d in decisions)
        rec["all_done"] = all(s in done for s in (A, B, C, X))
        rec["stopped_ok"] = rec["all_done"] and decisions[-1]["choice"] is None
        return rec

    def ep_standing(k, si, pi):
        instr, req, allowed, forbidden, pairs = STANDING[si]; T1, T2 = pairs[pi]; conv, log = [], []
        u0 = f"{instr} Now, first {langs[T1]}, then {langs[T2]}."
        plan, pans = parse_plan(k, u0, 2); conv.append(u0)
        if EVENT:
            log.append("Made a plan from the request: " + "; ".join(f"{i + 1}) {langs[s]}" for i, s in enumerate(plan)) + ".")
        rules, rans = extract_rules(k, u0, log)
        rec = dict(kind="standing", scenario=si, pair=pi, plan=[langs[s] for s in plan], plan_answer=pans, plan_ok=plan == [T1, T2], runs=[],
                   rules=[langs[i] for i in sorted(rules)], rules_answer=rans)
        for s in plan:
            res = yield ("run", s, a.budget); rec["runs"].append(dict(skill=langs[s], steps=res["steps"], finished=res["completed"]))
            note(k, log, s, res)
        yield ("home",)
        if EVENT:
            log.append("Moved the arm back to its rest position and waited.")
        ch, ans = decide(k, conv, log, req, rules)
        rec.update(request=req, answer=ans, choice=None if ch is None else langs[ch], violation=ch in forbidden, allowed=ch in allowed)
        if ch is not None:
            res = yield ("run", ch, a.budget); rec["runs"].append(dict(skill=langs[ch], steps=res["steps"], finished=res["completed"]))
        return rec

    kinds = a.kinds.split(",")
    eps = ([("interrupt", mi, 0, j) for mi in range(len(INTERRUPT)) for j in range(a.inits)] if "interrupt" in kinds else []) + \
          ([("standing", si, pi, j) for si in range(len(STANDING)) for pi in range(2) for j in range(a.inits)] if "standing" in kinds else [])
    if os.path.exists(f"{OUT}/episodes.jsonl"):  # resume
        seen = {tuple(e["key"]) for e in map(json.loads, open(f"{OUT}/episodes.jsonl"))}
        eps = [e for e in eps if tuple(e) not in seen]
    P = a.num_parallel
    vec = await asyncio.to_thread(LiberoAsyncVectorEnv, env_fns=[make_env_fn(bddls[1], bddls, 7)] * P, dummy_env_fn=make_env_fn(bddls[1], bddls, 7))
    fout = open(f"{OUT}/episodes.jsonl", "a")
    async with MultiPolicyWSClient(a.server_uri, P, timeout_s=600) as client:
        for b0 in range(0, len(eps), P):
            batch = eps[b0:b0 + P]; n = len(batch)
            first = [INTERRUPT[e[1]][0] if e[0] == "interrupt" else STANDING[e[1]][4][e[2]][0] for e in batch]
            inits = [suite.get_task_init_states(t)[(7 + 11 * e[3]) % 50] for t, e in zip(first, batch)]; inits += [inits[0]] * (P - n)
            obs = list(await asyncio.to_thread(vec.set_init_state_each, inits))
            act = [k < n for k in range(P)]
            for _ in range(20):
                ob, _, _, _ = await asyncio.to_thread(vec.step_selective, act, [LIBERO_DUMMY_ACTION] * P)
                obs = [x if x is not None else o for x, o in zip(ob, obs)]
            gst = [LiberoGripperCommandState.from_obs(o) for o in obs]
            rest = [(np.asarray(o["robot0_eef_pos"], np.float64).copy(), np.asarray(o["robot0_eef_quat"], np.float64).copy()) for o in obs]
            gens = [ep_interrupt(k, e[1]) if e[0] == "interrupt" else ep_standing(k, e[1], e[2]) for k, e in enumerate(batch)]
            seg, recs, run = [None] * P, [None] * P, [None] * P

            def advance(k, result):
                try:
                    seg[k] = gens[k].send(result) if result is not None else next(gens[k])
                except StopIteration as stop:
                    seg[k] = None; recs[k] = stop.value
                if seg[k] is not None and seg[k][0] == "run":
                    run[k] = dict(steps=0, held=0)
            for k in range(n):
                advance(k, None)

            async def go_home(js):  # scripted retract to the rest pose (as 125_e2e_missions.py)
                wps = {}
                for k in js:
                    c = np.asarray(obs[k]["robot0_eef_pos"], np.float64); h = rest[k][0]; zs = max(c[2] + 0.05, h[2])
                    wps[k] = [np.array([c[0], c[1], zs]), np.array([h[0], h[1], zs]), h.copy()]
                for _ in range(140):
                    mask = [False] * P; acts = [LIBERO_DUMMY_ACTION] * P
                    for k in js:
                        if not wps[k]:
                            continue
                        c = np.asarray(obs[k]["robot0_eef_pos"], np.float64); e = wps[k][0] - c
                        while np.linalg.norm(e) < 0.015 and len(wps[k]) > 1:
                            wps[k].pop(0); e = wps[k][0] - c
                        er = TU.quat2axisangle(TU.mat2quat(TU.quat2mat(rest[k][1]) @ TU.quat2mat(np.asarray(obs[k]["robot0_eef_quat"])).T))
                        if len(wps[k]) == 1 and np.linalg.norm(e) < 0.01 and np.linalg.norm(er) < 0.05:
                            wps[k] = []; continue
                        acts[k] = np.concatenate([np.clip(e / 0.05, -1, 1), np.clip(er / 0.5, -1, 1), [-1.0]]).astype(np.float32).tolist(); mask[k] = True
                    if not any(mask):
                        break
                    ob, _, _, _ = await asyncio.to_thread(vec.step_selective, mask, acts)
                    for k in range(P):
                        if mask[k]:
                            obs[k] = ob[k]
                for k in js:
                    gst[k] = LiberoGripperCommandState.from_obs(obs[k])

            while any(s is not None for s in seg):
                homes = [k for k in range(n) if seg[k] is not None and seg[k][0] == "home"]
                if homes:
                    await go_home(homes)
                    for k in homes:
                        advance(k, {})
                    continue
                live = [k for k in range(n) if seg[k] is not None]
                raws = [build_libero_raw_obs(obs=obs[k], task_description=langs[seg[k][1]], embodiment_type="libero")[0] for k in live]
                await client.infer_batch(live, raws)
                chunks = {k: client.action_chunks[k] for k in live}
                for s in range(EXEC):
                    mask = [k in live for k in range(P)]; acts = [LIBERO_DUMMY_ACTION] * P
                    for k in live:
                        single = {kk: (np.asarray(v)[min(s, np.asarray(v).shape[0] - 1)] if np.asarray(v).ndim >= 1 else v) for kk, v in chunks[k].items()}
                        acts[k] = single_action_to_libero_action(single, obs=obs[k], gripper_state=gst[k])
                    ob, _, _, infos = await asyncio.to_thread(vec.step_selective, mask, acts)
                    ended = []
                    for k in live:
                        obs[k] = ob[k]; r = run[k]; r["steps"] += 1
                        r["held"] = r["held"] + 1 if infos[k]["goal_sat"][seg[k][1]] else 0
                        if r["held"] >= 5 or r["steps"] >= seg[k][2]:
                            ended.append((k, dict(completed=r["held"] >= 5, steps=r["steps"])))
                    for k, res in ended:
                        advance(k, res)
                    if ended:
                        break
            for k in range(n):
                r = recs[k] or {}; r.update(key=list(batch[k]), arm=a.tag or a.arm, init=batch[k][3])
                fout.write(json.dumps(r) + "\n"); fout.flush()
                print(json.dumps({kk: r.get(kk) for kk in ("key", "kind", "plan_ok", "resume_ok", "redo", "all_done", "violation", "allowed")}), flush=True)
    await asyncio.to_thread(vec.close)
    E = [json.loads(line) for line in open(f"{OUT}/episodes.jsonl")]
    I = [e for e in E if e.get("kind") == "interrupt"]; S = [e for e in E if e.get("kind") == "standing"]
    rate = lambda xs, f: (sum(bool(f(x)) for x in xs), len(xs))  # noqa: E731
    summ = {"arm": a.tag or a.arm, "interrupt": {k: rate(I, f) for k, f in (("plan_ok", lambda e: e["plan_ok"]), ("x_ok", lambda e: e["interrupt_decision"]["correct"]),
                                                                           ("resume_ok", lambda e: e["resume_ok"]), ("redo", lambda e: e["redo"]),
                                                                           ("all_done", lambda e: e["all_done"]), ("stopped_ok", lambda e: e["stopped_ok"]))},
            "standing": {k: rate(S, f) for k, f in (("plan_ok", lambda e: e["plan_ok"]), ("violation", lambda e: e["violation"]), ("allowed", lambda e: e["allowed"]))}}
    json.dump(summ, open(f"{OUT}/summary.json", "w"), indent=1)
    print(json.dumps(summ), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
