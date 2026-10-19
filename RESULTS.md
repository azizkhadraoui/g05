# G0.5 FM + LaWAM: experiments and results

*Status: 19 Oct 2026, morning. Simulation only (LIBERO). Shared GPU cluster: 4×H200 allowance, plus T4/V100s via an fp16 port of G0.5 serving and training.*
*Protocol: "G0.5 FM + LaWAM: H200 experiment protocol, v2". Groups not listed below were not run (see §9).*

---

## 0. Closed-loop headline results (G0.5 in the LIBERO simulator)

| Ability | Experiment | Without | **With** | Paired test |
|---|---|---|---|---|
| **World model** | Our 43M latent world model plans over frozen G0.5's action proposals, given a goal video and no instruction (LIBERO-Goal, §8g) | 2% (random command), 7% (no instruction) | **77%** (correct instruction: 98%) | p = 5e-23 |
| **Memory** | External memory module, instruction withdrawn at 2 s (LIBERO-Goal, MEM-2, §8d) | 40% | **86%** | p = 1e-10 |
| **Memory** | LIBERO-Mem repetition tasks, fine-tuned G0.5 + a memory module that counts cycles and stops at N (§8h) | 5.8% alone; 4.2% with a current-frame (Markovian) counter | **19.2%** (oracle counter: 41.7%) | p = 0.003 vs alone; p = 5e-4 vs Markovian |
| **Steering** | Switch-recovery fine-tune vs matched control (LIBERO-Goal, §7) | 20% after a mid-task switch | **28%** (released G0.5: 29–31%) | p = 0.008 |
| Steering via world model | Goal video switches at 2 s (§8g) | 32% when the command switches instead | 15% | worse, p = 5e-4 |

## 1. Summary

| Question | Result |
|---|---|
| Can a pretrained latent world model (LaWAM) predict LIBERO futures? | **Yes.** It predicts DINOv3 features 0.4 s ahead with 0.40× the error of persistence (W00). |
| Can a frozen G0.5 drive it causally? | **Yes.** A small prior on G0.5's VLM states recovers most of the oracle gain: 0.69× persistence vs 0.62× for the oracle (F00). |
| Does feeding the predicted future into G0.5's action expert help? | **No. The staged version hurts by 7.5 pts** (p = 0.012). The harm comes from the *content* of the prediction (F03). |
| Does joint training of world model + policy fix that? | **It removes the harm** (+6.7 to +7.5 pts vs staged, significant), returning to plain-G0.5 level. **No gain over G0.5**, which is at a 98% ceiling. |
| Is G0.5 steerable mid-task? | **No.** It follows the initial command (98%, even from another task's scene), but **only 29% after a mid-task command switch**. |
| Does any adapter (current-frame, foresight, joint) improve steering? | **No.** All adapters are worse than plain G0.5 at switching (21–25%) and at instruction withdrawal (24–32% vs 42%). |
| Does G0.5 need memory? | **Yes.** Withdrawing the instruction after 2 s drops success to 42%, and stale "memory" drops it to 20%. |
| Does visual episodic memory recover it *inside* G0.5 (memory adapters)? | **No.** Neither fixed-offset memory nor 3D-Mem-style snapshot memory reduces the offline loss. |
| **Does an external memory module that re-issues the task recover it (MEM-2, closed loop)?** | **Yes: 40% → 83–86%, +43 to +46 pts (p ≈ 1e-10).** That recovers ~75% of the 59 points lost. It does not help after a command revision, because G0.5 rarely executes the revised command. |
| Does memory carry task information the current frame lacks (MEM-1)? | **Yes, where the scene is ambiguous.** LIBERO-Spatial task-ID error falls from 2.9% to ~0% (3 seeds, every CI excludes 0). |
| Does memory help the world model (WM-MEM)? | **Barely.** 0.4% lower error. Given the actions, LIBERO dynamics are Markovian. |
| **Closed loop: can our 17M world model plan for G0.5?** | **Yes.** Given only a goal video (no instruction), world-model selection over G0.5's proposals reaches **77% success vs 2% (random command) and 7% (no instruction)**, p ≈ 1e-21. The correct instruction gives 98% (§8g). |
| **Standard memory benchmark (LIBERO-Mem): does G0.5 need memory, and does memory supply it?** | **Yes and yes.** Zero-shot G0.5 completes 19% of subgoals and 0/120 repeat/swap/rotate episodes. Cycle counting on the repetition tasks goes from **15% (current frame) to 72% (episodic memory), +57 pts [+51, +63]** (§8f). |
| Can a *small* JEPA-style world model of our own work? | **Yes, clearly.** 17M parameters: 0.41× / 0.32× / 0.28× persistence error at 0.4 / 0.8 / 1.6 s. Beats LaWAM's 538M model in-domain, is action-sensitive, and ranks the expert's action chunk at 46% vs 25% chance. |

**Main takeaways**
1. Small, in-domain, action-conditioned latent world models are cheap and work well, so planning and ranking in latent space is a promising direction.
2. Injecting a pretrained world model's predictions into a strong VLA as an extra input does not help. Done naively (staged) it hurts closed-loop control; done jointly it is neutral.
3. Steerability and memory are real weaknesses of G0.5, and extra conditioning does not fix them.
   - **Memory:** external memory modules do help in closed loop, by re-issuing the task or by counting and stopping.
   - **Steering:** recovery fine-tuning helps only modestly. It recovers what the adapter lost, but doesn't beat the released G0.5.
4. **Planning in a small world model works in closed loop** (77% from a goal video vs 2–7% without), provided it gets near-term subgoals. It can't re-steer G0.5 after a goal switch, because it can only choose among G0.5's own proposals.

---

## 2. Setup

- **Policy:** Galaxea G0.5, released LIBERO checkpoint `g05-libero`, flow-matching mode (Qwen3.5-2B VLM + 24-layer action expert, ~2.9B parameters). **Frozen** in every hybrid experiment.
- **World model:** LaWAM (arXiv 2606.15768). We use the LaWM components from `jialei02/lawam_pretrain`:
  - **D:** decoder predicting future DINOv3 ViT-B/16 features from (current features, 32-d latent action);
  - **Q:** inverse-dynamics encoder giving the teacher latent z\*;
  - **DINOv3**, frozen.
- **Hybrid** (protocol §1/§3):
  - **P:** latent prior, 10.5M parameters. Predicts z_hat from G0.5's causal VLM prefix states.
  - **u_hat = D(u_t, z_hat):** the predicted future features.
  - **A:** adapter, 7.4M parameters. Two Perceiver resamplers (16 tokens each) feed residual cross-attention into expert layers 2, 5, …, 23. The output projection is zero-initialised, so A starts as an exact no-op.
  - **F01** feeds the adapter (u_t, u_t); **F02** feeds (u_t, u_hat). Both are matched in parameters, seed, data order and budget (5,000 steps, batch 64, LR 1e-4 cosine, AdamW).
- **Data:** LaWAM's merged LIBERO set (LeRobot v3, 1,693 demos, 273k frames at 20 Hz), normalised with the *released checkpoint's* statistics.
- **Evaluation:**
  - LIBERO with G0.5's stock client: `action_steps=10`, 20 wait steps, 256 px images, env seed 7.
  - Paired comparisons share starting states and seeds. The paired analysis reports a task-macro bootstrap 95% CI and an exact McNemar test.

### Issues found and fixed (worth knowing for anyone reusing these assets)

| Issue | Effect | Fix |
|---|---|---|
| **The gripper action sign is inverted** in LaWAM's LIBERO dataset relative to G0.5's (stored as {0,1} with the opposite meaning) | Frozen G0.5 training loss of **2.44 vs 0.072** after the fix. The first F01 run silently learned to undo the sign, which would have looked like a big gain. | Converted dataset with `a6 → 1 − a6`; the first F01 run was discarded |
| MuJoCo EGL `read_pixels` aborts on some H200 nodes | Whole evaluation suites crashed | Evaluations exclude those nodes; per-task clients with retry |
| G0.5's eval client over-counts successes when `num_parallel > num_trials` | 46/20 "successes" in a smoke run | Enforce `num_parallel ≤ num_trials` |
| Gated DINOv3 / offline Hugging Face loading in LaWAM's config | B01 crashed on load | Point the LaWM config at the local DINOv3 copy |

---

## 3. Baselines

### B00: released G0.5 FM, full LIBERO (2,000 episodes, 50 trials × 40 tasks)

| Spatial | Object | Goal | Long | **Average** |
|---|---|---|---|---|
| 99.2% | 99.8% | 98.2% | 98.4% | **98.9%** (paper: 98.9%) |

### B01: released LaWAM LIBERO SFT, through LaWAM's own benchmark script

| Suite | Result | Paper |
|---|---|---|
| LIBERO-Long | **96.4%** (482/500) | 97.0% |
| Object, Goal, Spatial | *cancelled on 18 Oct to free H200s for the memory/world-model runs* | |

These are system baselines. Different backbones and data make them a system comparison, not an ablation.

### Screen tasks (pre-registered: the 6 lowest-B00 tasks)

The 6 tasks with the lowest B00 success, chosen before any hybrid result:
- `libero_goal:2` (0.92)
- `libero_10:1, 3, 6` (0.96 each)
- `libero_spatial:6` (0.96)
- `libero_10:8` (0.98)

Each screen runs these 6 tasks × 20 paired starting states.

---

## 4. World-model quality

### Pretraining audit
LaWAM policy pretraining changed **only the LaWM decoder** (relative L2 drift 0.26). The inverse-dynamics encoder, DINOv3, VQ and state decoder are bit-identical to `jialei02/lawam_lam`.

### W00: oracle decoding (512 random LIBERO frame pairs)

Ratio = predicted-future error ÷ persistence error (lower is better).

| | τ = 0.4 s | τ = 0.4 s, moving half | τ = 1.2 s |
|---|---|---|---|
| `lawam_lam` decoder | **0.40×** (beats persistence on 100% of pairs) | 0.35× | 0.27× |
| `lawam_pretrain` decoder | **0.59×** (94% of pairs) | 0.53× | |

Controls:
- A shuffled latent action makes predictions **1.4–3.5× worse**.
- Images flipped 180° make predictions worse (0.40 → 0.68), confirming the dataset orientation is right.
- The `lawam_pretrain` decoder looks worse here only because policy pretraining adapted it to VLM-predicted latents rather than oracle ones.

### F00: causal prior P (3,000 steps, evaluated on 2,048 frames from held-out episodes)

| | All frames | Moving half |
|---|---|---|
| Latent cosine to z\* | 0.93 | 0.94 |
| Predicted-future error ÷ persistence | **0.69×** | **0.63×** |
| Oracle (W00, true z\*) ÷ persistence | 0.62× | 0.54× |
| Beats persistence | 91% | 98% |
| Wrong latent ÷ predicted | 1.43× | 1.40× |

**Gate passed:** G0.5's frozen representation carries enough information to drive LaWAM's world model causally.

---

## 5. Does LaWAM foresight help G0.5? (F01, F02, F03, J1–J3)

### Training (final action / flow-matching loss)
- F01: 0.018
- F02: ~0.012
- J1: 0.0148
- J2: 0.0126
- J3: 0.0144

Lower training loss did **not** translate into closed-loop gains.

### Screen results (6 tasks × 20 paired trials)

| Policy | Success | vs G0.5: diff [95% CI], p |
|---|---|---|
| G0.5 (B00) | **118/120 (98.3%)** | — |
| F01: current-frame adapter | 116/120 (96.7%) | −1.7 [−5.0, +1.7], p = 0.63 |
| **F02: staged predicted-future adapter** | **109/120 (90.8%)** | **−7.5 [−12.5, −2.5], p = 0.012** |
| J1: joint (P + decoder + A), future detached | 117/120 (97.5%) | −0.8 [−2.5, 0.0], p = 1.0 |
| J2: joint end-to-end (action loss into P/decoder) | 118/120 (98.3%) | 0.0 [−2.5, +2.5], p = 1.0 |
| J3: F02 with future dropout p = 0.5 | 117/120 (97.5%) | −0.8 [−4.2, +2.5], p = 1.0 |

The same policies compared against F02:

| Comparison | Diff [95% CI] | p |
|---|---|---|
| F02 vs F01 | −5.8 [−10.8, −0.8] | 0.065 |
| **J1 vs F02** | **+6.7 [+1.7, +11.7]** | 0.021 |
| **J2 vs F02** | **+7.5 [+2.5, +12.5]** | 0.012 |
| **J3 vs F02** | **+6.7 [+1.7, +11.7]** | 0.039 |
| J1 / J2 / J3 vs F01 | +0.8 / +1.7 / +0.8 | all n.s. |

### F03: evaluation-time corruption of F02's future input

| Future input replaced by | Success | vs normal F02 |
|---|---|---|
| (normal predicted future) | 90.8% | — |
| the current frame | 96.7% | +5.8 [0.0, +11.7], p = 0.09 |
| another episode's prediction | **98.3%** | **+7.5 [+2.5, +12.5], p = 0.012** |
| masked (attention excludes the future tokens) | 0% | out-of-distribution input; this shows the expert depends heavily on the adapter, not that the future is used |

**Interpretation**
- The staged adapter **over-trusts the correctly-timed prediction**. In closed loop the prediction drifts away from the demonstration distribution and misleads the actions. Replacing it with uninformative inputs restores performance.
- Joint training, or robustness training (J3), removes the harm but does not add a measurable benefit on LIBERO, which is at ceiling (98%).

---

## 6. Steering and memory probes (S02 / M02-style, LIBERO-Goal)

LIBERO-Goal's 10 tasks share **one identical scene** and differ only in the instruction.
- **Pairs:** target task t, with source task s = (t+3) mod 10.
- **Size:** 10 paired starting states per pair, so 100 episodes per cell.
- **Switch timing:** switches happen after 40 action steps (2 s).

| Condition | G0.5 | F01 | F02 | J1 | J2 | J3 |
|---|---|---|---|---|---|---|
| **ref**: own scene, own command | 98 | 97 | 97 | 97 | **99** | 97 |
| **S02-cf**: another task's scene, target command | 98 | 98 | 95 | 97 | 97 | **99** |
| **S02-switch**: source command for 2 s, then target | **29** | 21 | 21 | 21 | 25 | 22 |
| **M02-none**: instruction blank after 2 s | **42** | 25 | 26 | 24 | 32 | 30 |
| M02-stale: target for 2 s, then the source instruction | 20 | 19 | 20 | — | — | — |
| M02-revise-none: source → target → blank | 20 | 20 | 20 | — | — | — |
| M02-revise-stale: source → target → source | 16 | 15 | 16 | — | — | — |

Paired differences for G0.5 against its own reference. The CIs cover these 10 command pairs only.

| Comparison | Diff [95% CI] | p |
|---|---|---|
| S02-cf | 0.00 [−0.04, +0.04] | 1.0 |
| **S02-switch** | **−0.69 [−0.72, −0.66]** | 3e−21 |
| **M02-none** | **−0.56 [−0.61, −0.50]** | 3e−17 |
| **M02-stale** | **−0.78 [−0.80, −0.75]** | 7e−24 |
| revise-none vs switch | −0.09 | 0.004 |
| revise-stale vs switch | −0.13 | 2e−4 |

For G0.5, the per-pair successes after a switch were 0/10 for 7 of the 10 command pairs.

**Interpretation**
- **Command following at the start is excellent; mid-task re-steering is poor.** By the protocol's rule, this command-following deficit justifies steering training (S00/S01).
- **G0.5 is fully Markovian.** Withdrawing the instruction costs 56 pts.
- **Stale memory is worse than none.** The policy faithfully executes whatever it is told, so a memory module must handle revisions correctly (the X02 concern).
- **Every adapter variant reduces steerability and instruction-free robustness** compared with plain G0.5. Extra visual conditioning learned from demonstrations increases reliance on the initial context. J2 is the least affected.

---

## 7. Steerability fix (S00/S01-style): closed-loop result

- **Collection** (T4, released G0.5): on LIBERO-Goal, G0.5 executes the source command for 2 s, then the target. Successful post-switch segments are recorded, from **training pairs only** (s = t+1, t+2, t+4, t+5; 20 trials each). That gives **145 recovery episodes**. The evaluation pairs (s = t+3) are never used.
- **Fine-tune:**
  - **F01-recov:** F01 + 2,000 steps on a 50/50 mix of original and recovery data.
  - **F01-cont (matched control):** the same 2,000 steps on original data only.
  - Both: batch 16, 4 flow samples, fp16, T4.
- **Re-test:** steering probes on the held-out pairs, 100 paired episodes per cell.

| Condition | Released G0.5 | F01 (adapter) | F01-cont (control) | **F01-recov** | recov vs cont |
|---|---|---|---|---|---|
| ref: own command | 98% | 97% | 99% | 96% | −3, n.s. |
| cf: another task's scene | 98% | 98% | 98% | 95% | −3, n.s. |
| **switch at 2 s** | 29% (T4: 31%) | 21% | 20% | **28%** | **+8, p = 0.008** (8 wins, 0 losses) |
| instruction withdrawn at 2 s | 42% | 25% | 23% | 31% | +8, p = 0.13 |

**Interpretation**
- **Recovery data improves switching** over a matched control (+8 pts, significant), at a small non-significant cost on ref/cf.
- **The gain only undoes the adapter's own steerability loss.** F01-recov is back to the released G0.5 level (28% vs 29%), not above it. With 145 episodes and an adapter-only fine-tune, the policy still mostly finishes the old task.
- **Next steps:** more recovery data (DAgger rounds), a full G0.5 fine-tune rather than an adapter, and switch-time augmentation. All of these need the H200 path.

---

## 8. New components beyond LaWAM

### 8a. Small JEPA-style world model (our own)

- **Model:** action-conditioned transformer, **17.3M parameters**. Predicts pooled DINOv3 features (8×8 tokens) Δt = 0.4 s ahead from the current features plus an 8-step action chunk.
- **Training:** 1-step + 2-step loss, 20k steps, trained on a T4. Episodes with index % 20 == 0 are held out.

| Horizon | Error ÷ persistence | On moving frames | Beats persistence |
|---|---|---|---|
| 0.4 s | **0.41×** | 0.36× | 99.7% |
| 0.8 s (rollout) | **0.32×** | 0.28× | 99.8% |
| 1.6 s (rollout) | **0.28×** | 0.25× | 99.9% |

- **Action sensitivity:** permuted actions give **2.35×** the error; the correct actions are better 99.3% of the time.
- **Offline planning signal (P01-lite):** among 4 candidate action chunks (the expert's + 3 from other episodes), choosing the one whose predicted 1.6 s future is closest to the episode's goal image picks the **expert's chunk 46%** of the time (chance 25%).
- **Hard-negative planning (WM-1):** the 3 distractors are chunks from the *same* episode, taken 1.2 s before, 1.2 s after and 2.4 s after the current time (971 held-out start states):

  | Goal given to the planner | Expert chunk ranked first | Chance |
  |---|---|---|
  | **Subgoal image 1.6 s ahead** | **99.5%** | 25% |
  | Final frame of the episode | 20.8% | 25% |

  With a near subgoal, the model almost always identifies the right action chunk, even against distractors that look alike. With a distant goal, "skipping ahead" chunks from later in the episode get closer to it, so greedy one-shot ranking fails. **The world model is ready for subgoal-conditioned short-horizon planning (MPC). Long horizons need subgoals from a higher level (H00-style).**
- **vs LaWAM**, same 8×8 resolution, 512 frames:

  | | Error ÷ persistence |
  |---|---|
  | ours (executed actions) | **0.41×** |
  | LaWAM, 538M encoder + decoder (privileged posterior latent) | 0.65× |

  Caveats: LaWAM is general-purpose, pretrained at a 1.2 s horizon and natively 16×16; ours is LIBERO-specific.

### 8b. Episodic memory adapters (offline decision test)

- **Training:** adapter + memory branch (4 past frames → resampler, 16 tokens). The instruction is blanked with p = 0.5 after 2 s. 3,000 steps, with held-out episodes excluded.
- **Evaluation:** frozen-noise flow-matching loss on 768 held-out frames (lower is better).

| | Instruction present | **Instruction blank** | Blank + shuffled memory | Memory masked |
|---|---|---|---|---|
| Masked-memory control | 0.0169 | **0.0404** | — | — |
| Fixed-offset memory (1/2/4/8 s back) | 0.0168 | **0.0404** | 0.0409 | 0.122 (out of distribution) |
| **SM1: 3D-Mem-style snapshot memory** | 0.0168 | **0.0405** | 0.0409 | 0.118 (out of distribution) |

SM1's memory works like this:
- **Writing:** event-based. A new snapshot is stored when the DINOv3 features change past a threshold calibrated to ~10 snapshots per episode (median 8), similar to 3D-Mem's ~11.
- **Reading:** the first snapshot plus the 3 most recent snapshots.

**Interpretation:** losing the instruction makes action prediction 2.4× worse, and neither fixed-offset nor event-selected visual memory recovers any of it. The policy barely uses its memory content (a shuffled memory changes little). Recent visual history doesn't carry the task identity in a form this training discovers. That points to **remembering semantic or task information** (instruction/goal memory, or memory read by a planner) rather than raw visual history.

*Relation to 3D-Mem (CVPR 2025):* 3D-Mem targets navigation and QA with RGB-D, 3D object fusion and GPT-4o retrieval. Only its snapshot-selection idea was adapted here, to a low-level manipulation policy. This is not a reproduction.

### 8c. MEM-1: does memory carry task information the current frame lacks?

- **Model:** a 4-layer transformer reads pooled DINOv3 features and predicts the task (10-way within the suite).
- **Evaluation:** held-out episodes, frames ≥ 2 s into the episode. 95% CIs come from resampling episodes. Seeds 0–2.

| Where | Error, current frame only | Error, current + 4 past frames (1/2/4/8 s back) |
|---|---|---|
| **LIBERO-Spatial, all frames ≥ 2 s** | **2.9%** | **~0%**: every seed's CI excludes 0 |
| Spatial + Object + Goal, 4–8 s | 1.43% | 0.07% |
| First 2–4 s | 3.2% | 3.4% (no useful history yet) |
| Goal, Object, Long | ~0% | ~0% |

LIBERO-Spatial has two identical bowls. After pick-up, only the history shows which one was taken. 3D-Mem-style snapshot memory gives the same result as fixed offsets.

### 8d. MEM-2: closed-loop memory module for G0.5

- **Pipeline:**
  1. G0.5 gets the instruction for 2 s, then a memory token in its place.
  2. At that moment the MEM-1 decoder reads the episode's frames and outputs a task.
  3. That task's instruction is sent to G0.5.
- **Setup:** G0.5 is unchanged. LIBERO-Goal, 10 tasks × 10 paired trials. Run on T4s in fp16 (our port; `ref` 99% matches H200's 98%, `m_none` 40% vs 42%).

| Condition | Success | vs no memory: diff [95% CI over tasks], McNemar p | Task inferred correctly |
|---|---|---|---|
| Instruction throughout (ceiling) | 99% | +59 | — |
| Instruction withdrawn, no memory | **40%** | — | — |
| + memory module, current frame | **86%** | **+46 [+18, +74]**, p = 1e-10 | 87% |
| + memory module, first + current frame | 83% | +43 [+18, +71], p = 2e-11 | 84% |
| + memory module, fixed offsets | 83% | +43 [+15, +72], p = 7e-10 | 85% |
| + memory module, 3D-Mem-style snapshots | **86%** | **+46 [+16, +76]**, p = 3e-10 | **90%** |
| Command revised at 2 s, withdrawn at 4 s, no memory | 20% | — | — |
| same + memory module (any variant) | 20% | 0 | 6–9% |

**Interpretation**
- **A separate memory module recovers ~75% of the performance G0.5 loses when the instruction disappears.** That's 43–46 of the 59 points, with no change to G0.5.
- **In LIBERO-Goal the current frame already identifies the task**, so memory variants match the current-frame module here (83–86%). Memory specifically matters where the scene is ambiguous (Spatial, §8c).
- **Revisions fail for a steering reason, not a memory reason.** After a revision G0.5 mostly keeps doing the old task (switch success 29%, §6), so the history shows the old task and the module re-issues it. Memory cannot fix steerability. The two problems have to be solved together.

### 8e. World-model ablations (WM-1, 8k steps each) and memory in the world model (WM-MEM)

| Variant | Params | Error ÷ persistence at 0.4 / 0.8 / 1.6 s | Planning, random negatives | Planning, hard negatives with 1.6 s subgoal |
|---|---|---|---|---|
| Base (1+2-step loss) | 17.3M | 0.450 / 0.350 / 0.317 | 45.8% | 99.4% |
| 1-step loss only | 17.3M | 0.448 / 0.363 / 0.344 | 45.8% | 99.3% |
| Small | 2.7M | 0.559 / 0.459 / 0.429 | 46.0% | 99.3% |
| **Large** | 43.0M | **0.403 / 0.306 / 0.274** | 45.5% | **99.8%** |
| 25% of training episodes | 17.3M | 0.469 / 0.368 / 0.336 | 46.0% | 99.2% |
| **+ memory, 0.8–4 s history** | 17.9M | 0.449 / 0.349 / 0.315 | — | — |

- **Scaling helps**: 2.7M → 17M → 43M parameters improves every horizon.
- **The multi-step loss matters for long rollouts**: 0.317 vs 0.344 at 1.6 s.
- **The model is data-efficient**: a quarter of the data costs only ~6%.
- **Memory in the world model gives a 0.4% gain**, which is significant at 0.4–0.8 s but negligible in size.

### 8f. LIBERO-Mem (standard non-Markovian benchmark; Chung et al., arXiv 2511.11478)

**Benchmark.** 10 LIBERO tasks in which the current frame can't tell you what to do:
- objects are visually identical;
- tasks require "N times", swaps and rotations;
- the "nearest/empty basket" tasks require remembering which basket was used.

Each task has 120 demos: 0–99 for training, **100–119 held out** and used for every evaluation below. Success comes from the benchmark's own subgoal-sequence checker (`_check_success(inc=True)` every step). Run on the T4 G0.5 port.

**(a) Closed loop: released G0.5, zero-shot, no LIBERO-Mem training** (20 held-out starting states per task)

| Task | Subgoals | G0.5 success | G0.5 subgoal completion | Human demo replay (checker ceiling) |
|---|---|---|---|---|
| Bowl back on plate (once) | 1 | 15% | 15% | 55% |
| Bottle on plate (once) | 1 | 35% | 35% | 5% |
| Bowl 3× / bottle 3× | 3 | 0% / 5% | 3% / 25% | 50% / 5% |
| Bowl 5× / bowl 7× | 5 / 7 | 0% / 0% | 3% / 1% | 30% / 40% |
| Swap 2 bowls / rotate 3 bowls | 3 / 4 | 0% / 0% | 10% / 1% | 50% / 35% |
| Cheese → nearest basket → *that* / *empty* basket to centre | 2 | 5% / 0% | **53% / 48%** | 100% / 90% |
| **Average** | | **6%** | **19%** | |

- **G0.5 manages single pick-and-place, but fails everything that needs tracking progress.** That's 0/120 on repeat, swap and rotate. On the basket tasks it does step 1 half the time and step 2 almost never.
- **For reference, the paper reports** π0 at 5% average subgoal completion and slot-memory models at 0–15%. Our protocol stops an episode at its first success, so overshooting isn't penalised. Treat the comparison as approximate.
- **The checker is strict:** human demo replays pass only 5–100% of the time.

**(b) Offline: can memory supply the missing state?**
- **What's decoded:** how many pick-and-place cycles have been completed. The label comes from gripper release events in the demo. Once the object is back on its plate, the current frame can't show this.
- **Data:** every demo re-rendered by exact state replay at 5 Hz, as G0.5 sees it (256 px, same image transform), using pooled DINOv3 features. 4-layer transformer decoder trained on demos 0–99 and evaluated on held-out demos 100–119 (16,471 frames).
- **Statistics:** paired, with 95% CIs from resampling episodes.

| Frames given to the decoder | All frames | After the 1st cycle | **Repetition tasks (3×/5×/7×), after the 1st cycle** | Swap / rotate, after the 1st cycle |
|---|---|---|---|---|
| Current frame only (Markovian) | 64.1% | 46.5% | **14.9%** | 77.1% |
| First + current | 63.1% | 44.6% | 11.8% | 75.4% |
| 3D-Mem-style snapshots (first + 3 recent) | 65.4% | 46.3% | 13.6% | 78.9% |
| 1/2/4/8 s back + current | 85.6% | 74.7% | 59.3%: **+44 [+41, +48]** | 89.8%: +13 [+9, +18] |
| **8 frames spanning the episode + current** | **87.9%** | **79.1%** | **71.9%: +57 [+51, +63]** | 84.6%: +8 [0, +16] |

- **Accuracy by count, current frame vs full-episode memory:**

  | Cycles completed so far | 1 | 2 | 3 | 4 | 5 | 6 |
  |---|---|---|---|---|---|---|
  | Current frame only | 59% | 49% | 43% | 22% | 1% | 6% |
  | Full-episode memory | 92% | 79% | 65% | 55% | 56% | 69% |

  **Short memory (≤ 8 s) helps for small counts. Large counts need memory that spans the whole episode.**
- **Selection matters:** event snapshots (first + 3 recent) and the first frame give no gain. A good memory has to keep the *count events*, not just recent or salient frames.
- **This doesn't yet change G0.5's success.** Turning (b) into a closed-loop gain needs G0.5 to execute LIBERO-Mem steps, which requires fine-tuning G0.5 on LIBERO-Mem (an H200 job), plus a memory module that tracks the count and re-prompts G0.5, as in MEM-2.

### 8g. Closed loop: planning with our world model on top of frozen G0.5 (P01)

**Task.** LIBERO-Goal: 10 tasks in one shared scene, 10 trials per task with paired starting states. The robot gets **no instruction**. Instead it gets a **goal video**: the mean DINOv3 features of the target task's training demos, at 0.4 s intervals.

**Planner.** Every 10 control steps (0.5 s):
1. G0.5 (frozen, released checkpoint) proposes one 32-step action chunk for each of the 10 LIBERO-Goal commands.
2. Our 43M world model (`wm_large8k`) rolls each chunk forward 4 × 0.4 s from the current features.
3. The chunk whose predicted 1.6 s future is closest to the goal-video frame 1.6 s ahead is executed.

The *commit* variant chooses this way for the first 3 replans (1.5 s), then keeps the best-scoring command. G0.5 runs on the T4 port.

| Planner | Success | vs random: McNemar | Correct command chosen |
|---|---|---|---|
| Correct instruction given (upper bound) | 98/100 | — | 100% |
| **World model, every replan** | **77/100** | **+75 pts, p = 5e-23** | 32–60% of replans |
| **World model, commit after 1.5 s** | **76/100** | **+74 pts, p = 2e-20** | 96–100% on 7/10 tasks |
| Random command each replan | 2/100 | — | 10% |
| No instruction (blank) | 7/100 | — | — |

Per task (success out of 10, tasks 0–9):

| Planner | t0 | t1 | t2 | t3 | t4 | t5 | t6 | t7 | t8 | t9 |
|---|---|---|---|---|---|---|---|---|---|---|
| World model, every replan | 3 | 10 | 10 | 7 | 10 | 6 | 5 | 9 | 9 | 8 |
| World model, commit | 10 | 4 | 10 | 5 | 0 | 9 | 9 | 10 | 9 | 10 |

**Interpretation**
- **A 43M latent world model turns a frozen VLA into a goal-conditioned planner.** It lifts success from 2–7% to 76–77% with no language and no policy training.
- **The world model picks behaviours that make progress, not command labels.** Per-step selection picks the target's own command only 32–60% of the time, yet succeeds 77%, because other commands' chunks often share the early motion.
- **The two variants fail on different tasks**, so they are complementary. Commit mode locks onto a look-alike command on t4 (bowl → top of cabinet vs stove/plate) and t1. Per-step mode wavers on t0 and t6.
- **A distant goal fails, a near subgoal works.** With only the final goal image, selection failed (0/5 in a pilot, target chosen 7%). With a time-matched subgoal 1.6 s ahead it works, matching the offline finding (99.5% with a near subgoal). Long-horizon tasks need subgoals, from demos (as here) or from a high-level planner.

**Steering through the planner (goal switch at 2 s, the same s = t+3 pairs as the language-switch probe):**

| | Success |
|---|---|
| Command switches at 2 s (G0.5 with the correct command before and after) | 32/100 |
| **Goal video switches at 2 s (world-model planner)** | **15/100** (p = 5e-4, worse) |

Planning can't steer G0.5 where G0.5's own proposals don't go. After a switch, G0.5's chunks under every candidate command stay committed to the old object, so the planner has nothing good to pick. **Steerability has to come from the policy, through training. A planner only helps if the proposals already contain the right behaviour.**

### 8h. Closed loop: memory module on LIBERO-Mem (fine-tuned G0.5)

- **Policy:** G0.5 fine-tuned on LIBERO-Mem training demos 0–99, re-rendered exactly as G0.5 sees them. Each demo is labelled with both its original instruction and a single-cycle instruction; the trainer is `07b_train_lmem.py`. 3,000 steps, batch 16, fp16:
  - **expert:** the action expert unfrozen (635M parameters), on a V100;
  - **adapter:** adapter only, on a T4.
- **Memory module:** a 4-layer transformer over DINOv3 features.
  - It counts completed pick-and-place cycles from 8 frames spanning the episode plus the current frame.
  - It's trained on demo frames with geometric cycle labels: the object is lifted, then set back down.
  - It halts the robot once the count reaches N on 2 consecutive frames.
  - The policy is prompted with the single-cycle instruction.
- **Evaluation:** 6 repetition tasks (bowl 1×/3×/5×/7×, bottle 1×/3×) × 20 **held-out** starting states (demos 100–119), paired across conditions.
  - Episodes are **full length**, and success is the benchmark's overshoot-aware check at the final step, so a policy that keeps repeating fails.
  - G0.5 runs on the T4 port.

| Condition | Expert fine-tune | Adapter fine-tune |
|---|---|---|
| A. Fine-tuned G0.5, original instruction, no module | 5.8% | 4.2% |
| B. Single-cycle instruction, no module (never stops) | 7.5% | 4.2% |
| **C. + memory module (full-episode counter)** | **19.2%** | **11.7%** |
| D. + current-frame counter (Markovian control) | 4.2% | 2.5% |
| E. + oracle counter (perfect memory, upper bound) | 41.7% | 27.5% |

Paired McNemar tests, expert fine-tune:
- **C vs A:** 21 wins vs 5, **p = 0.003**;
- **C vs D:** 22 wins vs 4, **p = 5e-4**;
- **C vs E:** 3 wins vs 30, p = 1e-6;
- **A vs E:** 46 wins vs 3, p = 7e-11.

Adapter fine-tune: C vs A p = 0.06, C vs D p = 0.003.

Per task, expert fine-tune (success out of 20):

| Condition | bowl 1× | bottle 1× | bowl 3× | bottle 3× | bowl 5× | bowl 7× |
|---|---|---|---|---|---|---|
| A. alone | 1 | 1 | 2 | 0 | 1 | 2 |
| C. memory module | 14 | 3 | 4 | 1 | 0 | 1 |
| D. current-frame counter | 2 | 0 | 0 | 0 | 3 | 0 |
| E. oracle counter | 18 | 3 | 13 | 1 | 8 | 7 |

**Interpretation**
- **The fine-tuned G0.5 can do the cycles but not stop.** Subgoal completion is 59%, yet final success is 6%: it keeps going and overshoots. That's the non-Markovian failure LIBERO-Mem was built to expose.
- **Memory fixes much of it, a Markovian module doesn't.** The episodic-memory counter triples success (5.8% → 19.2%). The same module reading only the current frame is no better than no module (4.2%).
- **There's large headroom.** A perfect counter reaches 41.7% (57.5% on the bowl tasks), so the learned counter captures about 40% of the gain. Its errors are mostly premature halts at high counts (5×/7×), where counting from video is hardest.
- **The bottle tasks are near zero for every condition.** The benchmark's checker also fails 95% of human-demo replays on them (§8f), so they mostly measure the checker. Removing the memory raises its error by 4–14%, so the model does read it. **Given the actions, LIBERO physics is Markovian. Memory is valuable for task identity, not for dynamics.**

---

## 9. Not run, and why

| Protocol groups | Reason |
|---|---|
| M00/M01 full factorial, M03, minute-scale memory (10–900 s), X02 at scale | Need memory-dependent task suites and minute-long episodes. LIBERO episodes last ~20 s. |
| S00/S01 as LoRA on the expert | Replaced by the adapter-based recovery fine-tune (§7) |
| H00–H04 (high-level planner, verifier) | Need subgoal annotations and a planner integration |
| G00 (LIBERO-plus generalisation), I00 (in-context), C00/C01 (continual), D00 (data efficiency) | Out of time for this round |
| P00 (planning with LaWAM's own world model) | Replaced by planning with our small world model (P01, done in closed loop, §8g) |
| X01 real embodiment (r1_vr) | No robot or real-data integration in this round |
| E00 runtime sweep | Not run. J1 throughput: 1.27 s/step (batch 64, 8 flow samples), ~45 samples/s, ~77% GPU use, 67 GB of 143 GB |

---

## 10. Suggested next steps

1. **Planning with the small world model:** closed-loop candidate ranking (P01). Sample K = 4 action chunks from G0.5, score them with the 17M world model against a goal, and execute the best.
2. **Steerability:** finish the recovery fine-tune (§7). If it works, iterate (DAgger-style) and extend it to a planner issuing subgoals.
3. **Memory:** semantic/task memory (instruction, completed subgoals) read by the policy or a planner, plus memory-dependent LIBERO tasks with longer chained episodes. Use §6 as the benchmark.
4. **World models on real data:** retrain the small world model on r1_vr data.

### Startup-team G0.5 training optimisations (b200-91h-run): what carries over

These were **not applied in this round**, for two reasons:
- **They target a different job.** They speed up a *full fine-tune* of G0.5 (the internal r1_g05 trainer). Here G0.5 is frozen: only small adapters/priors train, and the world-model and memory runs don't touch G0.5 at all.
- **Comparisons must stay matched.** Changing numerics mid-study (TF32, fused/compiled kernels) would break the paired comparisons against runs already finished.

They carry over directly to the next step, a G0.5 fine-tune on H200: steerability recovery at scale, joint world-model + policy training, or memory training through the expert.

| Optimisation | Applies to | Expected effect here |
|---|---|---|
| Batched ViT attention via SDPA | VLM vision tower | Large: the vision tower runs on every step of J1/J2-style joint training |
| Factored adaLN in the action expert | 24-layer expert, 8 flow samples | Large: the expert runs 8× per sample |
| TF32 matmuls | Everything in fp32 | Free speed-up; changes numerics, so use it for new studies only |
| Whole-layer `torch.compile` | Expert and VLM layers | Moderate; needs fixed shapes (pad prompts) |
| Prefix K/V broadcast across flow samples | Expert cross-attention to the VLM cache | Avoids 8× copies of the KV cache; memory and speed |
| Micro-batch 64 × 2 accumulation | Optimiser | Keeps batch 128 within 141 GB |
| `gc.freeze()` after start-up | Python overhead | Removes GC stalls |

Our J1 baseline is 1.27 s/step (batch 64, 8 flow samples, ~45 samples/s, ~77% GPU use). With the list above, ~50–55 windows/s per H200 is a realistic target for a full fine-tune.

---

## 11. Where things are (project root on the cluster, `g05/`)

| Item | Path |
|---|---|
| All job scripts | `jobs/` (`lawam_hybrid.py`, `07_train_hybrid.py`, `08_eval_screen.sh`, `19_steer_mem.sh`, `23_train_wm.py`, …) |
| Hybrid checkpoints and logs | `results/hybrid/{P,F01,F02,J1,J2,J3,Mmem,Mctrl,SM1,F01cont}_s*/` |
| Screen results and paired comparisons | `results/screen/*/screen_summary.json`, `results/screen/compare.json` |
| Steering and memory probes | `results/steer/*/steer_summary.json` |
| World model | `results/wm_s0/` |
| W00, F00, drift | `results/lawm_sanity/`, `results/hybrid/P_s0/f00_summary.json` |
| B01 | `results/B01_lawam/` |
| All closed-loop results + paired tests | `results/closed_loop_summary.json` (`jobs/53_closed_loop_summary.py`) |
| World-model planner | `jobs/49_wm_planner.py`, `50_wm_planner.sh`; `results/wmplan/{full_*,switch_*}/` |
| LIBERO-Mem: data, fine-tunes, memory module | `libero_mem/` (fork + raw demos); `jobs/43–48, 51, 52, 54`; `results/lmem_*`, `results/hybrid/LMEM_{ad_t4,ex_v100}/` |
| Steering recovery | `results/recovery/` (145 episodes); `results/hybrid/F01{recov,cont}_t4/`; `results/steer/F01{recov,cont}_t4/` |
| T4/V100 port | `jobs/t4_fp16.py`, `serve_policy_hybrid.py` (fp16 + SDPA + LaWM offload), `07b_train_lmem.py` (fp16 trainer), `GalaxeaVLA/.venv_sm70` (Volta torch) |
| Plan | `g05 LaWAM/PLAN.md` (local) |
