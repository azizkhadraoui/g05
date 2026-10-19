"""README figures. All numbers are the measured results recorded in RESULTS.md / results/closed_loop_summary.json."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import os
OUT = os.path.dirname(os.path.abspath(__file__))
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 130})
BLUE, ORANGE, GREEN, RED, GREY, PURPLE = "#2b6cb0", "#dd6b20", "#2f855a", "#c53030", "#a0aec0", "#6b46c1"


def bars(ax, labels, vals, colors, title, ylabel="Success (%)", ylim=105, fmt="{:.0f}"):
    x = np.arange(len(labels))
    ax.bar(x, vals, color=colors, width=0.7)
    for xi, v in zip(x, vals):
        ax.text(xi, v + ylim * 0.015, fmt.format(v), ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8.5); ax.set_ylim(0, ylim); ax.set_title(title, fontsize=10.5); ax.set_ylabel(ylabel)


# ---- Fig 1: closed-loop headline ----
fig, axs = plt.subplots(2, 2, figsize=(13, 8.6)); axs = axs.ravel()
bars(axs[0], ["correct\ninstruction", "world model\n(goal video)", "WM commit\n(goal video)", "no\ninstruction", "random\ncommand"],
     [98, 77, 76, 7, 2], [GREY, BLUE, BLUE, GREY, GREY], "World-model planning over frozen G0.5\n(LIBERO-Goal, 100 paired episodes)")
bars(axs[1], ["instruction\nkept", "withdrawn,\nno memory", "+ module\ncurrent", "+ module\nfirst+cur", "+ module\noffsets", "+ module\nsnapshots"],
     [99, 40, 86, 83, 83, 86], [GREY, RED, ORANGE, GREEN, GREEN, GREEN], "Memory module re-issues withdrawn\ninstruction (MEM-2, 100 episodes)")
x = np.arange(5); w = 0.38
lab = ["A alone", "B no stop", "C memory\nmodule", "D current-\nframe", "E oracle\ncounter"]
ex = [5.8, 7.5, 19.2, 4.2, 41.7]; ad = [4.2, 4.2, 11.7, 2.5, 27.5]
cols = [GREY, GREY, GREEN, ORANGE, PURPLE]
axs[2].bar(x - w / 2, ex, w, color=cols)
axs[2].bar(x + w / 2, ad, w, color=cols, alpha=0.45)
for xi, v in zip(x, ex):
    axs[2].text(xi - w / 2, v + 0.8, f"{v:.0f}", ha="center", fontsize=8.5)
for xi, v in zip(x, ad):
    axs[2].text(xi + w / 2, v + 0.8, f"{v:.0f}", ha="center", fontsize=8.5)
axs[2].set_xticks(x); axs[2].set_xticklabels(lab, fontsize=8.5); axs[2].set_ylim(0, 50); axs[2].set_ylabel("Success (%)")
axs[2].set_title("LIBERO-Mem repetition tasks: count and stop\n(120 held-out ep.; solid = expert FT, light = adapter FT)", fontsize=10.5)
bars(axs[3], ["released\nG0.5", "adapter\nF01", "F01-cont\n(control)", "F01-recov\n(+recovery)", "command switch\n(planner run)", "goal-video\nswitch (WM)"],
     [29, 21, 20, 28, 32, 15], [GREY, GREY, ORANGE, GREEN, GREY, BLUE], "Mid-task switch success\n(LIBERO-Goal held-out pairs, 100 episodes each)", ylim=45)
plt.tight_layout(); plt.savefig(f"{OUT}/fig1_closed_loop.png"); plt.close()

# ---- Fig 2: world-model prediction quality + offline planning ----
fig, axs = plt.subplots(1, 2, figsize=(11.5, 4))
h = [0.4, 0.8, 1.6]
series = {"ours 43M (8k steps)": ([0.403, 0.306, 0.274], BLUE, "-o"), "ours 17M (20k steps)": ([0.41, 0.32, 0.28], GREEN, "-o"),
          "ours 17M, 1-step loss": ([0.448, 0.363, 0.344], ORANGE, "--s"), "ours 17M, 25% data": ([0.469, 0.368, 0.336], PURPLE, "--^"),
          "ours 2.7M": ([0.559, 0.459, 0.429], GREY, "-d")}
for k, (v, c, st) in series.items():
    axs[0].plot(h, v, st, color=c, label=k)
axs[0].scatter([0.4], [0.65], marker="*", s=160, color=RED, zorder=5, label="LaWAM 538M (privileged latent)")
axs[0].axhline(1.0, color="k", lw=0.8, ls=":")
axs[0].text(1.0, 0.97, "persistence (nothing moves)", fontsize=8, va="top")
axs[0].set_xlabel("prediction horizon (s)"); axs[0].set_ylabel("error / persistence error (lower = better)"); axs[0].set_ylim(0, 1.05)
axs[0].set_title("Latent world-model error, held-out LIBERO episodes", fontsize=10.5); axs[0].legend(fontsize=7.5, loc="center right")
bars(axs[1], ["expert vs\nrandom chunks\n(final-frame goal)", "expert vs same-\nepisode chunks\n(final-frame goal)", "expert vs same-\nepisode chunks\n(1.6 s subgoal)"],
     [45.8, 20.8, 99.5], [BLUE, ORANGE, GREEN], "Offline planning test: pick the right\naction chunk among 4 (dotted = chance)", ylabel="expert chunk ranked first (%)")
axs[1].axhline(25, color="k", lw=0.8, ls=":")
plt.tight_layout(); plt.savefig(f"{OUT}/fig2_world_model.png"); plt.close()

# ---- Fig 3: planner per task ----
tasks = ["open middle\ndrawer", "bowl on\nstove", "bottle on\ncabinet", "top drawer\n+ bowl", "bowl on\ncabinet", "push plate\nto stove",
         "cheese in\nbowl", "turn on\nstove", "bowl on\nplate", "bottle on\nrack"]
d = {"correct instruction": ([10, 10, 10, 10, 9, 9, 10, 10, 10, 10], GREY), "WM every replan": ([3, 10, 10, 7, 10, 6, 5, 9, 9, 8], BLUE),
     "WM commit": ([10, 4, 10, 5, 0, 9, 9, 10, 9, 10], GREEN), "no instruction": ([0, 1, 0, 0, 0, 0, 1, 0, 5, 0], ORANGE)}
fig, ax = plt.subplots(figsize=(13, 3.6)); x = np.arange(10); w = 0.2
for j, (k, (v, c)) in enumerate(d.items()):
    ax.bar(x + (j - 1.5) * w, v, w, color=c, label=k)
ax.set_xticks(x); ax.set_xticklabels(tasks, fontsize=8); ax.set_ylabel("successes / 10")
ax.legend(fontsize=8, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.18))
plt.tight_layout(); plt.savefig(f"{OUT}/fig3_planner_per_task.png"); plt.close()

# ---- Fig 4: memory offline ----
fig, axs = plt.subplots(1, 2, figsize=(12, 4))
lab = ["Spatial,\nall >= 2 s", "Spatial+Object+\nGoal, 4-8 s", "first 2-4 s\n(all suites)"]
cur = [2.9, 1.43, 3.2]; mem = [0.005, 0.07, 3.4]; x = np.arange(3); w = 0.36
axs[0].bar(x - w / 2, cur, w, color=ORANGE, label="current frame only")
axs[0].bar(x + w / 2, mem, w, color=GREEN, label="+ memory (1/2/4/8 s back)")
for xi, a_, b_ in zip(x, cur, mem):
    axs[0].text(xi - w / 2, a_ + 0.05, f"{a_:.2f}", ha="center", fontsize=8.5)
    axs[0].text(xi + w / 2, b_ + 0.05, f"{b_:.2f}", ha="center", fontsize=8.5)
axs[0].set_xticks(x); axs[0].set_xticklabels(lab, fontsize=9); axs[0].set_ylabel("task-ID error (%)"); axs[0].legend(fontsize=8)
axs[0].set_title("MEM-1: which task is the robot doing? (LIBERO, 3 seeds)", fontsize=10.5)
cnt = [1, 2, 3, 4, 5, 6]
for k, v, c, st in [("current frame", [58.7, 48.5, 43.0, 21.9, 0.9, 6.0], ORANGE, "-o"), ("3D-Mem snapshots", [61.6, 46.2, 43.4, 12.4, 2.4, 1.2], RED, "--x"),
                    ("1/2/4/8 s back", [94.6, 80.2, 57.2, 35.3, 12.0, 24.4], PURPLE, "-s"), ("whole episode (8 frames)", [91.8, 78.8, 65.3, 55.1, 56.0, 68.7], GREEN, "-o")]:
    axs[1].plot(cnt, v, st, color=c, label=k)
axs[1].set_xlabel("cycles completed so far (true)"); axs[1].set_ylabel("count predicted correctly (%)"); axs[1].set_ylim(0, 100); axs[1].legend(fontsize=8)
axs[1].set_title("LIBERO-Mem: how many cycles are done? (held-out demos)", fontsize=10.5)
plt.tight_layout(); plt.savefig(f"{OUT}/fig4_memory_offline.png"); plt.close()

# ---- Fig 5: steering / memory probe grid ----
pol = ["released G0.5", "F01 current adapter", "F02 LaWAM foresight", "J1 joint (detached)", "J2 joint (end-to-end)", "J3 future dropout",
       "F01-cont (control)", "F01-recov (+recovery)"]
cols5 = ["own command", "other scene,\nnew command", "switch\nat 2 s", "instruction\nwithdrawn at 2 s"]
M = np.array([[98, 98, 29, 42], [97, 98, 21, 25], [97, 95, 21, 26], [97, 97, 21, 24], [99, 97, 25, 32], [97, 99, 22, 30], [99, 98, 20, 23], [96, 95, 28, 31]])
fig, ax = plt.subplots(figsize=(7.6, 4.6)); im = ax.imshow(M, cmap="RdYlGn", vmin=0, vmax=100, aspect="auto")
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        ax.text(j, i, M[i, j], ha="center", va="center", fontsize=9)
ax.set_xticks(range(4)); ax.set_xticklabels(cols5, fontsize=8.5); ax.set_yticks(range(len(pol))); ax.set_yticklabels(pol, fontsize=8.5)
ax.set_title("Steering and memory probes (LIBERO-Goal, % success, 100 episodes per cell)", fontsize=10); plt.colorbar(im, fraction=0.03)
plt.tight_layout(); plt.savefig(f"{OUT}/fig5_probe_grid.png"); plt.close()

# ---- Fig 6: injecting LaWAM into G0.5 ----
fig, axs = plt.subplots(1, 2, figsize=(12, 3.8))
bars(axs[0], ["G0.5", "F01\ncurrent", "F02\nforesight", "J1\njoint", "J2\nend-to-end", "J3\ndropout"], [98.3, 96.7, 90.8, 97.5, 98.3, 97.5],
     [GREY, ORANGE, RED, GREEN, GREEN, GREEN], "Feeding LaWAM's predicted future into G0.5\n(6 hardest tasks x 20 paired trials)", fmt="{:.1f}")
axs[0].set_ylim(80, 102)
bars(axs[1], ["F02, normal\nprediction", "replaced by\ncurrent frame", "replaced by another\nepisode's prediction"], [90.8, 96.7, 98.3], [RED, ORANGE, GREEN],
     "F03: corrupting F02's future input at test time", fmt="{:.1f}")
axs[1].set_ylim(80, 102)
plt.tight_layout(); plt.savefig(f"{OUT}/fig6_lawam_injection.png"); plt.close()

# ---- Fig 7: LIBERO-Mem zero-shot ----
t = ["bowl x1", "bottle x1", "bowl x3", "bottle x3", "bowl x5", "bowl x7", "swap 2", "rotate 3", "basket\n(that)", "basket\n(empty)"]
g = [15, 35, 3, 25, 3, 1, 10, 1, 53, 48]; c = [55, 5, 73, 5, 76, 81, 78, 79, 100, 95]
fig, ax = plt.subplots(figsize=(12, 3.6)); x = np.arange(10); w = 0.38
ax.bar(x - w / 2, g, w, color=BLUE, label="released G0.5, zero-shot")
ax.bar(x + w / 2, c, w, color=GREY, label="human demo replay (benchmark checker ceiling)")
ax.set_xticks(x); ax.set_xticklabels(t, fontsize=8.5); ax.set_ylabel("subgoal completion (%)"); ax.legend(fontsize=8)
ax.set_title("LIBERO-Mem: G0.5 does single pick-and-place but cannot track progress", fontsize=10.5)
plt.tight_layout(); plt.savefig(f"{OUT}/fig7_libero_mem_zeroshot.png"); plt.close()
print("ok")

# ---- Fig 9: RoboBrain as the high-level layer ----
fig, axs = plt.subplots(2, 2, figsize=(13, 8.6)); axs = axs.ravel()
x = np.arange(3); w = 0.38
axs[0].bar(x - w / 2, [9, 6, 6], w, color=GREY, label="RoboBrain 2.0-3B")
axs[0].bar(x + w / 2, [16, 9, 37], w, color=PURPLE, label="RoboBrain 2.5-4B")
for xi, a_, b_ in zip(x, [9, 6, 6], [16, 9, 37]):
    axs[0].text(xi - w / 2, a_ + 1, f"{a_}", ha="center", fontsize=8.5); axs[0].text(xi + w / 2, b_ + 1, f"{b_}", ha="center", fontsize=8.5)
axs[0].axhline(10, color="k", lw=0.8, ls=":"); axs[0].set_xticks(x)
axs[0].set_xticklabels(["which task?\n(5 frames)", "which task?\n(current frame)", "which command\nreaches goal image?"], fontsize=8.5)
axs[0].set_ylim(0, 50); axs[0].set_ylabel("accuracy (%)"); axs[0].legend(fontsize=8)
axs[0].set_title("Zero-shot on held-out LIBERO-Goal demos\n(dotted = chance)", fontsize=10.5)
bars(axs[1], ["no memory", "RoboBrain\ncurrent frame", "RoboBrain\n5 frames", "trained\nclassifier"], [40, 31, 36, 86], [RED, GREY, PURPLE, GREEN],
     "MEM-2: instruction withdrawn at 2 s\n(100 paired episodes)")
bars(axs[2], ["random\ncommand", "our WM\n(goal image)", "RoboBrain\n(goal image)", "our WM\n(goal video)", "correct\ninstruction"], [2, 24, 42, 77, 98],
     [GREY, BLUE, PURPLE, BLUE, GREY], "Planning over frozen G0.5\n(100 paired episodes)")
bars(axs[3], ["G0.5 FT\nalone", "RoboBrain\ncurrent", "RoboBrain\nwhole episode", "trained\ncounter", "perfect\ncounter"], [5.8, 4.2, 2.5, 19.2, 41.7],
     [GREY, GREY, PURPLE, GREEN, GREY], "LIBERO-Mem: count cycles and stop\n(120 held-out episodes)", ylim=50)
plt.tight_layout(); plt.savefig(f"{OUT}/fig9_robobrain.png"); plt.close()
print("fig9 ok")
