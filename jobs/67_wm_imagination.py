"""Explainer video data: the world model's 1.6 s-ahead imagination along one held-out LIBERO-Goal demo.

Every 4 frames (0.2 s): camera now, camera 1.6 s later, and the DINOv3 tokens (8x8) of the true future and of our 43M
world model's 4-step rollout driven by the executed actions, as PCA colours (one PCA for the whole episode).
Output: results/video/wm_imagination.npz
"""
import glob, importlib, os, sys
import numpy as np, pandas as pd, torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import t4_fp16  # noqa: F401,E402
G = "/export/home/kaziz/g05"; C = f"{G}/results/feat_cache"; DS = f"{G}/data/libero_merged_g05grip"
os.makedirs(f"{G}/results/video", exist_ok=True)
TASK = "open the top drawer and put the bowl inside"
m = np.load(f"{C}/meta.npz"); ep, fr, L = m["episode_index"], m["frame_index"], m["ep_length"]
act = m["action"].astype(np.float32); feats = np.load(f"{C}/feats.npy", mmap_mode="r")
d = pd.concat([pd.read_parquet(f, columns=["index", "task_index"]) for f in sorted(glob.glob(f"{DS}/data/*/*.parquet"))]).sort_values("index")
tidx = d["task_index"].values
tp = pd.read_parquet(f"{DS}/meta/tasks.parquet").reset_index(); tcol = [c for c in tp.columns if tp[c].dtype == object][0]
norm = lambda s: " ".join(str(s).lower().replace("_", " ").split())  # noqa: E731
ti = {norm(r[tcol]): int(r["task_index"]) for _, r in tp.iterrows()}[norm(TASK)]
starts = [s for s in np.nonzero(fr == 0)[0] if tidx[s] == ti and ep[s] % 20 == 0]  # held out from world-model training
s0 = starts[0]; n = int(L[s0])
ck = torch.load(f"{G}/results/wm_large8k/wm.pt", map_location="cpu", weights_only=False)
sys.argv = ["x", "--server_uri", "x", "--mode", "oracle", "--output_dir", "/tmp/x"]
W = importlib.import_module("49_wm_planner")
wm = W.WM(ck["args"]["width"], ck["args"]["depth"], ck["args"]["k"]).cuda().eval(); wm.load_state_dict(ck["model"])
fm, fs = ck["f_mean"].cuda(), ck["f_std"].cuda(); am, as_ = ck["a_mean"], ck["a_std"]
ts = np.arange(0, n - 33, 4); rows = s0 + ts
cur = torch.from_numpy(np.asarray(feats[rows], np.float32)).cuda(); fut = torch.from_numpy(np.asarray(feats[rows + 32], np.float32)).cuda()
u = (cur - fm) / fs
with torch.no_grad():
    for s in range(4):
        a = torch.from_numpy(((act[rows[:, None] + s * 8 + np.arange(8)[None]] - am) / as_).reshape(len(rows), -1)).cuda()
        u = wm(u, a).float()
pred = u * fs + fm
allf = torch.cat([cur, fut, pred]).reshape(-1, 768).cpu().numpy(); mu = allf.mean(0)
_, _, Vt = np.linalg.svd(allf - mu, full_matrices=False)
proj = lambda x: (x.reshape(-1, 768).cpu().numpy() - mu) @ Vt[:3].T  # noqa: E731
pcs = [proj(x).reshape(len(rows), 8, 8, 3) for x in (cur, fut, pred)]
lo, hi = np.percentile(np.concatenate([p.reshape(-1, 3) for p in pcs]), [2, 98], axis=0)
pcs = [np.clip((p - lo) / (hi - lo), 0, 1) for p in pcs]
err = lambda a_, b_: ((((a_ - fm) / fs) - ((b_ - fm) / fs)) ** 2).mean((1, 2)).cpu().numpy()  # noqa: E731
T = importlib.import_module("07_train_hybrid")
reader = T.FrameReader({"episode_index": torch.from_numpy(ep), "frame_index": torch.from_numpy(fr)})
f_now = reader.frames(torch.from_numpy(rows)).permute(0, 2, 3, 1).numpy(); f_fut = reader.frames(torch.from_numpy(rows + 32)).permute(0, 2, 3, 1).numpy()
np.savez_compressed(f"{G}/results/video/wm_imagination.npz", t=ts, frame_now=f_now, frame_future=f_fut, pca_now=pcs[0], pca_true=pcs[1],
                    pca_pred=pcs[2], err_pred=err(pred, fut), err_persist=err(cur, fut), task=TASK)
print("saved", len(ts), "steps; mean error ratio", float(err(pred, fut).mean() / err(cur, fut).mean()))
