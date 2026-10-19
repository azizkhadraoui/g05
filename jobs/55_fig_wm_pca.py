"""Figure data: what the world model predicts. Held-out LIBERO frames, true 1.6 s future, our 4-step rollout and
persistence, all as PCA-RGB maps of the pooled DINOv3 tokens (8x8). Output: results/figs/wm_pca.npz"""
import importlib, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import t4_fp16  # noqa: F401
G = "/export/home/kaziz/g05"; C = f"{G}/results/feat_cache"
os.makedirs(f"{G}/results/figs", exist_ok=True)
P = importlib.import_module("49_wm_planner".replace("49_wm_planner", "lawam_hybrid"))
T = importlib.import_module("07_train_hybrid")
m = np.load(f"{C}/meta.npz"); ep, fr, L = m["episode_index"], m["frame_index"], m["ep_length"]
act = m["action"].astype(np.float32)
feats = np.load(f"{C}/feats.npy", mmap_mode="r")
ck = torch.load(f"{G}/results/wm_large8k/wm.pt", map_location="cpu", weights_only=False)
sys.argv = ["x", "--server_uri", "x", "--mode", "oracle", "--output_dir", "/tmp/x"]
W = importlib.import_module("49_wm_planner")
wm = W.WM(ck["args"]["width"], ck["args"]["depth"], ck["args"]["k"]).cuda().eval(); wm.load_state_dict(ck["model"])
fm, fs = ck["f_mean"].cuda(), ck["f_std"].cuda(); am, as_ = ck["a_mean"], ck["a_std"]
rng = np.random.RandomState(3)
cand = np.nonzero((ep % 20 == 0) & (fr > 30) & (L - 1 - fr >= 40))[0]
rows = rng.choice(cand, 6, replace=False)
cur = torch.from_numpy(np.asarray(feats[np.sort(rows)], np.float32)).cuda(); rows = np.sort(rows)
fut = torch.from_numpy(np.asarray(feats[rows + 32], np.float32)).cuda()
u = (cur - fm) / fs
with torch.no_grad():
    for s in range(4):
        a = torch.from_numpy(((act[rows[:, None] + s * 8 + np.arange(8)[None]] - am) / as_).reshape(len(rows), -1)).cuda()
        u = wm(u, a).float()
pred = u * fs + fm
allf = torch.cat([cur, fut, pred]).reshape(-1, 768).cpu().numpy()
mu = allf.mean(0); U, S, Vt = np.linalg.svd(allf - mu, full_matrices=False)
proj = lambda x: (x.reshape(-1, 768).cpu().numpy() - mu) @ Vt[:3].T  # noqa: E731
pc = [proj(x).reshape(len(rows), 8, 8, 3) for x in (cur, fut, pred)]
lo, hi = np.percentile(np.concatenate([p.reshape(-1, 3) for p in pc]), [2, 98], axis=0)
pc = [np.clip((p - lo) / (hi - lo), 0, 1) for p in pc]
reader = T.FrameReader({"episode_index": torch.from_numpy(ep), "frame_index": torch.from_numpy(fr)})
f0 = reader.frames(torch.from_numpy(rows)).permute(0, 2, 3, 1).numpy(); f1 = reader.frames(torch.from_numpy(rows + 32)).permute(0, 2, 3, 1).numpy()
err = lambda a, b: ((((a - fm) / fs) - ((b - fm) / fs)) ** 2).mean((1, 2)).cpu().numpy()  # noqa: E731
np.savez(f"{G}/results/figs/wm_pca.npz", frame_t=f0, frame_t16=f1, pca_cur=pc[0], pca_true=pc[1], pca_pred=pc[2],
         err_pred=err(pred, fut), err_persist=err(cur, fut), rows=rows)
print("saved", err(pred, fut), err(cur, fut))
