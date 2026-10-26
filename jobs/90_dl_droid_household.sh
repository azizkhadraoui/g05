#!/bin/bash -l
#SBATCH -o /export/home/kaziz/g05/logs/slurm-%x-%j.out
#SBATCH -p cpu-all
#SBATCH -c 8
#SBATCH --mem 32000MB
#SBATCH --time 12:00:00
# Real household episodes from DROID (lerobot/droid_1.0.1, CC-BY 4.0; Franka, 3 cameras) for the zero-shot world-model
# evaluation: every episode whose instruction mentions towels, cloth, folding or laundry. Fetches the metadata, writes
# the selection to data/droid_household/selection.json, then downloads only the data/video files holding those episodes.
set -e
export TMPDIR=/export/home/kaziz/g05/tmp   # node-local /tmp can be full (here-documents need it)
G=/export/home/kaziz/g05; D=$G/data/droid_household; mkdir -p $D
export HF_TOKEN=$(cat $HOME/.cache/huggingface/token)
$G/GalaxeaVLA/.venv/bin/python - <<'PY'   # has pandas + huggingface_hub
import glob, json, os, re
import pandas as pd
from huggingface_hub import snapshot_download
R, D = "lerobot/droid_1.0.1", "/export/home/kaziz/g05/data/droid_household"
snapshot_download(R, repo_type="dataset", allow_patterns=["meta/**", "README.md"], local_dir=D, max_workers=8)
info = json.load(open(f"{D}/meta/info.json")); ver = info["codebase_version"]; print("version", ver, "episodes", info["total_episodes"])
FOLD = re.compile(r"\bfold", re.I); CLOTH = re.compile(r"\b(towels?|cloths?|clothes|clothing|shirts?|napkins?|garments?)\b", re.I)
LAUNDRY = re.compile(r"\b(laundry|washer|washing machine)\b", re.I)


class _KW:  # folding a towel/cloth, or laundry/washer (a looser keyword list matched 8k mostly unrelated episodes)
    pattern = "fold & (towel|cloth|clothes|shirt|napkin|garment)  OR  laundry|washer|washing machine"

    def search(self, t):
        if re.search(r"\bdish ?wash", t, re.I):
            return False
        return bool((FOLD.search(t) and CLOTH.search(t)) or LAUNDRY.search(t))


KW = _KW()
if ver.startswith("v3"):
    eps = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"{D}/meta/episodes/**/*.parquet", recursive=True))])
    tcol = "tasks" if "tasks" in eps.columns else [c for c in eps.columns if "task" in c][0]
    txt = eps[tcol].apply(lambda t: " | ".join(t) if hasattr(t, "__iter__") and not isinstance(t, str) else str(t))
    sel = eps[txt.apply(KW.search)]
    files = set()
    for _, r in sel.iterrows():
        files.add(info["data_path"].format(chunk_index=int(r["data/chunk_index"]), file_index=int(r["data/file_index"])))
        for vk in [k for k, v in info["features"].items() if v.get("dtype") == "video"]:
            files.add(info["video_path"].format(video_key=vk, chunk_index=int(r[f"videos/{vk}/chunk_index"]), file_index=int(r[f"videos/{vk}/file_index"])))
    rows = [{"episode_index": int(r["episode_index"]), "task": t} for (_, r), t in zip(sel.iterrows(), txt[sel.index])]
else:
    eps = [json.loads(l) for l in open(f"{D}/meta/episodes.jsonl")]
    sel = [e for e in eps if KW.search(" | ".join(e.get("tasks", [])))]
    chunk = info["chunks_size"]; files = set()
    for e in sel:
        c = e["episode_index"] // chunk
        files.add(info["data_path"].format(episode_chunk=c, episode_index=e["episode_index"]))
        for vk in [k for k, v in info["features"].items() if v.get("dtype") == "video"]:
            files.add(info["video_path"].format(episode_chunk=c, video_key=vk, episode_index=e["episode_index"]))
    rows = [{"episode_index": e["episode_index"], "task": " | ".join(e["tasks"])} for e in sel]
json.dump({"keywords": KW.pattern, "episodes": rows, "files": sorted(files)}, open(f"{D}/selection.json", "w"), indent=1)
print(len(rows), "household episodes in", len(files), "files; e.g.", [r["task"] for r in rows[:8]])
from huggingface_hub import HfApi
fl = sorted(files); size = sum(x.size or 0 for i in range(0, len(fl), 200) for x in HfApi().get_paths_info(R, fl[i:i + 200], repo_type="dataset"))
print(f"download size {size / 1e9:.1f} GB", flush=True)
if int(os.environ.get("META_ONLY", "0")) == 0 and ver.startswith("v3"):
    # v3.0 packs many episodes per video file (391 GB for these episodes): download the data parquet files whole, and cut
    # each selected episode's time range out of the remote videos over HTTP (ffmpeg seeks with range requests).
    import subprocess
    from concurrent.futures import ThreadPoolExecutor
    import imageio_ffmpeg
    snapshot_download(R, repo_type="dataset", allow_patterns=sorted(f for f in files if f.startswith("data/")), local_dir=D, max_workers=8)
    ff, tok = imageio_ffmpeg.get_ffmpeg_exe(), os.environ["HF_TOKEN"]
    vks = [k for k, v in info["features"].items() if v.get("dtype") == "video"]
    jobs = []
    for _, r in sel.iterrows():
        for vk in vks:
            path = info["video_path"].format(video_key=vk, chunk_index=int(r[f"videos/{vk}/chunk_index"]), file_index=int(r[f"videos/{vk}/file_index"]))
            out = f"{D}/clips/{int(r['episode_index']):06d}/{vk.split('.')[-1]}.mp4"
            jobs.append((path, float(r[f"videos/{vk}/from_timestamp"]), float(r[f"videos/{vk}/to_timestamp"]), out))

    def cut(j):
        path, t0, t1, out = j
        if os.path.exists(out) and os.path.getsize(out) > 0:
            return 0
        os.makedirs(os.path.dirname(out), exist_ok=True)
        cmd = [ff, "-y", "-loglevel", "error", "-headers", f"Authorization: Bearer {tok}\r\n", "-ss", f"{t0:.3f}", "-to", f"{t1:.3f}",
               "-i", f"https://huggingface.co/datasets/{R}/resolve/main/{path}", "-c:v", "libx264", "-crf", "18", "-preset", "veryfast", "-an", out + ".tmp.mp4"]
        for _ in range(3):
            if subprocess.run(cmd, capture_output=True).returncode == 0:
                os.replace(out + ".tmp.mp4", out); return 0
        return 1

    with ThreadPoolExecutor(int(os.environ.get("PAR", "16"))) as ex:
        fails = sum(ex.map(cut, jobs))
    print(f"clips: {len(jobs) - fails}/{len(jobs)} cut ({len(rows)} episodes x {len(vks)} cameras)", flush=True)
PY
du -sh $D; echo ALL_DONE
