#!/bin/bash -l
#SBATCH -o /export/home/kaziz/g05/logs/slurm-%x-%j.out
#SBATCH -p gpu-all
#SBATCH --gres gpu:T4_16GB:1
#SBATCH -c 8
#SBATCH --mem 36000MB
#SBATCH --time 02:00:00
# One PrefPI rollout batch on LIBERO-Goal (BowlToStove = task 1 by default): G0.5 server (94_frs_serve.py, optional
# fine-tuned expert) + LIBERO client (95_frs_client.py) recording every successful episode with its object heights.
# Env: OUT (iteration dir, required), EXPERT (expert ckpt, '' = released), CFG_W (1 = no guidance; PrefPI uses 5 from
#      iteration 1 on), PREFIX ('' or '[cfg] '), TASK (1), TRIALS (40), PAR (8), SEED (7)
source /export/home/kaziz/g05/env_g05.sh
cd $PROJECT_ROOT
: "${OUT:?}"; mkdir -p $OUT
TASK=${TASK:-1}; TRIALS=${TRIALS:-40}; PAR=${PAR:-8}; SEED=${SEED:-7}
PORT=$(( 20000 + SLURM_JOB_ID % 20000 ))
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export EXPERT_CKPT=${EXPERT:-}
echo "node=$(hostname) out=$OUT expert=${EXPERT:-released} cfg_w=${CFG_W:-1} prefix='${PREFIX:-}' task=$TASK trials=$TRIALS seed=$SEED"
$PY $G05_ROOT/jobs/94_frs_serve.py --ckpt_path checkpoints/g05-libero/model.pt --host 0.0.0.0 --port $PORT \
    eval_embodiment=libero --action_steps 10 --max_batch_size $PAR --max_wait_ms 1000 --seed $SEED &> $OUT/server.log &
SPID=$!; trap 'kill $SPID 2>/dev/null' EXIT
for i in $(seq 1 300); do
    grep -q "server listening on" $OUT/server.log 2>/dev/null && break
    kill -0 $SPID 2>/dev/null || { echo "server died"; tail -30 $OUT/server.log; exit 1; }
    sleep 2
done
grep -q "server listening on" $OUT/server.log || { echo "server not ready"; exit 1; }
rm -f $OUT/*_parallel_results.json
$PY $G05_ROOT/jobs/95_frs_client.py --server_uri ws://127.0.0.1:$PORT --task_suite_name libero_goal --task_id $TASK \
    --init_task_id $TASK --prompt_schedule "0:target" --frs_mode none --num_trials_per_task $TRIALS --num_steps_wait 10 \
    --num_parallel $PAR --env_resolution 256 --seed $SEED --output_dir $OUT --record_dir $OUT/rec --record_from 0 \
    --prompt_prefix "${PREFIX:-}" --cfg_plain_w ${CFG_W:-1} &> $OUT/client.log
ls $OUT/*_parallel_results.json > /dev/null 2>&1 || { echo "client failed"; tail -20 $OUT/client.log; exit 1; }
$PY - $OUT <<'PY'
import glob, json, sys, numpy as np
d = sys.argv[1]; r = json.load(open(glob.glob(f"{d}/*_parallel_results.json")[0]))["tasks"][0]
eps = [np.load(f, allow_pickle=True) for f in sorted(glob.glob(f"{d}/rec/*.npz"))]
L = [len(e["action"]) for e in eps]
bk = [k for k in (eps[0].files if eps else []) if k.startswith("objz_") and "bowl" in k]
H = [float(np.nanmax(e[bk[0]]) - e[bk[0]][0]) for e in eps] if bk else []
print(json.dumps({"successes": r["successes"], "episodes": r["total_episodes"], "recorded": len(eps),
                  "length_mean": float(np.mean(L)) if L else None, "bowl_key": bk[0] if bk else None,
                  "height_mean_cm": 100 * float(np.mean(H)) if H else None, "height_max_cm": 100 * float(np.max(H)) if H else None}))
PY
