#!/bin/bash -l
#SBATCH -o /export/home/kaziz/g05/logs/slurm-%x-%j.out
#SBATCH -p gpu-all
#SBATCH --gres gpu:T4_16GB:2
#SBATCH -c 16
#SBATCH --mem 100000MB
#SBATCH --time 03:00:00
# Showcase mission (73_showcase.py). GPU 0: G0.5 server (32-step chunks). GPU 1: RoboBrain, world model, DINOv3, memory.
source /export/home/kaziz/g05/env_g05.sh
cd $PROJECT_ROOT
D=$G05_ROOT/results/showcase; mkdir -p $D
PORT=$(( 20000 + SLURM_JOB_ID % 20000 ))
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONFAULTHANDLER=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES%%,*} $PY $G05_ROOT/jobs/serve_policy_hybrid.py --ckpt_path checkpoints/g05-libero/model.pt --host 0.0.0.0 --port $PORT \
    eval_embodiment=libero --action_steps 32 --max_batch_size 10 --max_wait_ms 200 --seed 7 &> $D/server.log &
SPID=$!
for i in $(seq 1 300); do ss -tlnpH "sport = :$PORT" 2>/dev/null | grep -q "$PORT" && break; kill -0 $SPID 2>/dev/null || { tail -30 $D/server.log; exit 1; }; sleep 2; done
$PY -u $G05_ROOT/jobs/73_showcase.py --server_uri ws://127.0.0.1:$PORT --episodes ${EPISODES:-6} --num_parallel ${PAR:-3} --output_dir $D ${EXTRA:-}
kill $SPID
