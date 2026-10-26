#!/bin/bash -l
#SBATCH -o /export/home/kaziz/g05/logs/slurm-%x-%j.out
#SBATCH -p gpu-all
#SBATCH --gres gpu:v100nv_32GB:1
#SBATCH -c 12
#SBATCH --mem 96000MB
#SBATCH --time 48:00:00
# GR00T N1.7-3B fine-tuning on the RoboCasa365 target split (50 tasks, human demos) on one V100, fp16 (83_gr00t_v100_launch.py).
# Default partial fine-tune of GR00T: action head (DiT) + projector trained, Cosmos-Reason2 backbone frozen.
# Env: OUT (required), MAX_STEPS (30000), BATCH (32), SAVE_STEPS (5000), TASKS_GLOB (target/*/*/*/lerobot), WORKERS (8)
set -eu
G=/export/home/kaziz/g05; GR=$G/baselines/Isaac-GR00T
DS=$G/robocasa365/robocasa/datasets/v1.0
DATASET_PATH=$(ls -d $DS/${TASKS_GLOB:-target/*/*/*/lerobot} | sort | paste -sd: -)
echo "node=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader) datasets=$(echo $DATASET_PATH | tr ':' '\n' | wc -l) out=$OUT"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled TOKENIZERS_PARALLELISM=false  # hub ids resolve to the local cache (Cosmos-Reason2-2B is linked there)
export CUDA_HOME=/cm/shared/apps/cuda12.6/toolkit/12.6.2   # deepspeed reads the toolkit version at import (torch is cu126)
cd $GR
.venv/bin/python -u $G/jobs/83_gr00t_v100_launch.py \
  --base_model_path $G/models/GR00T-N1.7-3B --dataset_path "$DATASET_PATH" --embodiment_tag NEW_EMBODIMENT \
  --modality_config_path $G/jobs/82_gr00t_rc365_config.py --num_gpus 1 --output_dir $OUT \
  --save_steps ${SAVE_STEPS:-5000} --save_total_limit 3 --max_steps ${MAX_STEPS:-30000} --warmup_ratio 0.05 \
  --weight_decay 1e-5 --learning_rate 1e-4 --global_batch_size ${BATCH:-32} --gradient_accumulation_steps ${ACC:-1} --dataloader_num_workers ${WORKERS:-8}
