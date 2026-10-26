#!/bin/bash -l
#SBATCH -o /export/home/kaziz/g05/logs/slurm-%x-%j.out
#SBATCH -p gpu-A100
#SBATCH -A a100
#SBATCH --qos a100_qos
#SBATCH -c 32
#SBATCH --mem 300000MB
#SBATCH --time 3-00:00:00
# LingBot-VLA 2.0 6B post-training on RoboCasa365 target-50 (rc365_v30, LeRobot v3) on A100s (flash-attention needs the
# glibc >= 2.32: both A100 nodes have 2.35). MODE=prep: MoGe-2 teacher download, robot + training configs, norm stats on 1% of the frames, and a
# short training run that reports speed and memory. MODE=train: the full post-training run.
# MODE=auto1: one GPU, bf16 weights (forward under bf16 autocast) + AnyPrecisionAdamW; probes Kahan / micro batch / compile
# for 20 steps, then trains with the first that works (keeps 2 checkpoints).
# Env: MODE (prep | train | auto1), MB (micro batch per GPU, 16), STEPS, SAVE (checkpoint every N steps), OUT (output dir)
set -e
L=/export/home/kaziz/g05/baselines/lingbot-vla-v2; G=/export/home/kaziz/g05; PY=$L/.venv/bin/python
export PATH=$L/.venv/bin:$HOME/.local/bin:$PATH TMPDIR=$G/tmp HF_HOME=$G/hf_cache TOKENIZERS_PARALLELISM=false
export QWEN3VL_PATH=$G/models/Qwen3-VL-4B-Instruct MASTER_PORT=$(( 29000 + SLURM_JOB_ID % 1000 ))
# torchcodec (video decoding in LingBot's LeRobot loader) needs FFmpeg's shared libraries: our FFmpeg 7.1 build; the
# A100 node's system libstdc++ is new enough for it (unlike the T4 nodes, see env_g05.sh)
export LD_LIBRARY_PATH=$G/ffmpeg/lib:${LD_LIBRARY_PATH:-}
cd $L
echo "node=$(hostname) gpus=$CUDA_VISIBLE_DEVICES mode=${MODE:?}"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
NGPU=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l); MB=${MB:-16}
cp $G/jobs/lingbot_robocasa365.yaml configs/robot_configs/robocasa365.yaml
if [ "$MODE" = prep ]; then
    $PY -c "import flash_attn; print('flash_attn', flash_attn.__version__)" || uv pip install --python $PY \
        https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
    [ -f $G/models/moge-2-vitb-normal/model.pt ] || HF_TOKEN=$(cat $HOME/.cache/huggingface/token) HF_HUB_DISABLE_XET=1 $PY -c "
from huggingface_hub import hf_hub_download
print(hf_hub_download('Ruicheng/moge-2-vitb-normal', 'model.pt', local_dir='$G/models/moge-2-vitb-normal'))"
    export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
    mkdir -p assets/norm_stats
    [ -f assets/norm_stats/robocasa365.json ] || bash train.sh scripts/compute_norm_stats.py ./configs/vla/norm_compute/post_data.yaml \
        --data.data_name robocasa365 --data.train_path $G/data/rc365_v30/target50 --data.norm_path assets/norm_stats/robocasa365.json \
        --data.data_ratio_for_norm_compute 0.01
    ls -la assets/norm_stats/robocasa365.json
    $PY $G/jobs/135_lingbot_cfg.py $NGPU $MB ${STEPS:-30} ${STEPS:-30} $G/tmp/lingbot_rc365_dryrun  # checkpoint at the end: inference smoke test
    START=$(date +%s)
    bash train.sh tasks/vla/train_lingbotvla.py ./configs/vla/robocasa365/robocasa365.yaml \
        --data.norm_stats_file assets/norm_stats/robocasa365.json
    echo "dry run: $(( $(date +%s) - START )) s for ${STEPS:-30} steps (incl. model loading)"
    find $G/tmp/lingbot_rc365_dryrun -maxdepth 4 -name hf_ckpt -o -maxdepth 4 -name lingbotvla_cli.yaml | head
elif [ "$MODE" = auto1 ]; then
    # one GPU: bf16 weights + LingBot's AnyPrecisionAdamW (bf16 state). Probe configurations for 20 steps, from the most
    # faithful (Kahan compensation, micro batch 4) down, then train with the first that fits, in this same allocation.
    export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OPT=anyprecision_adamw MIXED=0 LB_AUTOCAST=1
    export TMPDIR=$(mktemp -d /tmp/lb_${SLURM_JOB_ID}_XXXX)  # node-local: multiprocessing temp dirs on NFS fail to clean up at exit
    O=lingbotvla/optim/optimizer.py  # patch: LB_KAHAN=0 turns the compensation buffer off (saves 2 bytes/parameter)
    grep -q LB_KAHAN $O || sed -i 's/^import re$/import os\nimport re/; s/optim = AnyPrecisionAdamW(param_groups, lr, betas, eps, weight_decay)$/optim = AnyPrecisionAdamW(param_groups, lr, betas, eps, weight_decay, use_kahan_summation=os.environ.get("LB_KAHAN", "1") == "1")/' $O
    grep -n "LB_KAHAN" $O
    # patch: with bf16 weights (no mixed precision) the fp32 batch tensors meet bf16 matmuls; LB_AUTOCAST=1 runs the forward
    # pass under bf16 autocast, as the mixed-precision recipe does
    T=tasks/vla/train_lingbotvla.py
    grep -q LB_AUTOCAST $T || sed -i 's/^\( *\)with model_fwd_context:$/\1with model_fwd_context, torch.autocast("cuda", dtype=torch.bfloat16, enabled=os.environ.get("LB_AUTOCAST") == "1"):/' $T
    grep -n "LB_AUTOCAST" $T
    CHOSEN=""
    for cfg in "1 4 1" "0 4 1" "0 2 1"; do  # Kahan, micro batch, torch.compile (eager bf16 hits a dtype error: keep compile on)
        set -- $cfg; export LB_KAHAN=$1 COMPILE=$3; P_OUT=$G/tmp/lb_probe_k$1_mb$2_c$3; rm -rf $P_OUT
        $PY $G/jobs/135_lingbot_cfg.py 1 $2 20 1000000 $P_OUT
        START=$(date +%s)
        bash train.sh tasks/vla/train_lingbotvla.py ./configs/vla/robocasa365/robocasa365.yaml \
            --data.norm_stats_file assets/norm_stats/robocasa365.json > $P_OUT.log 2>&1 || true
        if grep -q "20/20" $P_OUT.log && ! grep -q "OutOfMemoryError" $P_OUT.log; then  # (healthy runs print warnings containing "Traceback")
            echo "probe kahan=$1 mb=$2 compile=$3: OK, $(( $(date +%s) - START )) s for 20 steps incl. loading"; CHOSEN="$1 $2 $3"; break
        fi
        echo "probe kahan=$1 mb=$2 compile=$3: failed ($(grep -m1 -o -e 'OutOfMemoryError' -e '[A-Za-z]*Error: .\{0,100\}' $P_OUT.log))"
    done
    [ -n "$CHOSEN" ] || { echo "no configuration fits one GPU"; exit 1; }
    set -- $CHOSEN; export LB_KAHAN=$1 COMPILE=$3
    ( while sleep 1800; do ls -d ${OUT:?}/checkpoints/global_step_* 2>/dev/null | sort -V | head -n -2 | xargs -r rm -rf; done ) &  # keep the newest 2
    $PY $G/jobs/135_lingbot_cfg.py 1 $2 ${STEPS:-200000} ${SAVE:-5000} $OUT
    bash train.sh tasks/vla/train_lingbotvla.py ./configs/vla/robocasa365/robocasa365.yaml \
        --data.norm_stats_file assets/norm_stats/robocasa365.json
else
    export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
    $PY $G/jobs/135_lingbot_cfg.py $NGPU $MB ${STEPS:?} ${SAVE:-2500} ${OUT:?}
    bash train.sh tasks/vla/train_lingbotvla.py ./configs/vla/robocasa365/robocasa365.yaml \
        --data.norm_stats_file assets/norm_stats/robocasa365.json
fi
