#!/bin/bash -l
#SBATCH -J g05_download
#SBATCH -o /export/home/kaziz/g05/logs/slurm-%x-%j.out
#SBATCH -p cpu-all
#SBATCH -c 4
#SBATCH --mem 16000MB
#SBATCH --time 08:00:00
# g05-base is fetched by 00b_download_base.sh. hf 1.31 parses extra positional patterns as filenames:
# pass exactly one pattern per --include.
source /export/home/kaziz/g05/env_g05.sh
export HF_XET_HIGH_PERFORMANCE=1
R=$G05_ROOT
HF=/export/home/kaziz/motion/miniconda3/envs/ardy/bin/hf    # CLI only; no torch import
declare -A RC
run() { local n=$1; shift; echo ">>> [$n] $(date +%T)"; "$@"; RC[$n]=$?; echo "<<< [$n] rc=${RC[$n]} $(date +%T)"; }
echo "node: $(hostname)"; curl -sfI --max-time 15 https://huggingface.co >/dev/null || { echo "NO INTERNET ON COMPUTE NODE"; exit 3; }
run g05_libero $HF download OpenGalaxea/G05 --include "g05-libero/*" --include "action_tokenizer.pt" --include "qwen3_5_2b_base_processor/*" --local-dir $PROJECT_ROOT/checkpoints
run lawam_pretrain $HF download jialei02/lawam_pretrain --local-dir $LAWAM_ROOT/results/Checkpoints/pretrain/lawam_pretrain
run lawam_lam $HF download jialei02/lawam_lam --local-dir $LAWAM_ROOT/latent_action_model/logs/dino_large_vae/lam_release
run libero_data $HF download jialei02/libero_merged_no_noops_20hz --repo-type dataset --local-dir $R/data/libero_merged_no_noops_20hz
run dinov3 $HF download facebook/dinov3-vitb16-pretrain-lvd1689m --local-dir $LAWAM_ROOT/weights/dinov3-vitb16-pretrain-lvd1689m
# eval bundle layout expected by g05-libero/.hydra/config.yaml
ln -sfn ../qwen3_5_2b_base_processor $PROJECT_ROOT/checkpoints/g05-libero/hf_processor
ln -sfn ../action_tokenizer.pt $PROJECT_ROOT/checkpoints/g05-libero/action_tokenizer.pt
mkdir -p $PROJECT_ROOT/data/libero $PROJECT_ROOT/data/stats
ln -sfn $R/data/libero_merged_no_noops_20hz $PROJECT_ROOT/data/libero/libero_merged_no_noops
find $PROJECT_ROOT/checkpoints $LAWAM_ROOT/results/Checkpoints $LAWAM_ROOT/latent_action_model/logs $LAWAM_ROOT/weights $R/data -path '*/.cache' -prune -o -type f -size +100M -printf '%s %p\n' 2>/dev/null | awk '{printf "%8.2f GB  %s\n", $1/1e9, $2}'
echo "== summary"; FAIL=0; for k in g05_libero lawam_pretrain lawam_lam libero_data dinov3; do echo "$k rc=${RC[$k]}"; [[ $k != dinov3 && ${RC[$k]} != 0 ]] && FAIL=1; done
[[ ${RC[dinov3]} != 0 ]] && echo "NOTE: dinov3 failed (expected while access is pending review)"
exit $FAIL
