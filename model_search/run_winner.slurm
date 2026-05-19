#!/bin/bash
#SBATCH --job-name=esm2_winner
#SBATCH --output=logs/%A_%a.out
#SBATCH --error=logs/%A_%a.err
#SBATCH --partition=killable
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=500000
#SBATCH --time=1440
#SBATCH --mail-type=ALL,TIME_LIMIT_80
#SBATCH --mail-user=annab4@mail.tau.ac.il

set -euo pipefail

source /home/iscb/wolfson/annab4/miniconda3/etc/profile.d/conda.sh
export KERAS_BACKEND=jax
conda activate scannet_keras3

export XLA_FLAGS="--xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0"
export TF_DETERMINISTIC_OPS=1

CSV_PATH="/home/iscb/wolfson/annab4/catalytic-sites-annotation/model_search/runs_winner.csv"
RUNNER_PATH="/home/iscb/wolfson/annab4/catalytic-sites-annotation/model_search/run_from_csv.py"
TRAIN_SCRIPT="/home/iscb/wolfson/annab4/main_scannet/ScanNet_Ub/baselines/esm2_grid_search.py"

mkdir -p logs

python "${RUNNER_PATH}" \
  --csv "${CSV_PATH}" \
  --index "${SLURM_ARRAY_TASK_ID}" \
  --script "${TRAIN_SCRIPT}"