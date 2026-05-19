#!/bin/bash
set -u  # падать на неинициализированных переменных
# set -e  # (опционально) падать на первой ошибке; если хочешь прогнать все фолды — не ставь

source /home/iscb/wolfson/annab4/miniconda3/etc/profile.d/conda.sh
export KERAS_BACKEND=jax
#conda activate scannet_keras3
conda activate scannet_esm2_py311

python_script="/home/iscb/wolfson/annab4/main_scannet/ScanNet_Ub/predict_bindingsites.py"
path_to_proteins="/home/iscb/wolfson/annab4/catalytic-sites-annotation/inference_scripts/to_predict.txt"

logdir="./logs_predict"
mkdir -p "$logdir"

echo "=== Launch ==="
mode="cv_catalytic"
additional="--use_ESM2 --esm2_dir /home/iscb/wolfson/annab4/Data/Human_ESM2_150M_layer30"
prediction_folder="--predictions_folder /home/iscb/wolfson/annab4/human_predictions_ESM2/"

python "$python_script" "$path_to_proteins" --mode "$mode" ${additional} ${prediction_folder} --molstar
echo "=== Finished ==="
