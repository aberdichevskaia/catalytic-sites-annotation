#!/bin/bash

#SBATCH --job-name=find_interesting_isoforms_thr35
#SBATCH --output=output_find_interesting_isoforms_thr35.txt  
#SBATCH --error=error_find_interesting_isoforms_thr35.txt
#SBATCH --partition=cpu-killable
#SBATCH --ntasks=1               
#SBATCH --cpus-per-task=16
#SBATCH --mem=500000                      
#SBATCH --time=2880    
#SBATCH --mail-type=ALL
#SBATCH --mail-user=annab4@mail.tau.ac.il   

source /home/iscb/wolfson/annab4/miniconda3/etc/profile.d/conda.sh
conda activate struct_bio

python /home/iscb/wolfson/annab4/catalytic-sites-annotation/downstream_tasks/isoforms_analysis/find_interesting_isoforms.py \
    --in_csv /home/iscb/wolfson/annab4/outputs/human_isoforms_screen_2.csv \
    --out_csv /home/iscb/wolfson/annab4/outputs/interesting_isoforms_thr58.csv \
    --thr 58 \
    --pdb_dir /home/iscb/wolfson/annab4/Data/PDB_Human_Isoforms
