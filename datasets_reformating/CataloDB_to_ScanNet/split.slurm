#!/bin/bash

#SBATCH --job-name=cross_validation_split_pool
#SBATCH --output=output_cross_validation_split_pool.txt  
#SBATCH --error=error_cross_validation_split_pool.txt
#SBATCH --partition=cpu-killable
#SBATCH --ntasks=1               
#SBATCH --cpus-per-task=128
#SBATCH --mem=500000                      
#SBATCH --time=1440    
#SBATCH --mail-type=ALL
#SBATCH --mail-user=annab4@mail.tau.ac.il              

source /home/iscb/wolfson/annab4/miniconda3/etc/profile.d/conda.sh
conda activate struct_bio                    

python /home/iscb/wolfson/annab4/catalytic-sites-annotation/datasets_reformating/CataloDB_to_ScanNet/split_and_write.py \
  --protein_table_json /home/iscb/wolfson/annab4/datasets/CataloDB/preprpocessed/protein_table.json \
  --train_raw_txt /home/iscb/wolfson/annab4/datasets/CataloDB/preprpocessed/train_raw.txt \
  --test_txt /home/iscb/wolfson/annab4/datasets/CataloDB/preprpocessed/test.txt \
  --cluster1_tsv /home/iscb/wolfson/annab4/datasets/CataloDB/fastas/cluster_level_1_cluster.tsv \
  --cluster2_tsv /home/iscb/wolfson/annab4/datasets/CataloDB/fastas/cluster_level_2_cluster.tsv \
  --out_dir /home/iscb/wolfson/annab4/datasets/CataloDB/final_splits \
  --n_splits 5 \
  --seed 42
