#!/bin/bash

#SBATCH --job-name=propagate_msa_pools
#SBATCH --output=output_propagate_msa_pools.txt  
#SBATCH --error=error_propagate_msa_pools.txt
#SBATCH --partition=cpu-killable
#SBATCH --ntasks=1               
#SBATCH --cpus-per-task=128
#SBATCH --mem=500000                      
#SBATCH --time=1440    
#SBATCH --mail-type=ALL
#SBATCH --mail-user=annab4@mail.tau.ac.il              

source /home/iscb/wolfson/annab4/miniconda3/etc/profile.d/conda.sh
conda activate struct_bio                   

thresholds=(0.3 0.5)

for thr in "${thresholds[@]}"
do
    #python /home/iscb/wolfson/annab4/catalytic-sites-annotation/msa_propagation/propagate.py --threshold ${thr} --alignment_type 1
    python /home/iscb/wolfson/annab4/catalytic-sites-annotation/msa_propagation/propagate_pool.py --threshold ${thr} --alignment_type 2 --workers 128
done 
