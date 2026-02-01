#!/bin/bash

#SBATCH --job-name=align_and_propagate
#SBATCH --output=output_align_and_propagate.txt  
#SBATCH --error=error_align_and_propagate.txt
#SBATCH --partition=cpu-killable
#SBATCH --ntasks=1               
#SBATCH --cpus-per-task=128
#SBATCH --mem=500000                      
#SBATCH --time=1440    
#SBATCH --mail-type=ALL
#SBATCH --mail-user=annab4@mail.tau.ac.il              

source /home/iscb/wolfson/annab4/miniconda3/etc/profile.d/conda.sh
conda activate struct_bio                   

thresholds=(0.2)

for thr in "${thresholds[@]}"
do
    python /home/iscb/wolfson/annab4/catalytic-sites-annotation/msa_propagation/align_and_propagate.py --threads 128 --threshold ${thr}
done 
