#!/bin/bash

#SBATCH --job-name=propagate_msa
#SBATCH --output=output_propagate_msa.txt  
#SBATCH --error=error_propagate_msa.txt
#SBATCH --partition=cpu-killable
#SBATCH --ntasks=1               
#SBATCH --cpus-per-task=4
#SBATCH --mem=500000                      
#SBATCH --time=1440    
#SBATCH --mail-type=ALL
#SBATCH --mail-user=annab4@mail.tau.ac.il              

source /home/iscb/wolfson/annab4/miniconda3/etc/profile.d/conda.sh
conda activate struct_bio                   

thresholds=(0.2 0.3)

for thr in "${thresholds[@]}"
do
    #python /home/iscb/wolfson/annab4/catalytic-sites-annotation/msa_propagation/propagate.py --threshold ${thr} --alignment_type 1
    python /home/iscb/wolfson/annab4/catalytic-sites-annotation/msa_propagation/propagate.py --threshold ${thr} --alignment_type 2
done 
