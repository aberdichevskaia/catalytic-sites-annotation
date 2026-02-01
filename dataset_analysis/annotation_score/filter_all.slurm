#!/bin/bash

#SBATCH --job-name=filter_all
#SBATCH --output=output_filter_all.txt  
#SBATCH --error=error_filter_all.txt
#SBATCH --partition=cpu-mol
#SBATCH --ntasks=1               
#SBATCH --cpus-per-task=64
#SBATCH --mem=500000                      
#SBATCH --time=14400    
#SBATCH --mail-type=ALL
#SBATCH --mail-user=annab4@mail.tau.ac.il              

source /home/iscb/wolfson/annab4/miniconda3/etc/profile.d/conda.sh
conda activate datascience                  

python /home/iscb/wolfson/annab4/catalytic-sites-annotation/annotation_score/filter_all.py
