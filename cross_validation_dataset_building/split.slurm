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

python /home/iscb/wolfson/annab4/catalytic-sites-annotation/cross_validation/split_with_pools.py
