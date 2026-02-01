#!/bin/bash

#SBATCH --job-name=prepare_3di_db
#SBATCH --output=output_prepare_3di_db.txt  
#SBATCH --error=error_prepare_3di_db.txt
#SBATCH --partition=cpu-mol
#SBATCH --ntasks=1               
#SBATCH --cpus-per-task=128
#SBATCH --mem=500000                      
#SBATCH --time=1440    
#SBATCH --mail-type=ALL
#SBATCH --mail-user=annab4@mail.tau.ac.il              

source /home/iscb/wolfson/annab4/miniconda3/etc/profile.d/conda.sh
conda activate struct_bio                   

python /home/iscb/wolfson/annab4/catalytic-sites-annotation/msa_propagation/prepare_3di_db.py
