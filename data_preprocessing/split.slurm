#!/bin/bash

#SBATCH --job-name=split_by_ec_number_mol
#SBATCH --output=output_split_by_ec_number_mol.txt  
#SBATCH --error=error_split_by_ec_number_mol.txt
#SBATCH --partition=cpu-mol
#SBATCH --ntasks=1               
#SBATCH --cpus-per-task=128
#SBATCH --mem=500000                      
#SBATCH --time=1440    
#SBATCH --mail-type=ALL
#SBATCH --mail-user=annab4@mail.tau.ac.il              

source /home/iscb/wolfson/annab4/miniconda3/etc/profile.d/conda.sh
conda activate datascience                    

python /home/iscb/wolfson/annab4/catalytic-sites-annotation/preprocessing/4_split_by_ec_number.py
