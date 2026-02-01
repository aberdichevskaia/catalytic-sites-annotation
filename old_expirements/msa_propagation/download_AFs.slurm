#!/bin/bash

#SBATCH --job-name=download_structures
#SBATCH --output=output_download_structures.txt  
#SBATCH --error=error_download_structures.txt
#SBATCH --partition=cpu-killable
#SBATCH --ntasks=1               
#SBATCH --cpus-per-task=8
#SBATCH --mem=500000                      
#SBATCH --time=1440    
#SBATCH --mail-type=ALL
#SBATCH --mail-user=annab4@mail.tau.ac.il              

source /home/iscb/wolfson/annab4/miniconda3/etc/profile.d/conda.sh
conda activate struct_bio                   

python /home/iscb/wolfson/annab4/catalytic-sites-annotation/msa_propagation/download_AFs.py
