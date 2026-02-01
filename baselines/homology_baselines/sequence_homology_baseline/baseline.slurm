#!/bin/bash

#SBATCH --job-name=sequence_homology_baseline
#SBATCH --output=output_sequence_homology_baseline.txt  
#SBATCH --error=error_sequence_homology_baseline.txt
#SBATCH --partition=cpu-killable
#SBATCH --ntasks=1               
#SBATCH --cpus-per-task=128
#SBATCH --mem=500000                      
#SBATCH --time=2880    
#SBATCH --mail-type=ALL
#SBATCH --mail-user=annab4@mail.tau.ac.il              

source /home/iscb/wolfson/annab4/miniconda3/etc/profile.d/conda.sh
conda activate struct_bio   

folds=(4 5)

for f in "${folds[@]}"
do
    python /home/iscb/wolfson/annab4/catalytic-sites-annotation/sequence_homology_baseline/baseline.py --cv_fold ${f}
done
