#!/bin/bash

#SBATCH --job-name=homology_baseline_f5
#SBATCH --output=output_homology_baseline_f5.txt  
#SBATCH --error=error_homology_baseline_f5.txt
#SBATCH --partition=cpu-killable
#SBATCH --ntasks=1               
#SBATCH --cpus-per-task=256
#SBATCH --mem=500000                      
#SBATCH --time=4320   
#SBATCH --mail-type=ALL
#SBATCH --mail-user=annab4@mail.tau.ac.il              

source /home/iscb/wolfson/annab4/miniconda3/etc/profile.d/conda.sh
conda activate struct_bio   

folds=(5)

for f in "${folds[@]}"
do
    python /home/iscb/wolfson/annab4/catalytic-sites-annotation/structural_homology_baseline/baseline.py --cv_fold ${f}
done
