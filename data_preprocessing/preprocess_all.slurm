#!/bin/bash

#SBATCH --job-name=preprocess_all
#SBATCH --output=output_preprocess_all.txt  
#SBATCH --error=error_preprocess_all.txt
#SBATCH --partition=cpu-mol
#SBATCH --ntasks=1               
#SBATCH --cpus-per-task=64
#SBATCH --mem=500000                      
#SBATCH --time=14400    
#SBATCH --mail-type=ALL
#SBATCH --mail-user=annab4@mail.tau.ac.il              

source /home/iscb/wolfson/annab4/miniconda3/etc/profile.d/conda.sh
conda activate struct_bio                    

python /home/iscb/wolfson/annab4/catalytic-sites-annotation/data_preprocessing/preprocessing_all.py \
  --input_file    /home/iscb/wolfson/annab4/DB/all_proteins/uniprot_data.json \
  --output_dir    /home/iscb/wolfson/annab4/DB/all_proteins/batches \
  --pdb_dir       /home/iscb/wolfson/annab4/DB/PDB \
  --protein_table /home/iscb/wolfson/annab4/DB/all_proteins/protein_table.json \
  --protein_fasta /home/iscb/wolfson/annab4/DB/all_proteins/protein_sequences.fasta \
  --num_batches   100
