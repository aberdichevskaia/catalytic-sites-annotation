mkdir -p /home/iscb/wolfson/annab4/outputs/isoform_groups

python "$(dirname "$0")/create_files_for_predictions.py" \
  --csv     /home/iscb/wolfson/annab4/outputs/isoforms_analysis.csv \
  --column  isoform \
  --pdb-dir /home/iscb/wolfson/annab4/Data/PDB_Human_Isoforms \
  --out-dir /home/iscb/wolfson/annab4/outputs/isoform_groups
