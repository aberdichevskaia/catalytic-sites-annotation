mkdir -p /home/iscb/wolfson/annab4/outputs/isoform_groups_thr35

python /home/iscb/wolfson/annab4/catalytic-sites-annotation/isoforms/create_files_for_predictions.py \
  --csv /home/iscb/wolfson/annab4/outputs/interesting_isoforms_thr35.csv \
  --pdb-dir /home/iscb/wolfson/annab4/Data/PDB_Human_Isoforms \
  --out-dir /home/iscb/wolfson/annab4/outputs/isoform_groups_thr35
