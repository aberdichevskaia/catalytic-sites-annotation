python "$(dirname "$0")/precompute_MSAs.py" \
  --structures_dir /home/iscb/wolfson/annab4/Data/PDB_Human_Isoforms \
  --out_msa_dir    /home/iscb/wolfson/annab4/Data/MSA_Human_Isoforms \
  --cores_per_job  4 \
  --only_missing \
  --chains A
