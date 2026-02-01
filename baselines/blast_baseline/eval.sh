python /home/iscb/wolfson/annab4/catalytic-sites-annotation/blast_baseline/eval_blast_pr_curves.py \
  --splits_glob "/home/iscb/wolfson/annab4/DB/all_proteins/cross_validation_chem/weight_based_v9/split*.txt" \
  --blast_dir   "/home/iscb/wolfson/annab4/DB/all_proteins/blast_baseline" \
  --out_dir     "/home/iscb/wolfson/annab4/DB/all_proteins/blast_baseline_eval" \
  --model_name  "BLAST" \
  --title_prefix "BLAST PR"
