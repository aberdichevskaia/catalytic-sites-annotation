mmseqs easy-cluster /home/iscb/wolfson/annab4/datasets/CataloDB/fastas/train.fasta cluster_level_1 tmp_mmseqs_l1 --min-seq-id 0.90 -c 0.80 --cov-mode 0

mmseqs easy-cluster /home/iscb/wolfson/annab4/datasets/CataloDB/fastas/cluster_level_1_rep_seq.fasta cluster_level_2 tmp_mmseqs_l2 --min-seq-id 0.40 -c 0.30 --cov-mode 0
