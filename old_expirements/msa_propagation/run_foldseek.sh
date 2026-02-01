# === 1) create structures database ===
foldseek createdb \
    /home/iscb/wolfson/annab4/DB/all_proteins/MSA_propagation/AF_structures/ \
    structDB \
    --threads 32

# === 2) clustering ===

#option 1: 3Di+AA Gotoh-Smith-Waterman (local, default)
foldseek cluster structDB_new clusterRes tmpDir_ --alignment-type 2 -c 0.5 --cov-mode 2 --min-seq-id 0.3 --threads 32


#OR option 2: TMalign (global)
foldseek cluster structDB_new clusterRes_TMAlign tmpDir__ --alignment-type 1 -c 0.5 --cov-mode 2 --min-seq-id 0.3 --threads 128

#adding customized penalties: --gap-open aa:3,nucl:10 --gap-extend aa:2,nucl:1


# === 3) generating MSA (A3M) ===
#3di version: 
foldseek result2msa structDB_new structDB_new clusterRes msaDB_3di --filter-msa 0 --msa-format-mode 6 --threads 64 

#TMAlign version:
foldseek result2msa structDB_new structDB_new clusterRes_TMAlign msaDB_TMAlign --filter-msa 0 --msa-format-mode 6 --threads 64 

# === 4) unpack .a3m files ===
foldseek unpackdb msaDB_3di msa_3di --unpack-suffix .a3m --unpack-name-mode 1
foldseek unpackdb msaDB_TMAlign msa_TMAlign --unpack-suffix .a3m --unpack-name-mode 1


# === create .tsv with clusters ===
foldseek createtsv structDB_new  clusterRes cluster_members_3di.tsv

