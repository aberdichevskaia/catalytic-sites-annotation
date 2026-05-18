# Catalytic Sites Annotation

Pipeline for building a dataset of enzyme catalytic sites and evaluating
binding-site prediction models against it.

## What this does

1. Fetches enzyme records from UniProt, maps catalytic residue annotations
   to PDB chains, and stores per-residue labels.
2. Clusters sequences (MMseqs2), groups by EC number, and builds a
   train / validation / test split and a 5-fold cross-validation split,
   both balanced by chemical class of the catalytic residues.
3. Computes ESM2 embeddings for all sequences (optional GPU step).
4. Runs ScanNet-based predictions on the resulting splits.
5. Evaluates predictions with AUCPR metrics and compares against two
   homology baselines (sequence Smith–Waterman and structural 3Di alignment).
6. Applies the trained model to the full human proteome (AlphaFold structures)
   and analyses isoform-level variation in predicted binding-site scores.

## Repository layout

```
data_preprocessing/          Steps 1–4: UniProt → batch pickles → split CSVs
cross_validation_dataset_building/  5-fold CV split (split.py)
precompute_esm2_embeddings/  ESM2 embedding cache scripts
inference_scripts/           ScanNet prediction runner scripts
prediction_results_analysis/ AUCPR evaluation and plots
dataset_analysis/            Dataset statistics and sanity checks
baselines/
  blast_baseline/            BLAST-based baseline
  homology_baselines/
    sequence_homology_baseline/   Smith–Waterman (AA) baseline
    structural_homology_baseline/ 3Di alignment baseline
downstream_tasks/
  human_proteome_prediction/ Screen AlphaFold Human proteome
  isoforms_analysis/         Isoform comparison pipeline
utils/                       Shared utilities (EC numbers, alignment, data loading)
```

## Setup

```bash
pip install -r requirements.txt
```

For ScanNet-based inference, clone the ScanNet repository and set:

```bash
export SCANNET_ROOT=/path/to/ScanNet_Ub
```

Copy `config.example.yaml` to `config.yaml` and fill in your local paths.
`config.yaml` is gitignored and never committed.

## Running the pipeline

### Step 1 — Preprocess UniProt JSON

```bash
python data_preprocessing/1_preprocessing_all.py \
    --input_file   /path/to/filtered_all_protein.json \
    --output_dir   /path/to/batches \
    --pdb_dir      /path/to/PDB_files \
    --num_batches  100
```

### Step 2 — Join batch outputs into a single table

```bash
python data_preprocessing/2_join_data.py \
    --batches_dir  /path/to/batches \
    --output_table /path/to/all_protein_table.json \
    --output_fasta /path/to/all_protein_sequences.fasta
```

### Step 3 — Fix EC numbers (sequence clustering cross-reference)

```bash
python data_preprocessing/3_ec_numbers_quick_fix.py \
    --protein_table    /path/to/all_protein_table.json \
    --cluster_level_1  /path/to/cluster_level_1_cluster.tsv \
    --cluster_level_2  /path/to/cluster_level_2_cluster.tsv \
    --output_table     /path/to/all_protein_table_modified.json
```

### Step 4 — Build train / val / test split

```bash
python data_preprocessing/4_split_by_ec_number.py \
    --cluster_level_1  /path/to/cluster_level_1_cluster.tsv \
    --cluster_level_2  /path/to/cluster_level_2_cluster.tsv \
    --protein_table    /path/to/all_protein_table_modified.json \
    --batches_dir      /path/to/batches \
    --output_dir       /path/to/splitted \
    --output_csv       /path/to/dataset.csv
```

### Step 5 — Build 5-fold cross-validation split

```bash
python cross_validation_dataset_building/split.py \
    --cluster_level_1  /path/to/cluster_level_1_cluster.tsv \
    --cluster_level_2  /path/to/cluster_level_2_cluster.tsv \
    --protein_table    /path/to/all_protein_table_modified.json \
    --batches_dir      /path/to/batches \
    --pdb_dir          /path/to/PDB_files \
    --output_dir       /path/to/cross_validation
```

### Step 6 — Precompute ESM2 embeddings (optional, GPU recommended)

```bash
python precompute_esm2_embeddings/precompute_esm2.py \
    --split_txts /path/to/cross_validation/split{1..5}.txt \
    --out_dir    /path/to/esm2_cache
```

### Step 7 — Run homology baselines (5-fold CV)

Sequence baseline:

```bash
for fold in 1 2 3 4 5; do
  python baselines/homology_baselines/sequence_homology_baseline/baseline.py \
      --cv_fold    $fold \
      --splits_json /path/to/3Di_DB_splits.json \
      --out_base   /path/to/sequence_homology_results
done
```

Structural baseline (3Di):

```bash
for fold in 1 2 3 4 5; do
  python baselines/homology_baselines/structural_homology_baseline/baseline.py \
      --cv_fold    $fold \
      --splits_json /path/to/3Di_DB_splits.json \
      --out_base   /path/to/structural_homology_results
done
```

### Step 8 — Evaluate results

```bash
python prediction_results_analysis/evaluate_results.py \
    --results_dir /path/to/predictions \
    --dataset_csv /path/to/dataset.csv \
    --out_dir     /path/to/evaluation
```

## Dependencies

See `requirements.txt`. Key packages:

| Package | Purpose |
|---------|---------|
| biopython | PDB parsing, sequence alignment |
| biotite | 3Di structure alphabet, Smith–Waterman alignment |
| networkx | Graph-based split construction |
| scikit-learn | AUCPR metrics |
| torch + fair-esm | ESM2 embeddings (optional) |

The ScanNet model itself is an external dependency — set `SCANNET_ROOT`
to the path of your ScanNet clone before running inference or human
proteome prediction scripts.
