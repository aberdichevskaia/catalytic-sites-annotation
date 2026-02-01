# evaluate_results.py — CV merge, multi-seed evaluation, case inspection, and SNR

This directory contains a single command-line tool, `evaluate_results.py`, that helps you:

1) **Merge** 5-fold cross-validation result pickles into one “merged” set  
2) **Evaluate + plot** AUCPR and MaxPrecision@k (aka “F@k” in some older scripts)  
3) **Evaluate multi-seed runs**: whiskers over seeds for MaxPrecision@k, and an averaged PR curve with per-seed curves overlaid  
4) **Inspect best/worst cases** (top-N proteins) by FP/FN in either **threshold** or **top-k** mode  
5) Compute **SNR** (Signal-to-Noise Ratio) for:
   - AUCPR: **one number per seed**
   - MaxPrecision@k: **SNR(k) curve** (one per k)

The tool also saves a **per-chain table** (`per_chain.csv`) so you can debug, stratify, join with metadata, and build custom plots later.

---

## Table of Contents

- [Installation](#installation)
- [Input data formats](#input-data-formats)
  - [Per-fold result pickles](#per-fold-result-pickles)
  - [dataset.csv](#datasetcsv)
  - [split*.txt label files](#splittxt-label-files)
  - [protein_table.json](#protein_tablejson)
- [Concepts and definitions](#concepts-and-definitions)
  - [MaxPrecision@k](#maxprecisionk)
  - [Weighted AUCPR](#weighted-aucppr)
  - [SNR](#snr)
- [Directory layout assumptions](#directory-layout-assumptions)
- [CLI overview](#cli-overview)
  - [`merge` subcommand](#merge-subcommand)
  - [`seeds` subcommand](#seeds-subcommand)
  - [`cases` subcommand](#cases-subcommand)
- [Outputs](#outputs)
- [How IDs become “UniProt IDs” and full names](#how-ids-become-uniprot-ids-and-full-names)
- [Troubleshooting](#troubleshooting)
- [Examples (recommended workflows)](#examples-recommended-workflows)

---

## Installation

This is a pure-Python script. It requires:

- Python 3.9+
- `numpy`
- `pandas`
- `matplotlib`
- `scikit-learn`

Example:

```bash
pip install numpy pandas matplotlib scikit-learn
````

---

## Input data formats

### Per-fold result pickles

Each fold produces pickles like:

* `train_results.pkl`
* `validation_results.pkl`
* `test_results.pkl`

Expected schema (minimum required keys):

```python
{
  "subset": str,             # e.g. "test"
  "model_name": str,         # optional but recommended
  "labels": list[list[int]], # per-chain 1D vectors (0/1) or int labels per residue
  "predictions": list[list[float]], # per-chain 1D scores per residue
  "weights": list[float],    # per-chain weights (e.g. W_Sequence or W_Structure or combined)
  "ids": list[Any],          # per-chain IDs (best: Sequence_ID strings)
  "splits": list[str],       # e.g. split1..split5 labels
  "batch_size": int          # optional
}
```

**Important:** `labels[i]` and `predictions[i]` must be **1D arrays** (per-chain residue-level vectors).
If your model outputs multi-class scores, you must define how to convert them into a single score per residue (most setups use “last class” as positive class).

---

### dataset.csv

Used to map chain IDs to human-readable labels.

Minimum recommended columns:

* `Sequence_ID` (e.g. `A0A1L8G2K9_A`)
* `full_name` (e.g. “DNA-dependent metalloprotease SPRTN”)
* `EC_number` (optional, but useful)
* plus your weights columns (optional)

Example row:

```
Sequence_ID,Set_Type,EC_number,Component_ID,full_name,W_Sequence,W_Structure
A0A1L8G2K9_A,split1,3.4.24.-,22,DNA-dependent metalloprotease SPRTN,1.0,1.0
```

---

### split*.txt label files

These are used when you need the **sequence** (e.g., chemotype analysis, or residue inspection).

Format:

```
>SEQID
A 1 M 0
A 2 G 0
...
```

The tool typically parses:

* `SEQID`
* amino acid letters (e.g. `M`, `G`, …)
* (optionally) labels column if needed for consistency checks

---

### protein_table.json

Used to map to UniProt-level metadata. Example structure:

```json
{
  "A0A044RE18": {
    "uniprot_id": "A0A044RE18",
    "EC_number": "3.4.21.75",
    "full_name": "Endoprotease bli",
    "pdb_ids": [],
    ...
  }
}
```

If your `ids` are `Sequence_ID` like `A0A1L8G2K9_A`, then:

* `uniprot_id = Sequence_ID.split("_")[0]`

So the tool can:

* print UniProt IDs
* join with `protein_table.json`
* print full names and EC numbers

---

## Concepts and definitions

### MaxPrecision@k

This is the metric you care about (sometimes previously named `F@k` in older scripts).

For each chain (protein sequence) you take the **top-k** residues by predicted score. Let:

* `TP@k` = number of true catalytic residues among these top-k
* `npos` = total number of true catalytic residues in the chain

Then per-chain MaxPrecision@k is:

**MaxPrecision@k(chain) = TP@k / min(npos, k)**

If `npos = 0`, MaxPrecision@k is defined as **0.0** for that chain.

The reported dataset metric is the **weighted mean** over chains:

**MaxPrecision@k = sum_i w_i * MP@k_i / sum_i w_i**

This matches the “top-k normalized by min(#pos,k)” formula.

---

### Weighted AUCPR

To compute AUCPR, the tool flattens residue-level vectors across chains and repeats the chain weight for each residue:

* `y_flat = concat(labels[i])`
* `p_flat = concat(predictions[i])`
* `w_flat = concat( [w_i repeated len(chain_i) times] )`

Then it computes `precision_recall_curve(y_flat, p_flat, sample_weight=w_flat)` and `auc(recall, precision)`.

---

### SNR

We use the definition you described:

> **signal = improvement over a baseline**
> **noise = standard deviation** (over seeds)

For a scalar metric (AUCPR):

* Let `m_seed` be AUCPR for each seed
* Let `mean_model = mean(m_seed)`
* Let `std_model  = std(m_seed)` (over seeds)
* Let `baseline` be AUCPR of a baseline run (scalar), or mean baseline if baseline has seeds.

Then:

**SNR_AUCPR = (mean_model − baseline) / std_model**

For MaxPrecision@k (a curve):

* For each k, compute `MP@k` per seed → you get a distribution over seeds
* Define:

**SNR_MP(k) = (mean_model(k) − baseline(k)) / std_model(k)**

Notes:

* If `std_model == 0`, SNR is undefined/infinite; the tool typically reports `nan` or a very large number (implementation-dependent).
* SNR is meaningful only if you have **at least 2 seeds**.

---

## Directory layout assumptions

### CV folds

You typically have:

```
BASE_DIR/
  results_adaptive_cutoff_cv1/
    test_results.pkl
    validation_results.pkl
    train_results.pkl
  results_adaptive_cutoff_cv2/
    ...
  ...
```

You pass:

* `--base-dir BASE_DIR`
* `--run-tpl results_adaptive_cutoff_cv{fold}`
* `--folds 1 2 3 4 5`

### Seeds (versions)

You typically have:

```
RUN_ROOT/
  modelX_v1/
    (contains CV-fold dirs OR already-merged pickles)
  modelX_v2/
  modelX_v3/
```

You pass:

* `--template "modelX_v{version}"`
* `--versions 1 2 3 4 5 6`

What “template” points to depends on your setup:

* some projects store CV fold outputs inside each `*_v{seed}` folder
* some store already-merged outputs per seed

The tool supports both patterns as long as you point it to the correct paths.

---

## CLI overview

Run:

```bash
python evaluate_results.py -h
python evaluate_results.py merge -h
python evaluate_results.py seeds -h
python evaluate_results.py cases -h
```

> ⚠️ If you see “unrecognized arguments”, do **not** guess flags.
> Always run the subcommand help: `python evaluate_results.py cases -h`
> because flag names may differ between script versions (e.g. `--mode` vs `--case-mode`, `--fp-th` vs `--fp_th`).

Below is the *intended* interface.

---

## `merge` subcommand

### Purpose

* merge fold pickles into one merged dataset per subset
* compute metrics (AUCPR + MP@k curve)
* save plots
* save a `per_chain.csv` table

### Typical usage

```bash
python evaluate_results.py merge \
  --base-dir /path/to/base_dir \
  --run-tpl results_adaptive_cutoff_cv{fold} \
  --folds 1 2 3 4 5 \
  --subsets test validation train \
  --save-dir /path/to/output_merged \
  --dedupe-train mean \
  --max-k 20 \
  --save-plots \
  --tick-grid 0.1
```

### Key options

* `--dedupe-train {mean,median,first,None}`

  * Train sets often contain duplicates across folds (because “train” is “all folds except the held-out one”).
  * Dedupe aggregates duplicate chains by ID across folds.
* `--tick-grid 0.1`

  * Forces consistent axes ticks (especially useful for stacking plots in papers).
  * The tool typically sets x/y limits to `[0,1]` and uses ticks `[0.0, 0.1, ..., 1.0]`.

---

## `seeds` subcommand

### Purpose

Given multiple seeds (versions) of the same model, produce:

* MP@k curve with whiskers (mean ± std across seeds)
* PR curve: mean curve + faint per-seed curves
* SNR for AUCPR and SNR(k) for MP@k
* Summary JSON/CSV

### Typical usage

```bash
python evaluate_results.py seeds \
  --root /path/to/runs_root \
  --template "modelX_v{version}" \
  --versions 1 2 3 4 5 6 \
  --subset test \
  --max-k 20 \
  --out-dir /path/to/seed_summary \
  --baseline /path/to/baseline_merged/test.pkl \
  --tick-grid 0.1 \
  --pr-grid 0.01
```

### Notes

* `--subset` chooses which subset to analyze (`test` is the most common choice).
* PR averaging requires interpolation to a common recall grid; `--pr-grid 0.01` means recall points `{0.00, 0.01, ..., 1.00}`.
* Baseline can be:

  * a merged pickle (recommended), or
  * a run directory containing `test_results.pkl`, depending on how your tool resolves it.

---

## `cases` subcommand

### Purpose

Inspect per-protein “best/worst” cases by **FP/FN** in two modes:

1. **threshold mode**: global score threshold `tau`
2. **topk mode**: take top-k residues per chain; optionally count “confident FP” above `fp_th`

It prints the top-N chains and can export a CSV.

### How cases decides “positive prediction”

#### A) `threshold` mode

* predicted positive if `score >= tau`
* predicted negative if `score < tau`
  Then:
* TP: `y=1` and `score>=tau`
* FP: `y=0` and `score>=tau`
* FN: `y=1` and `score<tau`

#### B) `topk` mode

* predicted positives are exactly the indices of the **top-k scores**
  Then:
* `FP@k`: negatives inside top-k
* `FN@k`: positives not in top-k (≈ `npos − TP@k`)
* additionally, “confident FP”:

  * `CFP@k(fp_th)`: FP inside top-k with `score >= fp_th`

### Typical usage

**Top-k mode:**

```bash
python evaluate_results.py cases \
  --results /path/to/merged/test.pkl \
  --dataset-csv /path/to/dataset.csv \
  --protein-table /path/to/protein_table.json \
  --mode topk \
  --k 5 \
  --fp-th 0.30 \
  --topn 30 \
  --csv-out /path/to/out_cases_topk.csv
```

**Threshold mode:**

```bash
python evaluate_results.py cases \
  --results /path/to/merged/test.pkl \
  --dataset-csv /path/to/dataset.csv \
  --protein-table /path/to/protein_table.json \
  --mode threshold \
  --tau 0.50 \
  --topn 30 \
  --fn-weight 2.0 \
  --fp-weight 1.0 \
  --csv-out /path/to/out_cases_tau050.csv
```

### What are `tau`, `fp_th`, and weights?

* `tau`
  The global threshold in **threshold mode**. Controls the FP/FN tradeoff.

* `fp_th`
  Used only in **topk mode** to define “confident false positives”:
  FP inside top-k that also have `score >= fp_th`.
  This helps surface cases where the model is **very confident but wrong**.

* `fn_weight` and `fp_weight`
  Used to compute a simple “error score” per chain for ranking:
  `error = fp_weight * FP + fn_weight * FN`
  Set `fn_weight > fp_weight` if missing true catalytic residues is worse than having extra false candidates.

---

## Outputs

### `merge`

In `--save-dir`:

* `test.pkl`, `validation.pkl`, and optionally `train_dedup.pkl`
* `metrics_summary.json`
* `*_pr.pdf` (PR curve)
* `*_mpk.pdf` or `*_fk.pdf` (MaxPrecision@k curve)
* `per_chain.csv` (one row per chain)

`per_chain.csv` typically contains:

* `id` (Sequence_ID or normalized id)
* `split` (split1..split5)
* `weight`
* `npos` (# positives)
* `MP@k` for chosen k(s)
* AUCPR (optional per-chain AUCPR if enabled)
* pointers to source fold files (optional)

### `seeds`

In `--out-dir`:

* `mpk_whiskers.pdf` (mean±std across seeds)
* `pr_overlay.pdf` (per-seed PR curves + mean PR curve)
* `snr_summary.json` (SNR_AUCPR + SNR_MP(k))
* `metrics_by_seed.csv` (per-seed AUCPR, MP@k values, etc.)

### `cases`

* printed summary in stdout
* optional `--csv-out` with:

  * chain id, uniprot id, full_name, EC, split
  * FP/FN counts (or FP@k/FN@k)
  * max FP score, etc.

---

## How IDs become “UniProt IDs” and full names

### Best practice (recommended)

Store `ids` in training pickles as `Sequence_ID`, exactly matching `dataset.csv`:

* `A0A1L8G2K9_A`

Then:

* `uniprot_id = "A0A1L8G2K9"`
* `full_name` is looked up from `dataset.csv` via `Sequence_ID`
* protein metadata can be joined via `protein_table.json` by `uniprot_id`


---

## Examples (recommended workflows)

### A) Merge and plot CV results

```bash
python evaluate_results.py merge \
  --base-dir /home/.../sequence_homology_baseline \
  --run-tpl results_adaptive_cutoff_cv{fold} \
  --folds 1 2 3 4 5 \
  --subsets test validation train \
  --save-dir /home/.../sequence_homology_baseline_merged \
  --dedupe-train mean \
  --max-k 20 \
  --save-plots \
  --tick-grid 0.1
```

### B) Multi-seed evaluation with SNR vs a baseline

```bash
python evaluate_results.py seeds \
  --root /home/.../scannet_experiments_outputs \
  --template "transfer_msa_alpha2_1_gamma_0_v{version}" \
  --versions 1 2 3 4 5 6 \
  --subset test \
  --baseline /home/.../handcrafted_baseline_merged/test.pkl \
  --out-dir /home/.../transfer_msa_alpha2_1_gamma_0_seed_summary \
  --max-k 20 \
  --tick-grid 0.1 \
  --pr-grid 0.01
```

### C) Inspect worst FP/FN cases (top-k mode)

```bash
python evaluate_results.py cases \
  --results /home/.../transfer_msa_alpha2_1_gamma_0_seed_summary/test_merged.pkl \
  --dataset-csv /home/.../dataset.csv \
  --protein-table /home/.../protein_table.json \
  --mode topk \
  --k 5 \
  --fp-th 0.30 \
  --topn 50 \
  --csv-out /home/.../cases_topk_k5_tau0p30.csv
```

