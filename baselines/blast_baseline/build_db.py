#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Correct BLAST DB builder: preserve split membership from filenames.

What this does
--------------
- Parses each split file (e.g., split1.txt ... split5.txt) independently.
- Builds per-split CSVs: db_split1.csv, ..., each with columns Entry,Sequence,Residue.
- Builds train-only CSVs: db_train_split1.csv = union of all splits except split1, etc.
- Writes an overlap report if any Entry appears in more than one split.

Why this version?
-----------------
Your previous run mixed splits because it merged all split files first
and then relied on dataset.csv to infer Set_Type. Here we rely solely
on the split filenames and headers, ensuring no cross-contamination.

Input format (split*.txt)
-------------------------
>Sequence_ID
A 1 M 0
A 2 G 1
...
- We reconstruct Sequence by concatenating the third column (AA) ordered by the second column (1-based).
- Residue indices with label==1 are converted to 0-based.

Outputs
-------
out_dir/
  db_split1.csv, ..., db_splitN.csv
  db_train_split1.csv, ..., db_train_splitN.csv
  overlap_report.csv  # Entries that appear in >1 split (for sanity check)

Usage
-----
python make_blast_db_per_split.py \
  --splits_glob "/path/to/split*.txt" \
  --out_dir /path/to/blast_db_out
"""
import argparse
import glob
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd

def parse_split_file(path: str) -> Dict[str, Tuple[str, List[int]]]:
    entries: Dict[str, Tuple[str, List[int]]] = {}
    cur_id = None
    rows: List[Tuple[int,str,int]] = []  # pos1, aa, label
    def flush():
        nonlocal cur_id, rows
        if cur_id is None:
            return
        rows.sort(key=lambda x: x[0])
        seq = "".join(aa for (pos, aa, lab) in rows)
        pos0 = [pos-1 for (pos, aa, lab) in rows if int(lab)==1]
        entries[cur_id] = (seq, pos0)
        cur_id = None
        rows = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if s.startswith(">"):
                flush()
                cur_id = s[1:].strip()
                rows = []
            else:
                parts = s.split()
                if len(parts) < 4:
                    continue
                try:
                    pos = int(parts[1])
                except ValueError:
                    continue
                aa = parts[2]
                try:
                    lab = int(parts[3])
                except ValueError:
                    lab = 0
                rows.append((pos, aa, lab))
    flush()
    return entries

def df_from_entries(d: Dict[str, Tuple[str, List[int]]]) -> pd.DataFrame:
    records = []
    for k,(seq,pos) in d.items():
        rec = {
            "Entry": k,
            "Sequence": seq,
            "Residue": "|".join(map(str, sorted(set(pos)))) if pos else ""
        }
        records.append(rec)
    return pd.DataFrame.from_records(records)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits_glob", required=True, help='e.g., "/path/split*.txt"')
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = sorted(glob.glob(args.splits_glob))
    if not paths:
        raise SystemExit(f"No files match: {args.splits_glob}")

    per_split_df = {}
    all_membership = []  # (Entry, split_name)

    # Build per-split DBs
    for p in paths:
        split_name = Path(p).stem  # "split1"
        data = parse_split_file(p)
        df = df_from_entries(data)
        per_split_df[split_name] = df
        df_out = out_dir / f"db_{split_name}.csv"
        df.to_csv(df_out, index=False)
        # membership
        all_membership.extend([(e, split_name) for e in df["Entry"].tolist()])
        print(f"[OK] {split_name}: {len(df)} entries -> {df_out}")

    # Overlap report
    mem_df = pd.DataFrame(all_membership, columns=["Entry","Split"])
    dup = (mem_df.groupby("Entry")["Split"]
                 .agg(list).reset_index(name="Splits")
                 .assign(n=lambda x: x["Splits"].apply(len))
                 .query("n > 1"))
    overlap_path = out_dir / "overlap_report.csv"
    dup.to_csv(overlap_path, index=False)
    print(f"[INFO] Overlap entries: {len(dup)} -> {overlap_path}")

    # Train-only unions
    split_names = list(per_split_df.keys())
    for s in split_names:
        others = [per_split_df[o] for o in split_names if o != s]
        if not others:
            continue
        train_df = pd.concat(others, axis=0, ignore_index=True)
        # drop duplicate Entry (keep first occurrence)
        train_df = train_df.drop_duplicates(subset=["Entry"], keep="first")
        out_csv = out_dir / f"db_train_{s}.csv"
        train_df.to_csv(out_csv, index=False)
        print(f"[OK] train for {s}: {len(train_df)} entries -> {out_csv}")

    print("[DONE] All per-split and train DBs built.")

if __name__ == "__main__":
    main()
