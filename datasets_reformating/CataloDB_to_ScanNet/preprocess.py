#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AF-only preprocessing for CataloDB benchmark.

- Source of truth: All_metadata.tsv (sequence, EC, full_name, ACT_SITE)
- Official split: train.fasta / test.fasta
- Output format:
  >UID_A
  A 1 M 0
  ...

Outputs:
- protein_table.json
- train_raw.txt
- test.txt
- prepare_summary.json
- (optional) structure_rejections.tsv if AFDB check is enabled
"""

import argparse
import json
import os
import re
from typing import Any, Dict, List

import pandas as pd

import biotite.structure.io.pdbx as pdbx
import biotite.database.afdb as afdb

from tqdm import tqdm


EXPECTED_COLS = [
    "Entry", "Entry Name", "Protein names", "Organism", "Length",
    "EC number", "Protein families", "Sequence", "BioCyc", "Site",
    "Cofactor", "Binding site", "Active site",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--all_metadata_tsv", required=True)
    p.add_argument("--train_fasta", required=True)
    p.add_argument("--test_fasta", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--structures_dir", default="", help="If set, AFDB mmCIF files are cached here")
    p.add_argument("--skip_afdb_check", action="store_true", default=False,
                   help="If set, do NOT download/check AFDB structures at preprocessing time")
    return p.parse_args()


def ensure_dir(path: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)


def extract_accession_from_fasta_header(header: str) -> str:
    if "|" in header:
        parts = header.split("|")
        if len(parts) >= 2 and parts[1]:
            return parts[1].strip()
    return header.split()[0].strip()


def read_fasta_ids(path: str) -> Dict[str, str]:
    seqs: Dict[str, str] = {}
    curr = None
    buf: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if curr is not None:
                    seqs[curr] = "".join(buf)
                curr = extract_accession_from_fasta_header(line[1:].strip())
                buf = []
            else:
                buf.append(line)
        if curr is not None:
            seqs[curr] = "".join(buf)
    return seqs


def normalize_metadata_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.shape[1] != len(EXPECTED_COLS):
        raise ValueError(f"All_metadata.tsv has {df.shape[1]} columns, expected {len(EXPECTED_COLS)}")
    df = df.copy()
    df.columns = EXPECTED_COLS
    return df


def parse_active_sites(active_site_field: Any) -> List[int]:
    if active_site_field is None:
        return []
    text = str(active_site_field)
    if not text or text.lower() == "nan":
        return []
    return [int(x) for x in re.findall(r"ACT_SITE\s+(\d+)", text)]


def parse_evidence_codes(active_site_field: Any) -> List[str]:
    if active_site_field is None:
        return []
    text = str(active_site_field)
    if not text or text.lower() == "nan":
        return []
    return sorted(set(re.findall(r"(ECO:\d+)", text)))


def first_ec(ec_field: Any) -> str:
    if ec_field is None:
        return "not found"
    text = str(ec_field).strip()
    if not text or text.lower() == "nan":
        return "not found"
    return text.split(";")[0].strip() or "not found"


def afdb_sequence_matches(uid: str, expected_seq: str, cache_dir: str) -> None:
    """
    Download AFDB CIF and validate chain A sequence equals expected.
    Raises on any mismatch/problem.
    """
    cif_path = afdb.fetch(uid, format="cif", target_path=cache_dir)
    cif = pdbx.CIFFile.read(cif_path)
    seq_map = pdbx.get_sequence(cif)
    if seq_map is None or len(seq_map) == 0:
        raise RuntimeError("No sequences found in AFDB CIF")
    if "A" in seq_map:
        got = str(seq_map["A"])
    elif len(seq_map) == 1:
        got = str(next(iter(seq_map.values())))
    else:
        raise RuntimeError(f"Chain A not found and multiple chains present: {list(seq_map.keys())}")
    if got != expected_seq:
        raise RuntimeError(f"AFDB sequence mismatch: expected_len={len(expected_seq)} got_len={len(got)}")


def make_annotation_block(uid: str, seq: str, labels: List[int]) -> List[str]:
    if len(seq) != len(labels):
        raise ValueError(f"Length mismatch for {uid}: seq={len(seq)} labels={len(labels)}")
    out = [f">{uid}_A"]
    for i, (aa, lab) in enumerate(zip(seq, labels), start=1):
        out.append(f"A {i} {aa} {int(lab)}")
    return out


def main() -> None:
    args = parse_args()
    ensure_dir(args.out_dir)
    ensure_dir(args.structures_dir)

    train_fasta = read_fasta_ids(args.train_fasta)
    test_fasta = read_fasta_ids(args.test_fasta)
    train_ids = set(train_fasta.keys())
    test_ids = set(test_fasta.keys())

    df = pd.read_csv(args.all_metadata_tsv, sep="\t", dtype=str, keep_default_na=False)
    df = normalize_metadata_df(df)

    protein_table: Dict[str, Dict[str, Any]] = {}
    train_lines: List[str] = []
    test_lines: List[str] = []
    rejects: List[Dict[str, str]] = []

    summary = {
        "n_train_fasta": len(train_ids),
        "n_test_fasta": len(test_ids),
        "n_train_written": 0,
        "n_test_written": 0,
        "n_train_skipped": 0,
        "n_test_skipped": 0,
        "afdb_check_enabled": (not args.skip_afdb_check),
    }

    for _, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing rows"):
        uid = row["Entry"].strip()
        if uid not in train_ids and uid not in test_ids:
            continue
        split = "train" if uid in train_ids else "test"

        seq = row["Sequence"].strip()
        fasta_seq = train_fasta.get(uid) if split == "train" else test_fasta.get(uid)
        if fasta_seq is not None and fasta_seq != seq:
            raise RuntimeError(
                f"[FATAL] Sequence mismatch All_metadata.tsv vs {split}.fasta for {uid}: "
                f"meta_len={len(seq)} fasta_len={len(fasta_seq)}"
            )

        active_sites = parse_active_sites(row["Active site"])
        if len(active_sites) == 0:
            raise RuntimeError(f"[FATAL] No ACT_SITE labels for {uid}")

        labels = [0] * len(seq)
        for pos in active_sites:
            if pos < 1 or pos > len(seq):
                raise RuntimeError(f"[FATAL] ACT_SITE out of range for {uid}: {pos}")
            labels[pos - 1] = 1

        if not args.skip_afdb_check:
            try:
                afdb_sequence_matches(uid, seq, args.structures_dir)
            except Exception as e:
                rejects.append({"split": split, "uniprot_id": uid, "reason": "afdb_check_failed", "detail": str(e)})
                if split == "test":
                    summary["n_test_skipped"] += 1
                else:
                    summary["n_train_skipped"] += 1
                continue

        protein_table[uid] = {
            "uniprot_id": uid,
            "uniprot_sequence": seq,
            "EC_number": first_ec(row["EC number"]),
            "full_name": row["Protein names"].strip(),
            "split": split,
            "active_sites_1based": sorted(active_sites),
            "evidence_codes": parse_evidence_codes(row["Active site"]),
        }

        block = make_annotation_block(uid, seq, labels)
        if split == "train":
            train_lines.extend(block)
            summary["n_train_written"] += 1
        else:
            test_lines.extend(block)
            summary["n_test_written"] += 1

    with open(os.path.join(args.out_dir, "protein_table.json"), "w", encoding="utf-8") as f:
        json.dump(protein_table, f, indent=2, ensure_ascii=False)

    with open(os.path.join(args.out_dir, "train_raw.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(train_lines) + "\n")

    with open(os.path.join(args.out_dir, "test.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(test_lines) + "\n")

    if not args.skip_afdb_check:
        pd.DataFrame(rejects).to_csv(
            os.path.join(args.out_dir, "structure_rejections.tsv"),
            sep="\t", index=False
        )

    with open(os.path.join(args.out_dir, "prepare_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("[DONE] AF-only preprocessing finished")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()