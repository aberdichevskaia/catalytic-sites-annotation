#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import sys
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Group isoforms by base protein id and write per-protein txt files "
            "containing absolute paths to PDB structures.\n\n"
            "Input modes (mutually exclusive):\n"
            "  --csv        CSV with isoform IDs (one row per isoform).\n"
            "  --base-ids-txt  Plain text file with base IDs (one per line). "
            "All matching PDBs in --pdb-dir are included automatically."
        )
    )

    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--csv",
        default=None,
        help="Path to CSV with isoform IDs.",
    )
    src.add_argument(
        "--base-ids-txt",
        default=None,
        metavar="FILE",
        help="Plain text file with base IDs (one per line, # lines ignored). "
             "All <base_id>-*.pdb files found in --pdb-dir are included.",
    )

    p.add_argument(
        "--column",
        default="protein id (uniprot / PDB)",
        help="Column name containing isoform IDs (used with --csv; default: 'protein id (uniprot / PDB)').",
    )
    p.add_argument(
        "--pdb-dir",
        required=True,
        help="Directory with PDB files named like O14492-1.pdb",
    )
    p.add_argument(
        "--out-dir",
        required=True,
        help="Output directory where *_isoforms.txt will be written",
    )
    p.add_argument(
        "--suffix",
        default=".pdb",
        help="PDB filename suffix (default: .pdb)",
    )
    p.add_argument(
        "--min-isoforms",
        type=int,
        default=2,
        help="Write group file only if base protein has at least this many isoforms (default: 2).",
    )
    p.add_argument(
        "--strict",
        action="store_true",
        help="Fail if any isoform PDB is missing. By default missing files are skipped with a warning.",
    )
    return p.parse_args()


def base_id_from_isoform(isoform_id: str) -> str:
    # Example: "O00555-3" -> "O00555"
    # If no '-', keep as-is.
    s = str(isoform_id).strip()
    if not s:
        return ""
    return s.split("-", 1)[0]


def _groups_from_csv(args) -> dict[str, list[str]]:
    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"ERROR: CSV not found: {csv_path}", file=sys.stderr)
        raise SystemExit(2)

    df = pd.read_csv(csv_path)
    if args.column not in df.columns:
        print(
            f"ERROR: Column '{args.column}' not found in CSV. "
            f"Available columns: {list(df.columns)}",
            file=sys.stderr,
        )
        raise SystemExit(2)

    isoform_ids = (
        df[args.column].dropna().astype(str).map(str.strip)
    )
    isoform_ids = isoform_ids[isoform_ids != ""].unique().tolist()

    groups: dict[str, list[str]] = {}
    for iso_id in isoform_ids:
        b = base_id_from_isoform(iso_id)
        if b:
            groups.setdefault(b, []).append(iso_id)
    return groups


def _groups_from_base_ids_txt(args) -> dict[str, list[str]]:
    txt_path = Path(args.base_ids_txt)
    if not txt_path.exists():
        print(f"ERROR: base-ids-txt not found: {txt_path}", file=sys.stderr)
        raise SystemExit(2)

    pdb_dir = Path(args.pdb_dir)
    groups: dict[str, list[str]] = {}

    with txt_path.open(encoding="utf-8") as f:
        for line in f:
            base_id = line.strip()
            if not base_id or base_id.startswith("#"):
                continue
            # Glob all matching isoform PDBs
            matched = sorted(pdb_dir.glob(f"{base_id}-*{args.suffix}"))
            iso_ids = [p.stem for p in matched]  # stem = O00555-3 (no suffix)
            if iso_ids:
                groups[base_id] = iso_ids
            else:
                print(f"WARN: no PDBs for base_id={base_id}", file=sys.stderr)

    return groups


def main() -> int:
    args = parse_args()

    pdb_dir = Path(args.pdb_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not pdb_dir.exists():
        print(f"ERROR: PDB dir not found: {pdb_dir}", file=sys.stderr)
        return 2

    if args.csv is not None:
        groups = _groups_from_csv(args)
    else:
        groups = _groups_from_base_ids_txt(args)

    # Write group files
    written = 0
    skipped_due_to_count = 0
    missing_total = 0

    index_rows = []

    for base_id, iso_ids in sorted(groups.items()):
        iso_ids_sorted = sorted(set(iso_ids))

        if len(iso_ids_sorted) < args.min_isoforms:
            skipped_due_to_count += 1
            continue

        pdb_paths: list[Path] = []
        missing_here: list[str] = []
        for iso_id in iso_ids_sorted:
            pdb_path = pdb_dir / f"{iso_id}{args.suffix}"
            if pdb_path.exists():
                pdb_paths.append(pdb_path.resolve())
            else:
                missing_here.append(iso_id)

        if missing_here:
            missing_total += len(missing_here)
            msg = f"WARNING: {base_id}: missing PDBs for isoforms: {', '.join(missing_here)}"
            if args.strict:
                print(f"ERROR: {msg}", file=sys.stderr)
                return 3
            print(msg, file=sys.stderr)

        if len(pdb_paths) < args.min_isoforms:
            # After skipping missing, group may become too small
            continue

        out_file = out_dir / f"{base_id}_isoforms.txt"
        with out_file.open("w", encoding="utf-8") as f:
            for pth in pdb_paths:
                f.write(str(pth) + "\n")

        index_rows.append(
            {
                "base_id": base_id,
                "n_isoforms_in_csv": len(iso_ids_sorted),
                "n_pdb_found": len(pdb_paths),
                "group_file": str(out_file.resolve()),
            }
        )
        written += 1

    index_path = out_dir / "isoform_groups_index.tsv"
    if index_rows:
        pd.DataFrame(index_rows).to_csv(index_path, sep="\t", index=False)

    print("=== Done ===")
    if args.csv is not None:
        print(f"Input CSV: {args.csv}")
    else:
        print(f"Input base-ids-txt: {args.base_ids_txt}")
    print(f"PDB dir: {pdb_dir}")
    print(f"Output dir: {out_dir}")
    print(f"Groups written: {written}")
    print(f"Groups skipped (too few isoforms in CSV): {skipped_due_to_count}")
    print(f"Total missing PDB files: {missing_total}")
    print(f"Index: {index_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
