#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import ast

import pandas as pd


def count_sites(cell) -> int:
    """
    Count the number of predicted amino acids in a cell.

    Supported formats:
    - "" or NaN -> 0
    - "243,286" -> 2
    - "243" -> 1
    - "['0', 'A', '243']" -> 1
    - "['0', 'A', '243', '0', 'A', '286']" -> 2
    """
    if cell is None:
        return 0

    s = str(cell).strip()
    if not s or s.lower() in {"nan", "none"}:
        return 0

    # 1) Try to parse as a Python literal (list/tuple)
    try:
        v = ast.literal_eval(s)
    except (ValueError, SyntaxError):
        v = None

    if isinstance(v, (list, tuple)):
        items = list(v)

        # case like ['0','A','243', ...] -> assume triplets (model, chain, resid)
        if len(items) % 3 == 0:
            return len(items) // 3

        # otherwise count only numeric elements
        cnt_num = 0
        for it in items:
            try:
                int(it)
                cnt_num += 1
            except (TypeError, ValueError):
                pass
        if cnt_num > 0:
            return cnt_num
        return len(items)

    # 2) Fallback: treat as a comma-separated string
    tokens = [t.strip() for t in s.split(",") if t.strip()]
    if not tokens:
        return 0

    cnt_num = 0
    for t in tokens:
        try:
            int(t)
            cnt_num += 1
        except ValueError:
            pass

    return cnt_num if cnt_num > 0 else len(tokens)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Filter rows by base UniProt ID, keeping only groups where the number "
            "of predicted sites differs at the given threshold."
        )
    )
    parser.add_argument("--in_csv", required=True, help="Input CSV with predictions")
    parser.add_argument("--out_csv", required=True, help="Output CSV")
    parser.add_argument(
        "--thr",
        type=int,
        choices=[35, 65, 85],
        required=True,
        help="Threshold in percent: 35, 65, or 85",
    )
    parser.add_argument(
        "--base_col",
        default="base uniprot id",
        help="Name of the base ID column (default: 'base uniprot id')",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.in_csv)

    thr_col = f"predicted with {args.thr}% threshold"
    if thr_col not in df.columns:
        raise SystemExit(f"Column '{thr_col}' not found in CSV.")

    if args.base_col not in df.columns:
        raise SystemExit(f"Base ID column '{args.base_col}' not found in CSV.")

    df["_n_sites"] = df[thr_col].apply(count_sites)

    grp = df.groupby(args.base_col)["_n_sites"].transform(lambda x: x.nunique() > 1)

    df_out = df[grp].copy()

    df_out.drop(columns=["_n_sites"], inplace=True)

    df_out.to_csv(args.out_csv, index=False)
    print(
        f"Saved {len(df_out)} rows to {args.out_csv} "
        f"(out of {len(df)} total), threshold {args.thr}%."
    )


if __name__ == "__main__":
    main()
