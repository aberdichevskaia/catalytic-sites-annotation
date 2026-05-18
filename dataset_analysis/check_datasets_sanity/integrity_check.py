#!/usr/bin/env python3
import argparse
import logging
import pandas as pd
import sys

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

def check_column(df, col):
    """Return {value: set_of_split_types} for values appearing in more than one split."""
    bad = {}
    for val, group in df.groupby(col):
        types = set(group['Set_Type'])
        if len(types) > 1:
            bad[val] = types
    return bad

def main():
    ap = argparse.ArgumentParser(description="Check component integrity of a cross-validation dataset CSV.")
    ap.add_argument("--table", required=True, help="Path to dataset.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.table)
    required = {'Cluster_1','Cluster_2','Component_ID','Set_Type'}
    if not required.issubset(df.columns):
        print("Missing required columns:", required - set(df.columns))
        sys.exit(1)

    overall_ok = True
    for col in ['Cluster_1','Cluster_2','Component_ID']:
        bad = check_column(df, col)
        if bad:
            overall_ok = False
            logging.error("Integrity violations for '%s':", col)
            for val, types in bad.items():
                print(f"  {col} = {val!r} -> found in splits {sorted(types)}")
        else:
            logging.info("All '%s' values are contained within a single split.", col)

    if overall_ok:
        logging.info("Validation passed: component integrity is intact.")
    else:
        logging.warning("Violations found. Fix the distribution of the listed values.")

if __name__ == "__main__":
    main()
