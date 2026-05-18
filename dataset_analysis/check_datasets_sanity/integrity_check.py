#!/usr/bin/env python3
import pandas as pd
import sys

def check_column(df, col):
    """Return {value: set_of_split_types} for values appearing in more than one split."""
    bad = {}
    for val, group in df.groupby(col):
        types = set(group['Set_Type'])
        if len(types) > 1:
            bad[val] = types
    return bad

def main(path):
    df = pd.read_csv(path)
    required = {'Cluster_1','Cluster_2','Component_ID','Set_Type'}
    if not required.issubset(df.columns):
        print("Missing required columns:", required - set(df.columns))
        sys.exit(1)

    overall_ok = True
    for col in ['Cluster_1','Cluster_2','Component_ID']:
        bad = check_column(df, col)
        if bad:
            overall_ok = False
            print(f"\n[FAIL] Integrity violations for '{col}':")
            for val, types in bad.items():
                print(f"  {col} = {val!r} -> found in splits {sorted(types)}")
        else:
            print(f"[OK] All '{col}' values are contained within a single split.")

    if overall_ok:
        print("\n[PASS] Validation passed: component integrity is intact.")
    else:
        print("\n[WARN] Violations found. Fix the distribution of the listed values.")

table_path = "/home/iscb/wolfson/annab4/DB/all_proteins/cross_validation_chem/weight_based_v5/dataset.csv"
main(table_path)
