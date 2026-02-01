# find_best_permutation.py
import sys
import re
sys.path.append('/home/iscb/wolfson/annab4/main_scannet/ScanNet_Ub')

import itertools
import numpy as np
from pathlib import Path
import utilities.dataset_utils as dataset_utils

# Пример: 
BINDING_LABEL_FILES = sorted(Path("/home/iscb/wolfson/omriyakir/ubinet/datasets/scanNet/data_for_training/v4/seq_id_0.95_asaThreshold_0.1_bound_0.21/").glob("labels_fold*.txt"))   
CATALYTIC_LABEL_FILES = sorted(Path("/home/iscb/wolfson/annab4/DB/all_proteins/cross_validation_chem/weight_based_v9/").glob("split*.txt"))  

# Если у тебя в v3 binding лейблы 'vec_bool', а каталитика 'int' — укажи так:
LABEL_TYPE_BINDING = 'vec_bool'
LABEL_TYPE_CATALYTIC = 'int'

HEADER_RE = re.compile(r"""
    ^>                  # '>'
    \s*([0-9A-Za-z]{4}) # PDB code (4 chars)
    (?:_[^\s-]+)?       # optional augmentation like _0 or _12
    -([0-9A-Za-z])      # chain after '-'
    (?:[^\S\r\n].*)?$   # optional trailing stuff until EOL
""", re.VERBOSE)

def parse_origin_to_key(s: str) -> str | None:
    """
    Normalize origin/header to canonical 'PDB_CHAIN'.
    Works with lines like '>6hei_0-A' or plain tokens like '6hei_0-A'.
    Returns None if cannot parse.
    """
    s = str(s).strip()
    if s.startswith('>'):
        m = HEADER_RE.match(s)
        if m:
            pdb, chain = m.group(1).upper(), m.group(2).upper()
            return f"{pdb}_{chain}"
        # Fallbacks (rare): try splitting manually
        token = s[1:].strip().split()[0]
    else:
        token = s.split()[0]

    # Manual split: prefer '-' between left and CHAIN
    if '-' in token:
        left, chain = token.split('-', 1)
    elif '_' in token:
        # Sometimes chain is after underscore
        parts = token.split('_')
        if len(parts) >= 2:
            left, chain = parts[0], parts[-1]
        else:
            return None
    else:
        return None

    pdb = left[:4]
    if len(pdb) != 4:
        return None
    return f"{pdb.upper()}_{chain.upper()}"


# --- REPLACE your base_chain_id(...) and read_fold_ids(...) with the two below ---

def base_chain_id(s: str) -> str:
    """
    Canonicalize origin ID for intersection counting only.
    - Keep PDB code + chain, drop model markers like '_0-'
    - Upper-case PDB code and chain
    - Leave UniProt-like IDs (A0A1L8G2K9_A) as-is (upper-case)
    """
    s = str(s).strip()
    # cut off augmentation/file suffixes if any
    for sep in ['|', ';', '#']:
        s = s.split(sep)[0]

    s_up = s.upper()

    # PDB patterns like 2AYO_0-A or 6HEI_1-B -> 2AYO_A / 6HEI_B
    m = re.match(r'^([0-9A-Z]{4})_(?:\d-)?([A-Z0-9]+)$', s_up)
    if m:
        pdb, chain = m.groups()
        #return f'{pdb}_{chain}'
        return pdb

    # If looks like UNIPROT_AC_CHAIN (already like A0A1..._A), keep upper-case
    return s_up

def read_fold_ids(label_file: Path, label_type: str) -> set[str]:
    """
    Try to read origins via dataset_utils.read_labels(...).
    If that fails or returns nothing, fallback to scanning '>' header lines.
    """
    # 1) Try the original loader
    try:
        list_origins, _, _, _ = dataset_utils.read_labels(str(label_file), label_type=label_type)
        ids = {base_chain_id(o) for o in list_origins}
        if ids:
            return ids
    except Exception:
        pass

    # 2) Fallback: read raw file, collect headers
    ids: set[str] = set()
    with label_file.open("r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            if line.startswith('>'):
                key = base_chain_id(line)
                if key:
                    ids.add(key)
    if not ids:
        raise ValueError(f"No headers parsed in {label_file}")
    return ids

def load_folds(files, label_type):
    files = list(sorted(files, key=lambda p: str(p)))
    assert len(files) == 5, f"Need 5 folds, got {len(files)}"
    return [read_fold_ids(f, label_type) for f in files]

def union_except(sets, skip_idxs):
    out = set()
    for i, s in enumerate(sets):
        if i in skip_idxs: 
            continue
        out |= s
    return out

def schedule_cost(B, C, perm):
    """
    B: list[set] of 5 binding folds
    C: list[set] of 5 catalytic folds
    perm: tuple of 5 ints, mapping binding index -> catalytic index
    CV schedule (as in your code):
      for k in 0..4:
        val_b = k
        test_b = (k+1)%5
        train_b = others
        val_c = perm[k]
        test_c = perm[(k+1)%5]
        train_c = others (under perm)
    Cost = train-test leaks both ways (+ optional val-test)
    """
    total_train_test = 0
    total_val_test = 0
    aligned_overlap = 0  # prefer putting shared chains on same fold in both sets
    for k in range(5):
        val_b  = k
        test_b = (k+1) % 5
        train_b = union_except(B, {val_b, test_b})

        val_c  = perm[k]
        test_c = perm[(k+1) % 5]
        train_c = union_except(C, {val_c, test_c})

        # train<->test leaks across datasets
        leak_bt_ct = len(train_b & C[test_c])
        leak_ct_bt = len(train_c & B[test_b])
        total_train_test += leak_bt_ct + leak_ct_bt

        # optional: val<->test (strоже, но полезно)
        total_val_test += len(B[val_b] & C[test_c]) + len(C[val_c] & B[test_b])

        # encourage aligning shared chains on same k (diagonal preference)
        aligned_overlap += len(B[k] & C[perm[k]])

    # lower is better; last term is negative to *maximize* aligned_overlap
    return (total_train_test, total_val_test, -aligned_overlap)

def intersection_matrix(B, C):
    M = np.zeros((5,5), dtype=int)
    for i in range(5):
        for j in range(5):
            M[i,j] = len(B[i] & C[j])
    return M

def main():
    B = load_folds(BINDING_LABEL_FILES, LABEL_TYPE_BINDING)
    C = load_folds(CATALYTIC_LABEL_FILES, LABEL_TYPE_CATALYTIC)

    M = intersection_matrix(B, C)
    print("Intersection matrix |B_i ∩ C_j|:\n", M, "\n")

    best_perm, best_score = None, None
    for perm in itertools.permutations(range(5)):
        score = schedule_cost(B, C, perm)
        if best_score is None or score < best_score:
            best_perm, best_score = perm, score

    print("Best permutation (binding idx -> catalytic idx):", best_perm)
    print("Score (train-test leak, val-test leak, -aligned_overlap):", best_score)

    # Pretty schedule view
    print("\nCV schedule (k: val_b=k, test_b=k+1; val_c=perm[k], test_c=perm[k+1]):")
    for k in range(5):
        print(f" k={k}:  val_b={k}, test_b={(k+1)%5}  ||  val_c={best_perm[k]}, test_c={best_perm[(k+1)%5]}")

if __name__ == "__main__":
    main()
