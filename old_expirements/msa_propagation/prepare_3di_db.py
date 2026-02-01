#!/usr/bin/env python3
import os
import json
import glob
from collections import defaultdict
from multiprocessing import Pool, cpu_count

import os
import biotite.structure.io.pdbx as pdbx

import numpy as np

import biotite.database.afdb as afdb
import biotite.structure as struc
import biotite.structure.alphabet as strucalph
from biotite.structure.info import one_letter_code
from biotite.structure.io.pdbx import CIFFile, get_structure
from biotite.structure.alphabet import I3DSequence
from biotite.sequence import ProteinSequence
from biotite.sequence.align import align_optimal, SubstitutionMatrix

# ------------------------------------------------------------------
# Параметры
# ------------------------------------------------------------------
ANNOTATION_DIR = "/home/iscb/wolfson/annab4/DB/all_proteins/cross_validation_chem/cleaned"
PROTEIN_TABLE       = "/home/iscb/wolfson/annab4/DB/all_protein_table_modified.json"
with open(PROTEIN_TABLE) as f:
    protein_data = json.load(f)
    
PDB_DIR        = "/home/iscb/wolfson/annab4/Data/PDB_files"
SAVE_FILE      = "/home/iscb/wolfson/annab4/DB/all_proteins/MSA_propagation/3Di_DB_all.json"
K              = 50   # минимальная длина 3Di
THREADS        = min(128, cpu_count())   # или считайте из argparse

# identity-матрица для AA-выравнивания
alph    = ProteinSequence.alphabet
n       = len(alph)
mat     = np.full((n,n), -100, dtype=int)
np.fill_diagonal(mat, 1)
SUB_MAT = SubstitutionMatrix(alph, alph, mat)
GAP_P   = (-100, -10)

# ------------------------------------------------------------------
# Вспомогательные функции
# ------------------------------------------------------------------
def parse_all_annotations(dir_path):
    ann = {}
    for fn in glob.glob(os.path.join(dir_path, "*.txt")):
        with open(fn) as f:
            current = None
            entries = []
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.startswith(">"):
                    if current is not None:
                        ann[current] = entries
                    parts = line[1:].split("_")
                    current = (parts[0], parts[1])
                    entries = []
                else:
                    _, resnum, aa, lab = line.split()
                    entries.append((int(resnum), aa, int(lab)))
            if current is not None:
                ann[current] = entries
    return ann

def compute_index_mapping(aligned_filtered, aligned_original):
    mapping = {}
    i_f = i_o = 0
    for a,b in zip(aligned_filtered, aligned_original):
        if a!='-' and b!='-':
            mapping[i_f] = i_o
            i_f += 1; i_o += 1
        elif a!='-' and b=='-':
            i_f += 1
        elif a=='-' and b!='-':
            i_o += 1
    return mapping

# ------------------------------------------------------------------
# Функция, обрабатывающая ОДНУ аннотацию
# ------------------------------------------------------------------
def process_entry(item):
    """
    item: ((uid, chain), entries)
    entries: list of (resnum, AA, label)
    """
    (uid, chain), entries = item

    # 1) Пропускаем PDB-коды
    if len(uid)==4 and uid.isalnum():
        return ('skipped_pdb', (uid,chain), None)

    # 2) Скачиваем .cif из AFDB
    try:
        cif_path = afdb.fetch(uid, format="cif", target_path=PDB_DIR)
        cif      = CIFFile.read(cif_path)
    except Exception as e:
        return ('skipped_fetch', (uid,chain), str(e))

    # 3) Извлекаем нужную цепь + фильтруем аминокислоты
    struct     = get_structure(cif, model=1, use_author_fields=True)
    chain_atoms= struct[struct.chain_id == chain]
    chain_atoms= chain_atoms[struc.filter_amino_acids(chain_atoms)]
    if chain_atoms.array_length() == 0:
        return ('skipped_empty', (uid,chain), None)

    # 4) Проверяем, что последовательность правильная
    expected_seq = "".join(aa for (_,aa,_) in entries)
    chains_sequences = pdbx.get_sequence(cif)
    struct_seq = str(chains_sequences.get("A"))
    
    if expected_seq != struct_seq:
        return ('skipped_mismatch', (uid,chain), None)
     
    # 4) Строим 3Di-последовательность
    symbols3di = strucalph.to_3di(chain_atoms)[0][0]
    if len(symbols3di) < K:
        return ('skipped_short', (uid,chain), len(symbols3di))

    # 5) Оригинальная AA-последовательность и метки
    orig_aa = "".join(aa for (_,aa,_) in entries)
    labs    = [lab for (_,_,lab) in entries]

    # 6) AA-последовательность из атомов (filtered)
    _, three = struc.get_residues(chain_atoms)
    filt_aa = []
    for AA3 in three:
        try:
            c = one_letter_code(AA3)
        except:
            c = 'X'
        filt_aa.append(c if c in alph else 'X')
    filt_aa = "".join(filt_aa)

    # 7) Локальное выравнивание filtered_aa → orig_aa
    psa = ProteinSequence(filt_aa)
    osa = ProteinSequence(orig_aa)
    aln = align_optimal(psa, osa, SUB_MAT, gap_penalty=GAP_P)[0]
    g_f, g_o = aln.get_gapped_sequences()
    mapping = compute_index_mapping(g_f, g_o)

    # 8) Успешно — возвращаем «processed» + все поля
    return ('processed',
            (uid, chain,
             "".join(symbols3di),
             filt_aa,
             orig_aa,
             labs,
             mapping),
            None)

# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
if __name__ == "__main__":
    os.makedirs(PDB_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(SAVE_FILE), exist_ok=True)

    # 0) Считываем аннотации
    annotations = parse_all_annotations(ANNOTATION_DIR)
    print(f"Found {len(annotations)} annotation chains")

    # 1) Параллельно process_entry
    items = list(annotations.items())
    results = []
    with Pool(THREADS) as pool:
        for res in pool.imap_unordered(process_entry, items):
            results.append(res)

    # 2) Разбиваем на processed vs skipped
    skip_counts = defaultdict(int)
    debug_examples = defaultdict(list)
    processed_recs = []

    for status, payload, info in results:
        skip_counts[status] += 1
        if status == 'processed':
            processed_recs.append(payload)
        else:
            # сохраним до 5 примеров причины
            if len(debug_examples[status]) < 5:
                debug_examples[status].append((payload, info))

    # 3) Выводим статистику
    print("\n=== Summary ===")
    print(f" processed:       {len(processed_recs)}")
    for st, cnt in skip_counts.items():
        if st!='processed':
            print(f" {st:<15} {cnt}")
    print("\n=== Debug examples for skips ===")
    for st, exs in debug_examples.items():
        print(f"\n[{st}] (showing up to 5)")
        for (uid,chain), info in exs:
            print(f"  {uid}_{chain}  -> {info}")

    # 4) Собираем выходной JSON
    db_ids, db_3di_sequences, db_filtered_aa_sequences = [], [], []
    db_annotated_aa_sequences, db_labels, db_index_mappings = [], [], []
    for uid, chain, seq3di, filt_aa, orig_aa, labs, mapping in processed_recs:
        db_ids.append((uid, chain))
        db_3di_sequences.append(seq3di)
        db_filtered_aa_sequences.append(filt_aa)
        db_annotated_aa_sequences.append(orig_aa)
        db_labels.append(labs)
        db_index_mappings.append(mapping)

    out = {
        "db_ids"                    : db_ids,
        "db_3di_sequences"          : db_3di_sequences,
        "db_filtered_aa_sequences"  : db_filtered_aa_sequences,
        "db_annotated_aa_sequences" : db_annotated_aa_sequences,
        "db_labels"                 : db_labels,
        "db_index_mappings"         : db_index_mappings
    }
    with open(SAVE_FILE, "w") as f:
        json.dump(out, f)
    print(f"\nSaved {len(db_ids)} entries to {SAVE_FILE}")
