#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import sys
import argparse
from glob import glob
from typing import List, Tuple

# Подключаем ScanNet_Ub
SCANNET_ROOT = "/home/iscb/wolfson/annab4/main_scannet/ScanNet_Ub"
if SCANNET_ROOT not in sys.path:
    sys.path.insert(0, SCANNET_ROOT)

from preprocessing import PDBio, PDB_processing, sequence_utils

UNIPROT_RE = re.compile(r"[A-NR-Z0-9]{6,10}")

def ensure_slash(p: str) -> str:
    return p if p.endswith(os.sep) else p + os.sep

def entry_from_path(path: str) -> str:
    base = os.path.basename(path)
    m = re.match(r"AF-([A-NR-Z0-9]{6,10})-F\d+-model_", base)
    if m:
        return m.group(1)
    stem = os.path.splitext(base)[0]
    m2 = UNIPROT_RE.search(stem)
    return m2.group(0) if m2 else stem

def all_structures(structures_dir: str) -> List[str]:
    structures_dir = ensure_slash(structures_dir)
    # предпочитаем .cif, но посчитаем из любого формата
    cif = glob(os.path.join(structures_dir, "*.cif"))
    pdb = glob(os.path.join(structures_dir, "*.pdb"))
    # если один и тот же ACC встречается в обоих форматах, оставим .cif
    seen = {}
    for p in cif + pdb:
        acc = entry_from_path(p)
        if acc not in seen:
            seen[acc] = p
        else:
            # заменить .pdb на .cif при наличии
            if seen[acc].lower().endswith(".pdb") and p.lower().endswith(".cif"):
                seen[acc] = p
    return list(seen.values())

def target_msa_name(acc: str, model_id: str, chain_id: str) -> str:
    # формат, который ожидает predict_bindingsites при assembly=True:
    # 'MSA_<query_name>_<model>_<chain>.fasta'
    # где query_name у нас == ACC
    return f"MSA_{acc}_{model_id}_{chain_id}.fasta"

def build_one_msa(sequence: str, out_file: str, cores: int) -> None:
    # sequence_utils.call_mmseqs сам создаст *.fasta; нужно, чтобы директория существовала и была доступна на запись
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    # выбираем mmseqs или hhblits — библиотека сама возьмёт из pipeline,
    # но тут напрямую вызовем mmseqs (обычно fastest и настроен у вас)
    sequence_utils.call_mmseqs(sequence, out_file, cores=cores)

def main():
    ap = argparse.ArgumentParser(description="Precompute MSAs for AF Human so predict.py won't write anything.")
    ap.add_argument("--structures_dir", required=True, help="e.g. /home/iscb/wolfson/jeromet/AFDB/Human")
    ap.add_argument("--out_msa_dir", required=True, help="Writable folder for MSA files")
    ap.add_argument("--cores_per_job", type=int, default=4, help="Threads per mmseqs run")
    ap.add_argument("--only_missing", action="store_true", help="Skip if target MSA file exists")
    ap.add_argument("--chains", default="all", choices=["all","A"], help="Generate MSAs for all chains or only chain A")
    args = ap.parse_args()

    out_dir = ensure_slash(args.out_msa_dir)
    os.makedirs(out_dir, exist_ok=True)

    files = all_structures(args.structures_dir)
    if not files:
        print(f"[ERR] No structures found in {args.structures_dir}")
        sys.exit(1)

    total = 0
    done = 0
    skipped = 0
    failed = 0

    for path in files:
        acc = entry_from_path(path)
        try:
            _, chains = PDBio.load_chains(file=path)  # возвращает список chain_obj
        except Exception as e:
            print(f"[WARN] skip {acc}: cannot parse structure ({e})")
            failed += 1
            continue

        # составим пары (model_id, chain_id, sequence)
        triples: List[Tuple[str,str,str]] = []
        for ch in chains:
            # get_full_id() -> (structure_id, model_id, chain_id, ...)
            fid = ch.get_full_id()
            model_id = str(fid[1])
            chain_id = str(fid[2]).strip() or "A"
            if args.chains == "A" and chain_id != "A":
                continue
            seq = PDB_processing.process_chain(ch)[0]
            if not seq or len(seq) == 0:
                continue
            triples.append((model_id, chain_id, seq))

        if not triples:
            print(f"[WARN] {acc}: no sequences extracted; skipping")
            skipped += 1
            continue

        for model_id, chain_id, seq in triples:
            total += 1
            out_name = target_msa_name(acc, model_id, chain_id)   # e.g. MSA_Q8NHM5_0_A.fasta
            out_file = os.path.join(out_dir, out_name)
            if args.only_missing and os.path.exists(out_file):
                skipped += 1
                continue
            try:
                print(f"[MSA] {acc} {model_id}_{chain_id} -> {out_file}")
                build_one_msa(seq, out_file, cores=args.cores_per_job)
                done += 1
            except Exception as e:
                print(f"[FAIL] {acc} {model_id}_{chain_id}: {e}")
                failed += 1

    print(f"[SUMMARY] targets: {total} | built: {done} | skipped: {skipped} | failed: {failed}")
    print(f"[HINT] Now run predict.py with: --use_msa --msa_dir {out_dir}")

if __name__ == "__main__":
    main()
