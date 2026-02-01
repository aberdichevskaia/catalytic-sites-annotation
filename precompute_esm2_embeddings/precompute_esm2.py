#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import sys
import csv
import argparse
import hashlib
from typing import List, Tuple, Dict

import numpy as np


AA20 = set("ACDEFGHIKLMNPQRSTVWY")

def wrap_list(arrays):
    try:
        return np.array(list(arrays))
    except:
        return np.array(list(arrays), dtype=object)

def read_labels(input_file, nmax=np.inf, label_type='int'):
    assert label_type in ['int','vec_bool', 'float']
    list_origins = []
    list_sequences = []
    list_labels = []
    list_resids = []

    with open(input_file, 'r') as f:
        count = 0
        for line in f:
            if (line[0] == '>'):
                if count == nmax:
                    break
                if count > 0:
                    list_origins.append(origin)
                    list_sequences.append(sequence)
                    list_labels.append(np.array(labels))
                    list_resids.append(np.array(resids))

                origin = line[1:-1]
                sequence = ''
                labels = []
                resids = []
                count += 1
            else:
                line_splitted = line[:-1].split(' ')
                resids.append(line_splitted[:-2])
                sequence += line_splitted[-2]
                if label_type == 'int':
                    labels.append(int(line_splitted[-1]))
                elif label_type == 'vec_bool':
                    labels.append( np.array([bool(int(u)) for u in line_splitted[-1]] ) )
                elif label_type == 'float':                    
                    labels.append(float(line_splitted[-1]))
                else:
                    raise ValueError

    list_origins.append(origin)
    list_sequences.append(sequence)
    list_labels.append(np.array(labels))
    list_resids.append(np.array(resids))

    list_origins = np.array(list_origins)
    list_sequences = np.array(list_sequences)
    list_labels = wrap_list(list_labels)
    list_resids = wrap_list(list_resids)
    return list_origins, list_sequences, list_resids, list_labels


def sanitize_sequence(seq: str) -> str:
    seq = seq.strip().upper()
    return "".join((aa if aa in AA20 else "X") for aa in seq)


def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def esm_cache_path(out_dir: str, origin: str) -> str:
    sub = origin[:2]
    return os.path.join(out_dir, sub, f"{origin}.npy")


def batch_by_tokens(
    items: List[Tuple[str, str]],
    max_tokens: int = 8000,
    max_batch_size: int = 16,
) -> List[List[Tuple[str, str]]]:
    """
    Packs sequences into batches so that sum(len(seq)+2) <= max_tokens.
    Also caps batch size.
    """
    batches = []
    batch = []
    tok_sum = 0

    for origin, seq in items:
        tok = len(seq) + 2
        if batch and (tok_sum + tok > max_tokens or len(batch) >= max_batch_size):
            batches.append(batch)
            batch = []
            tok_sum = 0
        batch.append((origin, seq))
        tok_sum += tok

    if batch:
        batches.append(batch)
    return batches


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits_dir", required=True, help="Directory with split*.txt files")
    ap.add_argument("--out_dir", required=True, help="Output directory for .npy embeddings")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="Device for ESM2")
    ap.add_argument("--dtype", default="float16", choices=["float16", "float32"], help="Save dtype")
    ap.add_argument("--layer", type=int, default=30, help="ESM2 layer (t30 -> 30)")
    ap.add_argument("--max_tokens", type=int, default=8000, help="Max tokens per batch (sum of (L+2))")
    ap.add_argument("--max_batch_size", type=int, default=16, help="Max sequences per batch")
    ap.add_argument("--skip_existing", action="store_true", help="Skip if output file exists")
    ap.add_argument("--manifest", default=None, help="Optional manifest TSV path")
    ap.add_argument("--qos_log_every", type=int, default=50, help="Print progress every N proteins")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    manifest_path = args.manifest or os.path.join(args.out_dir, "manifest.tsv")

    # ---- read all splits using exactly the same parser as training ----
    origin_to_seq: Dict[str, str] = {}
    splits = glob.glob(os.path.join(args.splits_dir, "*.txt"))
    for sp in splits:
        list_origins, list_sequences, _, _ = read_labels(sp)
        for origin, seq in zip(list_origins, list_sequences):
            origin = str(origin)
            seq = sanitize_sequence(str(seq))
            if origin in origin_to_seq and origin_to_seq[origin] != seq:
                print(f"[WARN] origin {origin} has different sequences across splits!", file=sys.stderr)
            origin_to_seq[origin] = seq

    items = sorted(origin_to_seq.items(), key=lambda x: len(x[1]), reverse=True)
    print(f"Total unique proteins: {len(items)}")

    # ---- lazy imports for torch + esm ----
    import torch
    import esm

    if args.device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA requested but not available; falling back to CPU", file=sys.stderr)
        args.device = "cpu"

    model, alphabet = esm.pretrained.esm2_t30_150M_UR50D()
    model.eval()
    model = model.to(args.device)
    batch_converter = alphabet.get_batch_converter()

    save_dtype = np.float16 if args.dtype == "float16" else np.float32

    # Prepare manifest
    with open(manifest_path, "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["origin", "len", "sha1", "path", "status", "error"])

        # batch packing
        batches = batch_by_tokens(items, max_tokens=args.max_tokens, max_batch_size=args.max_batch_size)

        done = 0
        for bidx, batch in enumerate(batches, 1):
            # optionally skip entire batch items if existing
            batch2 = []
            for origin, seq in batch:
                out_path = esm_cache_path(args.out_dir, origin)
                if args.skip_existing and os.path.exists(out_path):
                    w.writerow([origin, len(seq), sha1(seq), out_path, "skipped_exists", ""])
                else:
                    batch2.append((origin, seq))

            if not batch2:
                continue

            # ESM input format: list of (name, sequence)
            data = [(origin, seq) for origin, seq in batch2]
            try:
                _, _, toks = batch_converter(data)
                toks = toks.to(args.device)

                with torch.no_grad():
                    out = model(toks, repr_layers=[args.layer], return_contacts=False)
                reps = out["representations"][args.layer]  # (B, T, 640)

                # Save each sequence separately
                for i, (origin, seq) in enumerate(batch2):
                    rep = reps[i, 1 : len(seq) + 1].detach().cpu().numpy()  # (L, 640)
                    rep = rep.astype(save_dtype, copy=False)

                    out_path = esm_cache_path(args.out_dir, origin)
                    os.makedirs(os.path.dirname(out_path), exist_ok=True)

                    tmp_path = out_path + ".tmp.npy"
                    np.save(tmp_path, rep)
                    os.replace(tmp_path, out_path)

                    w.writerow([origin, len(seq), sha1(seq), out_path, "ok", ""])
                    done += 1
                    if done % args.qos_log_every == 0:
                        print(f"Saved {done}/{len(items)}")

            except Exception as e:
                # Mark all in batch2 as failed
                for origin, seq in batch2:
                    out_path = esm_cache_path(args.out_dir, origin)
                    w.writerow([origin, len(seq), sha1(seq), out_path, "failed", repr(e)])
                print(f"[ERROR] batch {bidx}/{len(batches)} failed: {e}", file=sys.stderr)

    print(f"Manifest written to: {manifest_path}")
    print("Done.")


if __name__ == "__main__":
    main()
