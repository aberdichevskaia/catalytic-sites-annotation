#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import glob
import csv
import argparse
import hashlib
from typing import Dict, List, Tuple

import numpy as np

import biotite.structure as struc
import biotite.structure.io.pdbx as pdbx
import biotite.structure.io.pdb as pdb
from biotite.structure.info import one_letter_code


AA_ALPH = set("ACDEFGHIKLMNPQRSTVWY")


def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def esm_cache_path(out_dir: str, origin: str) -> str:
    sub = origin[:2] if len(origin) >= 2 else "__"
    return os.path.join(out_dir, sub, f"{origin}.npy")


def read_structure_atoms(struct_path: str, model: int = 1, use_author_fields: bool = True):
    """
    Read PDB or CIF with biotite and return AtomArray.
    """
    ext = os.path.splitext(struct_path)[1].lower()
    if ext in [".cif", ".mmcif", ".pdbx", ".bcif"]:
        cif = pdbx.CIFFile.read(struct_path)
        atoms = pdbx.get_structure(cif, model=model, use_author_fields=use_author_fields)
        return atoms
    elif ext == ".pdb":
        pdbf = pdb.PDBFile.read(struct_path)
        atoms = pdb.get_structure(pdbf, model=model)
        return atoms
    else:
        raise ValueError(f"Unsupported structure extension: {ext}")


def extract_chain_sequence_like_your_code(atoms, chain_id: str) -> str:
    """
    Exactly the logic you showed:
    - select chain
    - keep amino acids only
    - residues -> 3-letter -> 1-letter (unknown -> X)
    """
    chain_atoms = atoms[atoms.chain_id == chain_id]
    chain_atoms = chain_atoms[struc.filter_amino_acids(chain_atoms)]
    if chain_atoms.array_length() == 0:
        return ""

    _, three_letter = struc.get_residues(chain_atoms)

    seq_chars = []
    for aa3 in three_letter:
        try:
            c = one_letter_code(aa3)
            if c not in AA_ALPH:
                c = "X"
        except Exception:
            c = "X"
        seq_chars.append(c)

    return "".join(seq_chars)


def list_chain_ids(atoms) -> List[str]:
    # keep order but unique
    seen = set()
    out = []
    for c in atoms.chain_id:
        if c is None:
            continue
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


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


def make_origin(
    basename: str,
    chain_id: str,
    origin_mode: str,
    n_chains_in_file: int
) -> str:
    """
    origin_mode:
      - file: basename only (if multi-chain, we still disambiguate to avoid overwriting)
      - file+chain: basename_<chain>
      - auto: basename if single-chain else basename_<chain>
    """
    if origin_mode == "file":
        # If multiple chains, do not overwrite silently
        return basename if n_chains_in_file <= 1 else f"{basename}__{chain_id}"
    if origin_mode == "file+chain":
        return f"{basename}_{chain_id}"
    if origin_mode == "auto":
        return basename if n_chains_in_file <= 1 else f"{basename}_{chain_id}"
    raise ValueError(f"Unknown origin_mode: {origin_mode}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--struct_dir", required=True, help="Directory with .pdb/.cif")
    ap.add_argument("--out_dir", required=True, help="Output directory for .npy embeddings")
    ap.add_argument("--recursive", action="store_true", help="Recurse into subfolders")
    ap.add_argument("--patterns", nargs="+", default=["*.pdb", "*.cif"], help="Glob patterns to include")
    ap.add_argument("--origin_mode", default="auto", choices=["auto", "file", "file+chain"])
    ap.add_argument("--model", type=int, default=1, help="Model index for multi-model structures (biotite)")
    ap.add_argument("--use_author_fields", action="store_true", default=True, help="Use author chain IDs for CIF")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="Device for ESM2")
    ap.add_argument("--dtype", default="float16", choices=["float16", "float32"], help="Save dtype")
    ap.add_argument("--layer", type=int, default=30, help="ESM2 repr layer (t30 -> 30)")
    ap.add_argument("--max_tokens", type=int, default=8000, help="Max tokens per batch (sum of (L+2))")
    ap.add_argument("--max_batch_size", type=int, default=16, help="Max sequences per batch")
    ap.add_argument("--skip_existing", action="store_true", help="Skip if output file exists")
    ap.add_argument("--manifest", default=None, help="Optional manifest TSV path")
    ap.add_argument("--qos_log_every", type=int, default=50, help="Print progress every N embeddings")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    manifest_path = args.manifest or os.path.join(args.out_dir, "manifest.tsv")

    # ---- collect files ----
    files = []
    if args.recursive:
        for pat in args.patterns:
            files += glob.glob(os.path.join(args.struct_dir, "**", pat), recursive=True)
    else:
        for pat in args.patterns:
            files += glob.glob(os.path.join(args.struct_dir, pat))
    files = sorted(set(files))

    if not files:
        print("[ERROR] No structure files found.")
        sys.exit(1)

    # ---- extract sequences ----
    origin_to_seq: Dict[str, str] = {}
    origin_meta: Dict[str, Tuple[str, str]] = {}  # origin -> (source_file, chain_id)

    for fpath in files:
        base = os.path.splitext(os.path.basename(fpath))[0]
        try:
            atoms = read_structure_atoms(fpath, model=args.model, use_author_fields=args.use_author_fields)
            chain_ids = list_chain_ids(atoms)
            n_ch = len(chain_ids)

            for ch in chain_ids:
                seq = extract_chain_sequence_like_your_code(atoms, ch)
                if not seq:
                    continue

                origin = make_origin(base, ch, args.origin_mode, n_ch)

                # if collision but different sequence -> disambiguate
                if origin in origin_to_seq and origin_to_seq[origin] != seq:
                    dis = hashlib.sha1((fpath + "_" + ch).encode("utf-8")).hexdigest()[:8]
                    origin = f"{origin}__{dis}"

                origin_to_seq[origin] = seq
                origin_meta[origin] = (fpath, ch)

        except Exception as e:
            print(f"[WARN] Failed to parse {fpath}: {e}", file=sys.stderr)

    items = sorted(origin_to_seq.items(), key=lambda x: len(x[1]), reverse=True)
    print(f"Total files: {len(files)}")
    print(f"Total unique origins: {len(items)}")

    # ---- ESM2 ----
    import torch
    import esm

    if args.device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA requested but not available; falling back to CPU", file=sys.stderr)
        args.device = "cpu"

    save_dtype = np.float16 if args.dtype == "float16" else np.float32

    with open(manifest_path, "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["origin", "len", "sha1", "path", "status", "error", "source_file", "chain_id"])

        # --- pre-skip ---
        if args.skip_existing:
            items_to_process = []
            skipped = 0
            for origin, seq in items:
                out_path = esm_cache_path(args.out_dir, origin)
                src, ch = origin_meta.get(origin, ("", ""))
                if os.path.exists(out_path):
                    w.writerow([origin, len(seq), sha1(seq), out_path, "skipped_exists", "", src, ch])
                    skipped += 1
                else:
                    items_to_process.append((origin, seq))
            items = items_to_process
            print(f"Skip existing: {skipped} already cached, {len(items)} to compute")

        if not items:
            print("Nothing to compute: all embeddings already exist.")
            print(f"Manifest written to: {manifest_path}")
            print("Done.")
            return

        # --- load ESM only now ---
        esm_model, alphabet = esm.pretrained.esm2_t30_150M_UR50D()
        esm_model.eval()
        esm_model = esm_model.to(args.device)
        batch_converter = alphabet.get_batch_converter()

        batches = batch_by_tokens(items, max_tokens=args.max_tokens, max_batch_size=args.max_batch_size)

        done = 0
        total = len(items)

        for bidx, batch in enumerate(batches, 1):
            data = [(origin, seq) for origin, seq in batch]
            try:
                _, _, toks = batch_converter(data)
                toks = toks.to(args.device)

                with torch.no_grad():
                    out = esm_model(toks, repr_layers=[args.layer], return_contacts=False)

                reps = out["representations"][args.layer]

                for i, (origin, seq) in enumerate(batch):
                    rep = reps[i, 1:len(seq) + 1].detach().cpu().numpy()
                    rep = rep.astype(save_dtype, copy=False)

                    out_path = esm_cache_path(args.out_dir, origin)
                    os.makedirs(os.path.dirname(out_path), exist_ok=True)
                    tmp_path = out_path + ".tmp.npy"
                    np.save(tmp_path, rep)
                    os.replace(tmp_path, out_path)

                    src, ch = origin_meta.get(origin, ("", ""))
                    w.writerow([origin, len(seq), sha1(seq), out_path, "ok", "", src, ch])

                    done += 1
                    if done % args.qos_log_every == 0:
                        print(f"Saved {done}/{total}")

            except Exception as e:
                for origin, seq in batch:
                    out_path = esm_cache_path(args.out_dir, origin)
                    src, ch = origin_meta.get(origin, ("", ""))
                    w.writerow([origin, len(seq), sha1(seq), out_path, "failed", repr(e), src, ch])
                print(f"[ERROR] batch {bidx}/{len(batches)} failed: {e}", file=sys.stderr)

    print(f"Manifest written to: {manifest_path}")
    print("Done.")


if __name__ == "__main__":
    main()
