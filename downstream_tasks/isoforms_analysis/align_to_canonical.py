#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
from typing import List, Optional, Tuple

import biotite.sequence as seq
import biotite.sequence.align as align
import biotite.sequence.io.fasta as fasta
import biotite.structure as struc
import biotite.structure.info as info
import biotite.structure.io.pdb as pdb


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extract sequence from PDB (local file or RCSB by ID), align to canonical FASTA, output A3M + mapping."
    )

    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--pdb", default=None, help="Path to local PDB file.")
    g.add_argument("--pdb-id", default=None, help="4-character PDB ID (e.g. 1L2Y). Will be downloaded from RCSB.")

    p.add_argument("--download-dir", default="./rcsb_cache", help="Where to store downloaded PDB files (when using --pdb-id).")
    p.add_argument("--overwrite-download", action="store_true", help="Overwrite downloaded file if it exists.")
    p.add_argument("--download-format", default="pdb", choices=["pdb"], help="Download format (kept as 'pdb' to match parser).")

    p.add_argument("--ref-fasta", required=True, help="Path to canonical/reference FASTA (1 sequence).")
    p.add_argument("--chain", default=None, help="Chain ID to use (e.g. A). Default: use all chains in file order.")
    p.add_argument("--model", type=int, default=1, help="PDB model number (biotite models start at 1).")

    p.add_argument("--gap", type=int, default=-10, help="Linear gap penalty (negative).")
    p.add_argument("--gap-open", type=int, default=None, help="Affine gap opening penalty (negative).")
    p.add_argument("--gap-extend", type=int, default=None, help="Affine gap extension penalty (negative).")

    p.add_argument("--out-prefix", default=None, help="Prefix for outputs. Default: <pdb_stem_or_id>[_chain].")
    p.add_argument("--line-width", type=int, default=120, help="Line width for pretty output.")
    
    p.add_argument(
        "--out-dir",
        default=".",
        help="Directory to write outputs into (default: current directory).",
    )

    return p.parse_args()


def read_single_fasta_sequence(path: Path) -> Tuple[str, str]:
    ff = fasta.FastaFile.read(str(path))
    if len(ff) == 0:
        raise ValueError(f"FASTA is empty: {path}")
    name = next(iter(ff.keys()))
    seq_str = ff[name].replace(" ", "").replace("\n", "").strip()
    if not seq_str:
        raise ValueError(f"FASTA entry is empty: {name} in {path}")
    return name, seq_str


def resolve_pdb_path(args: argparse.Namespace) -> Tuple[Path, str]:
    """
    Returns (pdb_path, pdb_label), where pdb_label is used for output naming.
    """
    if args.pdb is not None:
        pdb_path = Path(args.pdb)
        if not pdb_path.exists():
            raise FileNotFoundError(f"PDB file not found: {pdb_path}")
        return pdb_path, pdb_path.stem

    # --pdb-id case
    pdb_id = str(args.pdb_id).strip()
    if len(pdb_id) != 4:
        raise ValueError(f"--pdb-id must be 4 characters, got: {pdb_id!r}")

    # Download from RCSB via biotite
    from biotite.database import rcsb  # local import to avoid requiring internet unless needed

    download_dir = Path(args.download_dir)
    download_dir.mkdir(parents=True, exist_ok=True)

    # fetch() returns the file path to the downloaded structure file :contentReference[oaicite:1]{index=1}
    fetched_path = rcsb.fetch(
        pdb_id.lower(),
        format=args.download_format,
        target_path=str(download_dir),
        overwrite=args.overwrite_download,
        verbose=True,
    )
    return Path(fetched_path), pdb_id.lower()


def extract_pdb_sequence(
    pdb_path: Path,
    model: int = 1,
    chain: Optional[str] = None,
) -> Tuple[str, List[str], List[int], List[str], List[str]]:
    """
    Returns:
      seq_str: 1-letter sequence
      chain_ids, res_ids, ins_codes, res_names3 per residue
    """
    pdb_file = pdb.PDBFile.read(str(pdb_path))
    atoms = pdb_file.get_structure(model=model)

    if chain is not None:
        atoms = atoms[atoms.chain_id == chain]

    aa_atoms = atoms[struc.filter_amino_acids(atoms)]
    if aa_atoms.array_length() == 0:
        raise ValueError("No amino-acid atoms found (after chain/filter).")

    res_starts = struc.get_residue_starts(aa_atoms)

    res_names3 = [str(x).upper() for x in aa_atoms.res_name[res_starts]]
    chain_ids = [str(x) for x in aa_atoms.chain_id[res_starts]]
    res_ids = [int(x) for x in aa_atoms.res_id[res_starts]]

    if hasattr(aa_atoms, "ins_code"):
        ins_codes_raw = aa_atoms.ins_code[res_starts]
        ins_codes = [("" if (x is None or str(x).strip() in {".", "?"}) else str(x).strip()) for x in ins_codes_raw]
    else:
        ins_codes = [""] * len(res_starts)

    # Use Biotite CCD-based one-letter mapping; fallback to X if unknown
    seq_chars = []
    for r3 in res_names3:
        aa1 = info.one_letter_code(r3)
        if aa1 is None:
            # Common practical fallback: map MSE->M if CCD mapping fails for some reason
            if r3 == "MSE":
                aa1 = "M"
            else:
                aa1 = "X"
        seq_chars.append(aa1)

    seq_str = "".join(seq_chars)
    return seq_str, chain_ids, res_ids, ins_codes, res_names3


def alignment_to_a3m(ref_gapped: str, pdb_gapped: str) -> Tuple[str, str]:
    ref_out = []
    pdb_out = []
    for r, q in zip(ref_gapped, pdb_gapped):
        if r != "-":
            ref_out.append(r.upper())
            pdb_out.append(q.upper() if q != "-" else "-")
        else:
            if q != "-":
                pdb_out.append(q.lower())
    return "".join(ref_out), "".join(pdb_out)


def write_pretty_alignment(out_path: Path, ref_gapped: str, pdb_gapped: str, width: int = 120) -> None:
    def count_nongap(s: str) -> int:
        return sum(1 for c in s if c != "-")

    ref_pos = 1
    pdb_pos = 1
    with out_path.open("w", encoding="utf-8") as f:
        for i in range(0, len(ref_gapped), width):
            ref_chunk = ref_gapped[i:i + width]
            pdb_chunk = pdb_gapped[i:i + width]

            ref_chunk_len = count_nongap(ref_chunk)
            pdb_chunk_len = count_nongap(pdb_chunk)

            ref_end = ref_pos + ref_chunk_len - 1 if ref_chunk_len > 0 else ref_pos - 1
            pdb_end = pdb_pos + pdb_chunk_len - 1 if pdb_chunk_len > 0 else pdb_pos - 1

            f.write(f"REF {ref_pos:>6}  {ref_chunk}  {ref_end:>6}\n")
            f.write(f"PDB {pdb_pos:>6}  {pdb_chunk}  {pdb_end:>6}\n\n")

            ref_pos += ref_chunk_len
            pdb_pos += pdb_chunk_len


def main() -> int:
    args = parse_args()

    pdb_path, pdb_label = resolve_pdb_path(args)

    ref_name, ref_seq_str = read_single_fasta_sequence(Path(args.ref_fasta))

    pdb_seq_str, chain_ids, res_ids, ins_codes, res_names3 = extract_pdb_sequence(
        pdb_path=pdb_path,
        model=args.model,
        chain=args.chain,
    )

    pdb_seq = seq.ProteinSequence(pdb_seq_str)
    ref_seq = seq.ProteinSequence(ref_seq_str)

    matrix = align.SubstitutionMatrix.std_protein_matrix()

    if (args.gap_open is not None) or (args.gap_extend is not None):
        if args.gap_open is None or args.gap_extend is None:
            raise ValueError("For affine gaps set BOTH --gap-open and --gap-extend (negative ints).")
        gap_penalty = (args.gap_open, args.gap_extend)
    else:
        gap_penalty = args.gap

    ali = align.align_optimal(
        pdb_seq, ref_seq, matrix,
        gap_penalty=gap_penalty,
        terminal_penalty=True,
        local=False,
        max_number=1,
    )[0]

    pdb_gapped, ref_gapped = ali.get_gapped_sequences()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.out_prefix is None:
        stem = pdb_label
        if args.chain is not None:
            stem = f"{stem}_chain{args.chain}"
        out_prefix = out_dir / stem
    else:
        # If user gave a relative prefix, treat it as inside out_dir.
        # If absolute, keep as-is.
        user_prefix = Path(args.out_prefix)
        out_prefix = user_prefix if user_prefix.is_absolute() else (out_dir / user_prefix)

    out_prefix.parent.mkdir(parents=True, exist_ok=True)


    # A3M
    ref_a3m, pdb_a3m = alignment_to_a3m(ref_gapped=ref_gapped, pdb_gapped=pdb_gapped)
    a3m_path = out_prefix.with_suffix(".a3m")
    with a3m_path.open("w", encoding="utf-8") as f:
        f.write(f">{ref_name}\n{ref_a3m}\n")
        f.write(f">{pdb_label}\n{pdb_a3m}\n")

    # Mapping: pdb seq index -> canonical index
    trace = ali.trace
    map_pdb_to_ref = [None] * len(pdb_seq_str)
    for i_pdb, i_ref in trace:
        i_pdb = int(i_pdb)
        i_ref = int(i_ref)
        if i_pdb >= 0 and i_ref >= 0:
            map_pdb_to_ref[i_pdb] = i_ref

    mapping_path = out_prefix.with_suffix(".mapping.tsv")
    with mapping_path.open("w", encoding="utf-8") as f:
        f.write("\t".join([
            "pdb_seqpos_1based",
            "chain",
            "res_id",
            "ins_code",
            "res_name3",
            "pdb_aa1",
            "canon_pos_1based",
            "canon_aa1",
        ]) + "\n")

        for i in range(len(pdb_seq_str)):
            ref_i = map_pdb_to_ref[i]
            if ref_i is None:
                canon_pos = "NA"
                canon_aa = "NA"
            else:
                canon_pos = str(ref_i + 1)
                canon_aa = ref_seq_str[ref_i]

            f.write("\t".join([
                str(i + 1),
                chain_ids[i],
                str(res_ids[i]),
                ins_codes[i] if ins_codes[i] != "" else ".",
                res_names3[i],
                pdb_seq_str[i],
                canon_pos,
                canon_aa,
            ]) + "\n")

    # Pretty alignment
    pretty_path = out_prefix.with_suffix(".pretty.txt")
    write_pretty_alignment(pretty_path, ref_gapped=ref_gapped, pdb_gapped=pdb_gapped, width=args.line_width)

    seq_id = align.get_sequence_identity(ali)
    print("=== Done ===")
    print(f"PDB path: {pdb_path}")
    print(f"PDB label: {pdb_label}")
    print(f"Chain: {args.chain if args.chain is not None else 'ALL'}")
    print(f"Reference: {args.ref_fasta} ({ref_name})")
    print(f"Score: {ali.score}")
    print(f"Sequence identity: {seq_id:.4f}")
    print(f"A3M: {a3m_path}")
    print(f"Mapping: {mapping_path}")
    print(f"Pretty: {pretty_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
