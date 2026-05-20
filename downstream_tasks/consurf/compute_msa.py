#!/usr/bin/env python3
"""
Pre-compute MSAs for all proteins in one or more structure directories.

For each protein:
  1. Extract chain A sequence from PDB (ATOM records).
  2. MMseqs2 search against UniRef50 (same parameters as stand_alone_consurf.py).
  3. Filter hits by identity / length thresholds; keep top --max_homologs by e-value.
  4. Run MAFFT; write <out_msa_dir>/<protein_id>.fasta

The resulting MSA files are compatible with stand_alone_consurf.py via:
    python stand_alone_consurf.py --structure X.pdb --chain A \\
        --msa <out_msa_dir>/<protein_id>.fasta --query query

Run in tmux:
    python compute_msa.py \\
        --structures_dirs /path/AFDB/Human_v2 /path/PDB_Human_Isoforms \\
        --out_msa_dir /path/to/output \\
        --db /path/to/UniRef50 \\
        --cores 32 \\
        --only_missing
"""

import argparse
import logging
import os
import re
import subprocess
import sys
import tempfile
from glob import glob

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

MMSEQS = "/home/iscb/wolfson/annab4/miniconda3/envs/consurf_env/bin/mmseqs"
MAFFT  = "mafft"

# MMseqs2 search parameters — same as stand_alone_consurf.py
MMSEQS_SENSITIVITY = "5.7"
MMSEQS_MAX_SEQS    = "10000000"
MIN_SEQ_ID         = 0.35   # --min-seq-id
MAX_SEQ_ID         = 0.95   # --max-seq-id
COVERAGE           = "0.6"  # -c  (minimum alignment coverage)

# Hit filtering thresholds — same as ConSurf defaults
REDUNDANCY_RATE = 95.0   # reject if pident >= this (too similar to query)
MIN_ID_PCT      = 35.0   # reject if pident < this
MIN_LEN_FRAC    = 0.60   # reject if hit length < this * query_length


# ── structure collection ──────────────────────────────────────────────────────

def all_structures(structures_dirs):
    """
    Return list of (protein_id, path) from one or more directories.
    For AFDB files that have both .pdb and .cif, .pdb is preferred.
    protein_id is the full filename stem (preserves isoform numbers).
    """
    seen = {}  # stem -> path
    for d in structures_dirs:
        for p in glob(os.path.join(d, "*.pdb")) + glob(os.path.join(d, "*.cif")):
            stem = os.path.splitext(os.path.basename(p))[0]
            if stem not in seen or p.lower().endswith(".pdb"):
                seen[stem] = p
    return sorted(seen.items())


# ── sequence extraction ───────────────────────────────────────────────────────

_AA3 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "SEC": "U", "PYL": "O",
}


def extract_sequence(pdb_path, chain_id="A"):
    seen = set()
    seq = []
    with open(pdb_path) as fh:
        for line in fh:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue
            if line[21] != chain_id:
                continue
            res_name = line[17:20].strip()
            res_key  = line[22:27].strip()  # seq number + insertion code
            if res_key in seen:
                continue
            aa = _AA3.get(res_name)
            if aa is None:
                continue
            seen.add(res_key)
            seq.append(aa)
    if not seq:
        raise ValueError(f"no ATOM residues for chain {chain_id} in {pdb_path}")
    return "".join(seq)


# ── MMseqs2 + MAFFT pipeline ──────────────────────────────────────────────────

def run_cmd(cmd, cwd):
    result = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"command failed:\n{cmd}\nstderr:\n{result.stderr[:500]}")


def mmseqs_search(query_fasta, db, tmp, cores):
    aln_out = os.path.join(tmp, "hits.tsv")
    cmds = [
        f"{MMSEQS} createdb {query_fasta} {tmp}/query_db",
        (
            f"{MMSEQS} search {tmp}/query_db {db} {tmp}/aln {tmp}/tmp2"
            f" -a 1 --max-seqs {MMSEQS_MAX_SEQS} -s {MMSEQS_SENSITIVITY}"
            f" --min-seq-id {MIN_SEQ_ID} --max-seq-id {MAX_SEQ_ID}"
            f" -c {COVERAGE} --filter-hits 1 --threads {cores}"
        ),
        (
            f"{MMSEQS} convertalis {tmp}/query_db {db} {tmp}/aln {aln_out}"
            ' --format-output "target,evalue,taln,tstart,tend,pident,theader"'
            f" --threads {cores}"
        ),
    ]
    for cmd in cmds:
        run_cmd(cmd, cwd=tmp)
    return aln_out


def parse_hits(aln_out, query_len, max_homologs):
    hits = []
    with open(aln_out) as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 6:
                continue
            target = parts[0]
            evalue = parts[1]
            taln   = parts[2]
            pident = float(parts[5])

            # strip gap chars and lowercase insertions (A3M style)
            seq = re.sub(r"[a-z\-]", "", taln)

            if not seq:
                continue
            if pident >= REDUNDANCY_RATE or pident < MIN_ID_PCT:
                continue
            if len(seq) < MIN_LEN_FRAC * query_len:
                continue

            try:
                ev = float(evalue)
            except ValueError:
                ev = float("inf")
            hits.append((ev, target, seq))

    hits.sort(key=lambda x: x[0])
    return hits[:max_homologs]


def run_mafft(input_fasta, output_fasta):
    # --auto: MAFFT picks the algorithm based on number of sequences;
    # fast enough for batch (FFT-NS-2 for large sets, L-INS-i for small).
    cmd = f"{MAFFT} --auto --quiet {input_fasta} > {output_fasta}"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0 or not os.path.exists(output_fasta) or os.path.getsize(output_fasta) == 0:
        raise RuntimeError(f"MAFFT failed: {result.stderr[:300]}")


def build_msa(pdb_path, chain, db, out_file, max_homologs, cores):
    log.info("  extracting sequence from %s chain %s", os.path.basename(pdb_path), chain)
    seq = extract_sequence(pdb_path, chain)
    log.info("  query length: %d", len(seq))

    with tempfile.TemporaryDirectory(prefix="cmsa_") as tmp:
        query_fasta = os.path.join(tmp, "query.fasta")
        with open(query_fasta, "w") as fh:
            fh.write(f">query\n{seq}\n")

        log.info("  MMseqs2 search")
        aln_out = mmseqs_search(query_fasta, db, tmp, cores)
        hits = parse_hits(aln_out, len(seq), max_homologs)
        log.info("  %d homologs selected", len(hits))

        pre_msa = os.path.join(tmp, "pre_msa.fasta")
        with open(pre_msa, "w") as fh:
            fh.write(f">query\n{seq}\n")
            for i, (_, target, hseq) in enumerate(hits):
                fh.write(f">h{i}|{target}\n{hseq}\n")

        log.info("  MAFFT")
        run_mafft(pre_msa, out_file)
    return len(hits)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--structures_dirs", nargs="+", required=True,
                    help="One or more directories with .pdb / .cif files")
    ap.add_argument("--out_msa_dir", required=True,
                    help="Output directory for MSA .fasta files")
    ap.add_argument("--db", default="/home/iscb/wolfson/sequence_database/MMSEQS/UniRef50",
                    help="MMseqs2 database path (default: UniRef50)")
    ap.add_argument("--chain",        default="A")
    ap.add_argument("--max_homologs", type=int, default=150)
    ap.add_argument("--cores",        type=int, default=4,
                    help="MMseqs2 threads")
    ap.add_argument("--only_missing", action="store_true",
                    help="Skip proteins whose MSA file already exists")
    args = ap.parse_args()

    os.makedirs(args.out_msa_dir, exist_ok=True)

    proteins = all_structures(args.structures_dirs)
    if not proteins:
        log.error("no structures found in %s", args.structures_dirs)
        sys.exit(1)
    log.info("found %d proteins", len(proteins))

    total = done = skipped = failed = 0
    for protein_id, pdb_path in proteins:
        total += 1
        out_file = os.path.join(args.out_msa_dir, f"{protein_id}.fasta")

        if args.only_missing and os.path.exists(out_file) and os.path.getsize(out_file) > 0:
            skipped += 1
            continue

        try:
            log.info("[%d/%d] %s", total, len(proteins), protein_id)
            n_hits = build_msa(pdb_path, args.chain, args.db, out_file,
                               args.max_homologs, args.cores)
            log.info("  -> %d homologs, saved to %s", n_hits, out_file)
            done += 1
        except Exception as e:
            log.error("  failed: %s", e)
            failed += 1

    log.info("total: %d | done: %d | skipped: %d | failed: %d",
             total, done, skipped, failed)


if __name__ == "__main__":
    main()
