#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Clean up an A3M/FASTA MSA:
- Replace '.' with '-'.
- Remove lowercase insertion letters.
- Keep only sequences that match the query length (first record).
"""

import argparse
import re
from typing import Iterable, Tuple


def read_fasta(path: str) -> Iterable[Tuple[str, str]]:
    name = None
    seq = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if name is not None:
                    yield name, "".join(seq)
                name, seq = line, []
            else:
                seq.append(line)
        if name is not None:
            yield name, "".join(seq)


def clean_sequences(records: Iterable[Tuple[str, str]]) -> list[Tuple[str, str]]:
    cleaned = []
    for header, seq in records:
        seq = seq.replace(".", "-")
        seq = re.sub(r"[a-z]", "", seq)
        cleaned.append((header, seq))
    return cleaned


def write_fasta(path: str, records: Iterable[Tuple[str, str]]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        for header, seq in records:
            handle.write(f"{header}\n")
            for i in range(0, len(seq), 80):
                handle.write(f"{seq[i:i + 80]}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean up an MSA file in A3M/FASTA format.")
    parser.add_argument("--in_fasta", required=True, help="Input MSA file.")
    parser.add_argument("--out_fasta", required=True, help="Output cleaned MSA file.")
    args = parser.parse_args()

    cleaned = clean_sequences(read_fasta(args.in_fasta))
    if not cleaned:
        raise SystemExit("No sequences found in input.")

    lengths = {len(seq) for _, seq in cleaned}
    print(f"[INFO] unique lengths after cleanup: {sorted(lengths)[:10]} (total {len(lengths)})")

    query_len = len(cleaned[0][1])
    filtered = [(h, s) for h, s in cleaned if len(s) == query_len]
    print(f"[INFO] keeping {len(filtered)}/{len(cleaned)} sequences with length == {query_len}")

    write_fasta(args.out_fasta, filtered)
    print(f"[OK] wrote {args.out_fasta}")


if __name__ == "__main__":
    main()
