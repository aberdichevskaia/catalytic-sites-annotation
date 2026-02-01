#!/usr/bin/env python3
import os
import json

def parse_annotation(file_path):
    """
    Parse a .txt annotation file into a dict:
      key = (protein_id, chain)
      value = list of (res_id, label)
    """
    annotations = {}
    with open(file_path, "r") as f:
        current_id, current_chain = None, None
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                parts = line[1:].split("_")
                current_id = parts[0]
                current_chain = parts[1] if len(parts) > 1 else "A"
                annotations[(current_id, current_chain)] = []
            else:
                cols = line.split()
                if len(cols) == 4:
                    _, _, res_id, label = cols
                    annotations[(current_id, current_chain)].append(
                        (res_id, int(label))
                    )
    return annotations

def main():
    base_dir = "/home/iscb/wolfson/annab4/DB/all_proteins/cross_validation"
    summary_by_split = {}

    for i in range(1, 6):
        split_name = f"split{i}"
        txt_path = os.path.join(base_dir, f"{split_name}.txt")
        ann = parse_annotation(txt_path)

        summary = {}
        for (prot_id, chain), entries in ann.items():
            seqid = f"{prot_id}_{chain}"
            seq_len = len(entries)
            pos_count = sum(label for _, label in entries)
            summary[seqid] = {
                "sequence_len": seq_len,
                "number_of_positive_labels": pos_count
            }

        summary_by_split[split_name] = summary

    out_json = "/home/iscb/wolfson/annab4/DB/all_proteins/cross_validation/annotations_summary.json"
    with open(out_json, "w") as jf:
        json.dump(summary_by_split, jf, indent=2, ensure_ascii=False)
    print(f"Saved summary to {out_json}\n")

    for split_name, summary in summary_by_split.items():
        sorted_by_len = sorted(
            summary.items(),
            key=lambda item: item[1]["sequence_len"]
        )
        top10_short = [seqid for seqid, _ in sorted_by_len[:10]]

        no_pos = [
            seqid
            for seqid, info in summary.items()
            if info["number_of_positive_labels"] == 0
        ]

        print(f"{split_name}:")
        print("  Top-10 shortest sequences:")
        print("    " + ", ".join(top10_short))
        print("  Sequences with no positive labels:")
        print("    " + (", ".join(no_pos) if no_pos else "— нет ни одного"))
        print()

if __name__ == "__main__":
    main()
