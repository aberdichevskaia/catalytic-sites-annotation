#!/usr/bin/env python3
import os
import pandas as pd

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

def get_no_positive_seqids(annotations):
    """
    Return set of sequence IDs (with chain) that have zero positive labels.
    """
    no_pos = set()
    for (pid, chain), entries in annotations.items():
        if sum(label for _, label in entries) == 0:
            no_pos.add(f"{pid}_{chain}")
    return no_pos

def main():
    base_dir = "/home/iscb/wolfson/annab4/DB/all_proteins/cross_validation_chem"
    save_dir = "/home/iscb/wolfson/annab4/DB/all_proteins/cross_validation_chem/cleaned"
    csv_path = os.path.join(base_dir, "dataset.csv")

    # load dataset.csv
    df = pd.read_csv(csv_path)

    splits = [f"split{i}" for i in range(1, 6)]
    all_to_remove = set()

    # process each split
    for split in splits:
        txt_file = os.path.join(base_dir, f"{split}.txt")
        annotations = parse_annotation(txt_file)
        no_pos_ids = get_no_positive_seqids(annotations)
        all_to_remove.update(no_pos_ids)

        # rewrite splitX.txt without these seqids
        out_lines = []
        skip = False
        with open(txt_file, "r") as fin:
            for line in fin:
                if line.startswith(">"):
                    seqid = line[1:].strip()
                    skip = (seqid in no_pos_ids)
                    if not skip:
                        out_lines.append(line)
                else:
                    if not skip:
                        out_lines.append(line)
        output_txt = os.path.join(save_dir, f"{split}.txt")
        with open(output_txt, "w") as fout:
            fout.writelines(out_lines)
        print(f"{split}: removed {len(no_pos_ids)} sequences with no positives")

    # drop from dataset.csv
    before = len(df)
    df = df[~df["Sequence_ID"].isin(all_to_remove)]
    save_csv_path = os.path.join(save_dir, "dataset.csv")
    df.to_csv(save_csv_path, index=False)
    print(f"dataset.csv: dropped {before - len(df)} rows; remaining {len(df)} rows")

if __name__ == "__main__":
    main()
