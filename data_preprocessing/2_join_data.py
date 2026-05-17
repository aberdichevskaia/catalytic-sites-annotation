import argparse
import json
import os
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord


def main():
    ap = argparse.ArgumentParser(
        description="Aggregate per-batch JSON tables into a single protein table and FASTA."
    )
    ap.add_argument("--batches_dir", required=True,
                    help="Directory containing batch*_table.json files (see config.example.yaml: batches_dir)")
    ap.add_argument("--output_table", required=True,
                    help="Output path for aggregated JSON table (see config.example.yaml: protein_table)")
    ap.add_argument("--output_fasta", required=True,
                    help="Output path for FASTA of UniProt sequences (see config.example.yaml: protein_fasta)")
    ap.add_argument("--num_batches", type=int, default=100)
    args = ap.parse_args()

    records = {}
    for i in range(1, args.num_batches + 1):
        path = os.path.join(args.batches_dir, f"batch{i}_table.json")
        try:
            with open(path) as f:
                data = json.load(f)
            for record in data:
                data[record]["batch_number"] = i
            records.update(data)
        except Exception as e:
            print(e)

    with open(args.output_table, "w") as f:
        json.dump(records, f)

    sequences = []
    for uniprot_id, data in records.items():
        seq_record = SeqRecord(Seq(data["uniprot_sequence"]), id=uniprot_id)
        sequences.append(seq_record)

    with open(args.output_fasta, "w") as f:
        SeqIO.write(sequences, f, "fasta")

    print(f"Written {len(records)} records to {args.output_table}")
    print(f"Written {len(sequences)} sequences to {args.output_fasta}")


if __name__ == "__main__":
    main()
