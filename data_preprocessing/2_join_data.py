import json
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

records = dict()

for i in range(1, 101):
    try:
        with open(f'/home/iscb/wolfson/annab4/DB/all_proteins/batches/batch{i}_table.json', 'r') as f:
            data = json.load(f)
            for record in data:
                data[record]['batch_number'] = i
            records.update(data)
    except Exception as e:
        print(e)

with open(f'/home/iscb/wolfson/annab4/DB/all_protein_table.json', 'w') as f:
    json.dump(records, f)
    
sequences = []

for uniprot_id, data in records.items():
    seq_record = SeqRecord(
        Seq(data["uniprot_sequence"]),
        id=uniprot_id
    )
    sequences.append(seq_record)
    
    
with open("/home/iscb/wolfson/annab4/DB/all_protein_sequences.fasta", "w") as f:
    SeqIO.write(sequences, f, "fasta")
