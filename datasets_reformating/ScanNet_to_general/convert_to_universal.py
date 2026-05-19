"""
Convert ScanNet-format catalytic sites dataset to a universal CSV format.

Input:
  - dataset.csv         : sequence metadata, cluster assignments, weights, split
  - split{1-5}.txt      : residue-wise catalytic site labels (chain resnum aa label)
  - all_protein_table_modified.json : UniProt metadata (sequence, EC, pdb_ids, evidence_codes)
  - pdb_to_uniprot.json : PDB ID -> UniProt ID mapping

Output:
  - catalytic_sites_dataset.csv : one row per protein, all info combined
"""

import json
import pandas as pd
from pathlib import Path

BASE = Path('/home/iscb/wolfson/annab4/DB')
SPLIT_DIR = BASE / 'all_proteins/cross_validation_chem/weight_based_v9'
OUT_FILE = Path('catalytic_sites_dataset.csv')


def parse_split_files(split_dir: Path) -> dict:
    """Parse all split*.txt files into {seq_id: {sequence, catalytic_residues}}."""
    result = {}
    for i in range(1, 6):
        current_id = None
        residues = []
        with open(split_dir / f'split{i}.txt') as f:
            for line in f:
                line = line.rstrip()
                if line.startswith('>'):
                    if current_id is not None:
                        result[current_id] = _process_residues(residues)
                    current_id = line[1:]
                    residues = []
                else:
                    parts = line.split()
                    if len(parts) == 4:
                        residues.append(parts)
            if current_id is not None:
                result[current_id] = _process_residues(residues)
    return result


def _process_residues(residues: list) -> dict:
    sequence = ''.join(r[2] for r in residues)
    catalytic = ','.join(f'{r[0]}:{r[1]}' for r in residues if r[3] == '1')
    return {'sequence': sequence, 'catalytic_residues': catalytic}


def lookup_entry(base_id: str, prot_table: dict, pdb_to_uni: dict) -> tuple[str, dict]:
    """Return (uniprot_id, table_entry) for a sequence base ID."""
    if base_id in prot_table:
        entry = prot_table[base_id]
        return entry['uniprot_id'], entry
    uniprot_id = pdb_to_uni[base_id]
    return uniprot_id, prot_table[uniprot_id]


def serialize_list(value) -> str:
    if isinstance(value, list):
        return ';'.join(str(v) for v in value)
    return str(value) if value else ''


def main():
    print('Loading metadata...')
    df = pd.read_csv(SPLIT_DIR / 'dataset.csv')

    with open(BASE / 'all_protein_table_modified.json') as f:
        prot_table = json.load(f)
    with open(BASE / 'all_proteins/pdb_to_uniprot.json') as f:
        pdb_to_uni = json.load(f)

    print('Parsing split files...')
    split_data = parse_split_files(SPLIT_DIR)

    print('Building output table...')
    rows = []
    for _, row in df.iterrows():
        seq_id = row['Sequence_ID']
        base = seq_id.rsplit('_', 1)[0]

        uniprot_id, entry = lookup_entry(base, prot_table, pdb_to_uni)
        seq_info = split_data[seq_id]

        rows.append({
            'sequence_id':       seq_id,
            'uniprot_id':        uniprot_id,
            'sequence':          seq_info['sequence'],
            'catalytic_residues': seq_info['catalytic_residues'],
            'split':             int(row['Set_Type'].replace('split', '')),
            'EC_number':         row['EC_number'],
            'protein_name':      row['full_name'],
            'cluster_1':         row['Cluster_1'],
            'cluster_2':         row['Cluster_2'],
            'component_id':      row['Component_ID'],
            'W_cluster_2':       row['W_Cluster_2'],
            'W_cluster_1':       row['W_Cluster_1'],
            'W_sequence':        row['W_Sequence'],
            'W_structure':       row['W_Structure'],
            'pdb_ids':           serialize_list(entry.get('pdb_ids', [])),
            'evidence_codes':    serialize_list(entry.get('evidence_codes', [])),
        })

    out_df = pd.DataFrame(rows)
    out_df.to_csv(OUT_FILE, index=False)
    print(f'Done. Written {len(out_df)} rows to {OUT_FILE}')
    print(out_df.dtypes)
    print(out_df.head(3).to_string())


if __name__ == '__main__':
    main()
