#!/usr/bin/env python3
#TODO: verify that the structure (PDB or AF) can actually be downloaded
#TODO: replace biopython with biotite
#TODO: add check that sequences without any positive labels are not included
#TODO: handle PDB entries mapped to multiple UniProt IDs — either drop them or record which chains are intended

import argparse
import json
import os
import pickle
import re
import numpy as np
import networkx as nx
from Bio.PDB import PDBList, MMCIFParser, PPBuilder
from Bio.SeqUtils import seq1
from Bio.Align import PairwiseAligner
from scipy.spatial import cKDTree

import warnings
warnings.filterwarnings("ignore")

pdbl = PDBList()
parser = MMCIFParser()
ppb = PPBuilder()

PDB_DIR = ""  # set from --pdb_dir in main()

def extract_ec_number(protein_description):
    # EC number extraction — tries multiple locations in the protein description
    try:
        if 'recommendedName' in protein_description:
            if 'ecNumbers' in protein_description['recommendedName']:
                return protein_description['recommendedName']['ecNumbers'][0]['value']
    except (KeyError, IndexError):
        pass
    try:
        if 'alternativeNames' in protein_description:
            for alternative in protein_description['alternativeNames']:
                if 'ecNumbers' in alternative:
                    return alternative['ecNumbers'][0]['value']
    except (KeyError, IndexError):
        pass
    try:
        if 'includes' in protein_description:
            for include in protein_description['includes']:
                if 'recommendedName' in include and 'ecNumbers' in include['recommendedName']:
                    return include['recommendedName']['ecNumbers'][0]['value']
    except (KeyError, IndexError):
        pass
    try:
        if 'contains' in protein_description:
            for contain in protein_description['contains']:
                if 'recommendedName' in contain and 'ecNumbers' in contain['recommendedName']:
                    return contain['recommendedName']['ecNumbers'][0]['value']
    except (KeyError, IndexError):
        pass
    try:
        if 'fragments' in protein_description:
            for fragment in protein_description['fragments']:
                if 'ecNumbers' in fragment:
                    return fragment['ecNumbers'][0]['value']
    except (KeyError, IndexError):
        pass
    return None

def get_chains_sequences(structure, chain_ids):
    sequences = []
    for chain in structure[0]:
        if chain.id in chain_ids:
            sequence = ""
            positions = []
            for residue in chain:
                if residue.id[0] == ' ':
                    residue_name = residue.resname
                    residue_id = residue.id[1]
                    try:
                        one_letter_code = seq1(residue_name)
                    except Exception as e:
                        continue
                    sequence += one_letter_code
                    positions.append(residue_id)
            sequences.append((chain.id, sequence, positions))
    return sequences

def parse_biological_assembly(pdb_id):
    try:
        cif_filename = pdbl.retrieve_pdb_file(pdb_id, file_format='mmCif', pdir=PDB_DIR)
        structure = parser.get_structure(pdb_id, cif_filename)
        return structure
    except Exception as e:
        print(f"[DEBUG] Error parsing biological assembly for {pdb_id}: {e}")
        return None

def chains_are_in_contact(chain1, chain2, threshold_c_alpha=8.0, threshold_heavy=4.0):
    calpha_atoms1 = [atom for atom in chain1.get_atoms() if atom.name == 'CA']
    calpha_atoms2 = [atom for atom in chain2.get_atoms() if atom.name == 'CA']
    if not calpha_atoms1 or not calpha_atoms2:
        return False

    calpha_coords1 = np.array([atom.get_coord() for atom in calpha_atoms1])
    calpha_coords2 = np.array([atom.get_coord() for atom in calpha_atoms2])

    diff = calpha_coords1[:, np.newaxis, :] - calpha_coords2[np.newaxis, :, :]
    is_contact = (diff**2).sum(axis=-1) < threshold_c_alpha**2
    c_alpha_contacts = is_contact.sum()
    heavy_contacts = 0

    heavy_coords1 = []
    for residue in chain1.child_list:
        coords = np.array([atom.get_coord() for atom in residue if atom.element in ['C', 'N', 'O', 'S']])
        heavy_coords1.append(coords)
    heavy_coords2 = []
    for residue in chain2.child_list:
        coords = np.array([atom.get_coord() for atom in residue if atom.element in ['C', 'N', 'O', 'S']])
        heavy_coords2.append(coords)

    for i, j in np.argwhere(is_contact):
        coords_i = heavy_coords1[i]
        coords_j = heavy_coords2[j]
        if coords_i.size == 0 or coords_j.size == 0:
            is_heavy_contact = False
        else:
            tree = cKDTree(coords_j)
            neighbors = tree.query_ball_point(coords_i, r=threshold_heavy)
            is_heavy_contact = any(len(lst) > 0 for lst in neighbors)
        heavy_contacts += int(is_heavy_contact)

        if c_alpha_contacts >= 4 or heavy_contacts >= 10:
            return True

    return False

def choose_representative_chains(structure, chain_ids):
    contact_graph = nx.Graph()
    for chain_id1 in chain_ids:
        for chain_id2 in chain_ids:
            if chain_id1 != chain_id2:
                try:
                    chain1 = structure[0][chain_id1]
                    chain2 = structure[0][chain_id2]
                except Exception as e:
                    continue
                if chains_are_in_contact(chain1, chain2):
                    contact_graph.add_edge(chain_id1, chain_id2)
    if contact_graph.number_of_nodes() == 0:
        return [chain_ids[0]]
    largest_component = max(nx.connected_components(contact_graph), key=len)
    representative_chains = list(largest_component)
    return representative_chains

def map_catalytic_sites(uniprot_seq, pdb_seq, uniprot_catalytic_sites):
    aligner = PairwiseAligner()
    aligner.mode = "global"
    aligner.match_score = 1
    aligner.mismatch_score = -1
    aligner.open_gap_score = -0.75
    aligner.extend_gap_score = -0.5
    try:
        alignment = next(aligner.align(uniprot_seq, pdb_seq))
    except Exception as e:
        print(f"[DEBUG] Alignment error: {e}")
        return [0] * len(pdb_seq)
    pdb_catalytic_sites = [0] * len(pdb_seq)
    aligned_u = alignment.aligned[0]  # list of (start, end) tuples for UniProt
    aligned_p = alignment.aligned[1]  # list of (start, end) tuples for PDB
    for (u_start, u_end), (p_start, p_end) in zip(aligned_u, aligned_p):
        block_length = u_end - u_start
        if (p_end - p_start) != block_length:
            continue
        for i in range(block_length):
            pdb_index = p_start + i
            uniprot_index = u_start + i
            pdb_catalytic_sites[pdb_index] = uniprot_catalytic_sites[uniprot_index]
    return pdb_catalytic_sites

def output_sequence(output, protein_id, chain, positions, sequence, catalytic_sites):
    header = f">{protein_id}_{chain}"
    output.append(header)
    for position, amino_acid, is_catalytic_site in zip(positions, list(sequence), catalytic_sites):
        output.append(f"{chain} {position} {amino_acid} {is_catalytic_site}")

def analyze_PDB_reference(pdb_id, chain_ids):
    structure = parse_biological_assembly(pdb_id)
    if structure is None:
        return []
    if len(chain_ids) > 1:
        chain_ids = choose_representative_chains(structure, chain_ids)
    return get_chains_sequences(structure, chain_ids)

def parse_chain_ranges(chain_ranges: str):
    chain_dict = {}
    chains = re.split(r',\s*', chain_ranges)
    for chain in chains:
        try:
            chain_id, ranges = chain.split('=')
            start, end = map(int, ranges.split('-'))
            chain_dict[chain_id] = (start, end)
        except Exception as e:
            print(f"[DEBUG] Error parsing chain_ranges '{chain}': {e}")
    return chain_dict

def process_batch(data_batch, batch_num, output_dir):
    print(f"[INFO] Starting batch {batch_num}, proteins in batch: {len(data_batch)}")
    output = []
    proteins_table = dict()
    processed = 0

    for idx, result in enumerate(data_batch):
        try:
            primary_accession = result.get('primaryAccession', 'unknown')
            print(f"[DEBUG] Processing protein {primary_accession} ({idx+1}/{len(data_batch)})")
            uniprot_sequence = result['sequence']['value']
            features = result.get('features', [])
            uniprot_catalytic_sites = [0] * len(uniprot_sequence)
            ec_number = extract_ec_number(result.get('proteinDescription', {}))
            if ec_number is None:
                print(f"[DEBUG] EC number not found for {primary_accession}")
                ec_number = "not found"
            try:
                full_name = result['proteinDescription']['recommendedName']['fullName']['value']
            except Exception as e:
                print(f"[DEBUG] Full name not found for {primary_accession}: {e}")
                full_name = "not found"
                
            evidence_codes = set()
            for feature in features:
                if feature.get("type") == "Active site":
                    for ev in feature.get("evidences", []):
                        code = ev.get("evidenceCode")
                        if code:
                            evidence_codes.add(code)
                            
            proteins_table[primary_accession] = {
                "uniprot_id": primary_accession,
                "uniprot_sequence": uniprot_sequence,
                "EC_number": ec_number,
                "full_name": full_name,
                "pdb_ids": [],
                "evidence_codes": list(evidence_codes),
                "batch_number": batch_num
            }

            for feature in features:
                if feature.get('type') == 'Active site':
                    try:
                        start = feature['location']['start']['value']
                        end = feature['location']['end']['value']
                        for pos in range(start, end + 1):
                            if pos - 1 < len(uniprot_catalytic_sites):
                                uniprot_catalytic_sites[pos - 1] = 1
                    except Exception as e:
                        print(f"[DEBUG] Error processing active site for {primary_accession}: {e}")

            output_sequence(output, protein_id=primary_accession, chain='A',
                            positions=range(1, len(uniprot_sequence) + 1),
                            sequence=uniprot_sequence, catalytic_sites=uniprot_catalytic_sites)

            cross_references = result.get('uniProtKBCrossReferences', [])
            for cross_reference in cross_references:
                if cross_reference.get('database') == 'PDB':
                    try:
                        pdb_id = cross_reference['id']
                        # Specific fix: remap deprecated PDB entry 8V2I to its replacement 9BP6
                        if pdb_id == "8V2I":
                            pdb_id = "9BP6"
                        proteins_table[primary_accession]["pdb_ids"].append(pdb_id)
                        props = cross_reference.get('properties', [])
                        if len(props) < 3:
                            continue
                        chain_ranges_str = props[2].get('value', '')
                        reference_chains_properties = parse_chain_ranges(chain_ranges_str)
                        for chain_id, (fr, to) in reference_chains_properties.items():
                            fr, to = fr - 1, to - 1
                            sequences = analyze_PDB_reference(pdb_id, [chain_id])
                            if not sequences:
                                continue
                            for pdb_chain_id, pdb_sequence, pdb_positions in sequences:
                                pdb_catalytic_sites = map_catalytic_sites(
                                    uniprot_seq=uniprot_sequence[fr:to],
                                    pdb_seq=pdb_sequence,
                                    uniprot_catalytic_sites=uniprot_catalytic_sites[fr:to]
                                )
                                output_sequence(output, protein_id=pdb_id, chain=pdb_chain_id,
                                                positions=pdb_positions, sequence=pdb_sequence,
                                                catalytic_sites=pdb_catalytic_sites)
                    except Exception as e:
                        print(f"[DEBUG] Error processing PDB for {primary_accession}, pdb_id {cross_reference.get('id')}: {e}")

            processed += 1

        except Exception as e:
            print(f"[ERROR] Skipping protein {result.get('primaryAccession', 'unknown')} due to error: {e}")
            continue

    annotations_file = os.path.join(output_dir, f"batch{batch_num}_annotations.pkl")
    table_file = os.path.join(output_dir, f"batch{batch_num}_table.json")
    try:
        with open(annotations_file, 'wb') as f:
            pickle.dump(output, f)
        with open(table_file, 'w') as f:
            json.dump(proteins_table, f, indent=4)
        print(f"[INFO] Batch {batch_num} saved. Proteins processed: {processed}")
    except Exception as e:
        print(f"[ERROR] Error saving files for batch {batch_num}: {e}")

# ---------------------- Entry point ----------------------
def main():
    global PDB_DIR

    ap = argparse.ArgumentParser(description="Preprocess UniProt JSON into per-batch annotation pickles.")
    ap.add_argument("--input_file", required=True,
                    help="UniProt JSON export (see config.example.yaml: uniprot_file)")
    ap.add_argument("--output_dir", required=True,
                    help="Directory to write batch pickles (see config.example.yaml: batches_dir)")
    ap.add_argument("--pdb_dir", required=True,
                    help="Directory for cached PDB/AF .cif files (see config.example.yaml: pdb_dir)")
    ap.add_argument("--num_batches", type=int, default=100)
    args = ap.parse_args()

    PDB_DIR = args.pdb_dir

    try:
        with open(args.input_file) as f:
            results = json.load(f)
    except Exception as e:
        raise SystemExit(f"[FATAL] Cannot open input file {args.input_file}: {e}")

    total = len(results)
    batch_size = (total + args.num_batches - 1) // args.num_batches

    print(f"[INFO] Total proteins: {total}. Processing {args.num_batches} batches of ~{batch_size} each.")

    os.makedirs(args.output_dir, exist_ok=True)

    for i in range(args.num_batches):
        batch_start = i * batch_size
        batch_end = min((i + 1) * batch_size, total)
        data_batch = results[batch_start:batch_end]
        print(f"[INFO] Processing batch {i+1}: proteins {batch_start}–{batch_end}")
        process_batch(data_batch, i + 1, args.output_dir)
        print(f"[INFO] Batch {i+1} done.\n")

if __name__ == "__main__":
    main()
