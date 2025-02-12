import json
import os
import pandas as pd

from Bio.PDB import PDBList, PPBuilder, MMCIFParser, Chain
from Bio.SeqUtils import seq1
from Bio import pairwise2

import networkx as nx
import numpy as np

import pickle
import re

pdbl = PDBList()
parser = MMCIFParser()
ppb = PPBuilder()


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
                    one_letter_code = seq1(residue_name)
                    sequence += one_letter_code
                    positions.append(residue_id)
            sequences.append((chain.id, sequence, positions))
    return sequences


def parse_biological_assembly(pdb_id):
    cif_filename = pdbl.retrieve_pdb_file(pdb_id, file_format='mmCif', pdir='../../pdb_files/')
    structure = parser.get_structure(pdb_id, cif_filename)
    return structure


def chains_are_in_contact(chain1, chain2, threshold_c_alpha=8.0, threshold_heavy=4.0):
    calpha_atoms1 = [atom for atom in chain1.get_atoms() if atom.name == 'CA']
    calpha_atoms2 = [atom for atom in chain2.get_atoms() if atom.name == 'CA']

    calpha_coordinates_atoms1 = np.array([atom.get_coord() for atom in calpha_atoms1])
    calpha_coordinates_atoms2 = np.array([atom.get_coord() for atom in calpha_atoms2])

    is_contact = ((calpha_coordinates_atoms1[:, np.newaxis] - calpha_coordinates_atoms2[np.newaxis, :])**2).sum(-1) < threshold_c_alpha**2
    c_alpha_contacts = is_contact.sum()
    heavy_contacts = 0

    for i, j in np.argwhere(is_contact):
        residue_i = chain1.child_list[i]
        residue_j = chain2.child_list[j]
        heavy_atoms_i = [atom for atom in residue_i if atom.element in ['C', 'N', 'O', 'S']]
        heavy_atoms_j = [atom for atom in residue_j if atom.element in ['C', 'N', 'O', 'S']]

        heavy_coordinates_atoms_i = np.array([atom.get_coord() for atom in heavy_atoms_i])
        heavy_coordinates_atoms_j = np.array([atom.get_coord() for atom in heavy_atoms_j])

        is_heavy_contact = (((heavy_coordinates_atoms_i[:, np.newaxis] - heavy_coordinates_atoms_j[np.newaxis, :])**2).sum(-1) < threshold_heavy**2).max()
        heavy_contacts += is_heavy_contact

        if c_alpha_contacts >= 4 or heavy_contacts >= 10:
            return True

    return False


def choose_representative_chains(structure, chain_ids):
    contact_graph = nx.Graph()

    for chain_id1 in chain_ids:
        for chain_id2 in chain_ids:
            if chain_id1 != chain_id2:
                chain1 = structure[0][chain_id1]
                chain2 = structure[0][chain_id2]
                if chains_are_in_contact(chain1, chain2):
                    contact_graph.add_edge(chain_id1, chain_id2)

    if contact_graph.number_of_nodes() == 0:
        return [chain_ids[0]] 

    largest_component = max(nx.connected_components(contact_graph), key=len)
    representative_chains = list(largest_component) #????
    return representative_chains


def map_catalytic_sites(uniprot_seq, pdb_seq, uniprot_catalytic_sites):
    gap_open_penalty = 0.5
    gap_extend_penalty = 0.75
    
    alignments = pairwise2.align.globalms(uniprot_seq, pdb_seq, match=1, mismatch=-1, open=-0.75, extend=-0.5) #gap penalties should be non-positive
    best_alignment = alignments[0]
    uniprot_aligned, pdb_aligned = best_alignment[:2]
    pdb_catalytic_sites = [0] * len(pdb_seq)

    uniprot_index, pdb_index = 0, 0
    for a, b in zip(uniprot_aligned, pdb_aligned):
        if a != '-' and b != '-':
            pdb_catalytic_sites[pdb_index] = uniprot_catalytic_sites[uniprot_index]
            uniprot_index += 1
            pdb_index += 1
        elif a == '-' and b != '-':
            pdb_index += 1
        elif a != '-' and b == '-':
            uniprot_index += 1
    return pdb_catalytic_sites


def output_sequence(output, protein_id, chain, positions, sequence, catalytic_sites):
    output.append(f">{protein_id}_{chain}")
    for position, amino_acid, is_catalytic_site in zip(positions, list(sequence), catalytic_sites):
        output.append(f"{chain} {position} {amino_acid} {is_catalytic_site}")    


def analyze_PDB_reference(pdb_id, chain_ids):
    structure = parse_biological_assembly(pdb_id)
    if len(chain_ids) > 1:
        chain_ids = choose_representative_chains(structure, chain_ids)
    return get_chains_sequences(structure, chain_ids)


def parse_chain_ranges(chain_ranges: str):
    chain_dict = {}
    chains = re.split(r',\s*', chain_ranges)
    for chain in chains:
        chain_id, ranges = chain.split('=')
        start, end = map(int, ranges.split('-'))
        chain_dict[chain_id] = (start, end)
    return chain_dict


def process_batch(data_batch, batch_num):
    output = []

    for result in data_batch:
        uniprot_sequence = result['sequence']['value']
        features = result['features']
        primary_accession = result['primaryAccession']
        uniprot_catalytic_sites = [0] * len(uniprot_sequence)

        for feature in features:
            if feature['type'] == 'Active site':
                start = feature['location']['start']['value']
                end = feature['location']['end']['value']
                for pos in range(start, end+1):
                    uniprot_catalytic_sites[pos-1] = 1

        output_sequence(output, protein_id=primary_accession, chain='A', positions=range(1, len(uniprot_sequence)+1), 
                        sequence=uniprot_sequence, catalytic_sites=uniprot_catalytic_sites)
        
        cross_references = result['uniProtKBCrossReferences']
        for cross_reference in cross_references:
            if cross_reference['database'] == 'PDB':
                pdb_id = cross_reference['id']
                reference_chains_properties = parse_chain_ranges(cross_reference['properties'][2]['value'])
                for chain_id, (fr, to) in reference_chains_properties.items():
                    fr, to = fr - 1, to - 1
                    sequences = analyze_PDB_reference(pdb_id, [chain_id])
                    if not sequences:
                        continue 
                    for chain_id, pdb_sequence, pdb_positions in sequences:
                        pdb_catalytic_sites = map_catalytic_sites(uniprot_sequence[fr:to], pdb_sequence, uniprot_catalytic_sites[fr:to])
                        output_sequence(output, protein_id=pdb_id, chain=chain_id, positions=pdb_positions, 
                                        sequence=pdb_sequence, catalytic_sites=pdb_catalytic_sites)

    output_file = f'../../DB/batch_{batch_num}_annotations.pkl'
    with open(output_file, 'wb') as f:
        pickle.dump(output, f)


if __name__ == "__main__":
    with open('/specific/a/home/cc/students/cs/annab4/uniprot_files/active_sites_proteins.json') as f:
        data = json.load(f)

    batch_size = 290
    num_batches = (len(data['results']) + batch_size - 1) // batch_size  

    for i in range(num_batches):
        batch_start = i * batch_size
        batch_end = min((i + 1) * batch_size, len(data['results']))
        data_batch = data['results'][batch_start:batch_end]
        process_batch(data_batch, i + 1)
