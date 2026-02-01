#!/usr/bin/env python3
#TODO: проверка того, что структура (не важно, pdb или AF) вообще скачиваемая 
#TODO: заменить biopython на biotite
#TODO: добавить проверку, что не добавляются "пустые" последовательности, без положительных лейблов
#TODO: как-то разобраться с pdb, которые относятся к нескольким uniprot. или сразу их удалять, или указывать, какие именно цепи имеются ввиду

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

# Создаём объект для работы с PDB
pdbl = PDBList()
parser = MMCIFParser()
ppb = PPBuilder()

# ---------------------- Функции для извлечения информации ----------------------
def extract_ec_number(protein_description):
    # Исправленная версия извлечения EC номера
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
    # Обрабатываем первую модель (structure[0])
    for chain in structure[0]:
        if chain.id in chain_ids:
            sequence = ""
            positions = []
            for residue in chain:
                # Учитываем только стандартные остатки
                if residue.id[0] == ' ':
                    residue_name = residue.resname
                    residue_id = residue.id[1]
                    try:
                        one_letter_code = seq1(residue_name)
                    except Exception as e:
                        # Если не удалось получить однобуквенный код – пропускаем остаток
                        continue
                    sequence += one_letter_code
                    positions.append(residue_id)
            sequences.append((chain.id, sequence, positions))
    return sequences

def parse_biological_assembly(pdb_id):
    try:
        cif_filename = pdbl.retrieve_pdb_file(pdb_id, file_format='mmCif', pdir='/home/iscb/wolfson/annab4/Data/PDB_files')
        structure = parser.get_structure(pdb_id, cif_filename)
        return structure
    except Exception as e:
        print(f"[DEBUG] Ошибка при парсинге биологической сборки для {pdb_id}: {e}")
        return None

def chains_are_in_contact(chain1, chain2, threshold_c_alpha=8.0, threshold_heavy=4.0):
    # Извлекаем CA-атомы и их координаты
    calpha_atoms1 = [atom for atom in chain1.get_atoms() if atom.name == 'CA']
    calpha_atoms2 = [atom for atom in chain2.get_atoms() if atom.name == 'CA']
    if not calpha_atoms1 or not calpha_atoms2:
        return False

    calpha_coords1 = np.array([atom.get_coord() for atom in calpha_atoms1])
    calpha_coords2 = np.array([atom.get_coord() for atom in calpha_atoms2])
    
    # Вычисляем матрицу расстояний между CA-атомами и определяем пары, расстояние между которыми меньше порога
    diff = calpha_coords1[:, np.newaxis, :] - calpha_coords2[np.newaxis, :, :]
    is_contact = (diff**2).sum(axis=-1) < threshold_c_alpha**2
    c_alpha_contacts = is_contact.sum()
    heavy_contacts = 0

    # Предварительно вычисляем координаты тяжелых атомов (C, N, O, S) для каждого остатка в обоих цепях
    heavy_coords1 = []
    for residue in chain1.child_list:
        coords = np.array([atom.get_coord() for atom in residue if atom.element in ['C', 'N', 'O', 'S']])
        heavy_coords1.append(coords)
    heavy_coords2 = []
    for residue in chain2.child_list:
        coords = np.array([atom.get_coord() for atom in residue if atom.element in ['C', 'N', 'O', 'S']])
        heavy_coords2.append(coords)

    # Для каждой пары CA контактов проверяем наличие контакта между тяжелыми атомами соответствующих остатков
    for i, j in np.argwhere(is_contact):
        # Извлекаем массивы координат для остатка i из chain1 и остатка j из chain2
        coords_i = heavy_coords1[i]
        coords_j = heavy_coords2[j]
        # Если хотя бы один массив пуст, контакта не будет
        if coords_i.size == 0 or coords_j.size == 0:
            is_heavy_contact = False
        else:
            # Используем cKDTree для быстрого поиска пар тяжелых атомов в пределах порога
            tree = cKDTree(coords_j)
            # Ищем соседей для всех точек остатка i
            neighbors = tree.query_ball_point(coords_i, r=threshold_heavy)
            is_heavy_contact = any(len(lst) > 0 for lst in neighbors)
        heavy_contacts += int(is_heavy_contact)

        # Если достигнут порог по CA или по тяжелым атомам, возвращаем True
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
    # Используем PairwiseAligner вместо pairwise2
    aligner = PairwiseAligner()
    aligner.mode = "global"
    aligner.match_score = 1
    aligner.mismatch_score = -1
    aligner.open_gap_score = -0.75
    aligner.extend_gap_score = -0.5
    try:
        alignment = next(aligner.align(uniprot_seq, pdb_seq))
    except Exception as e:
        print(f"[DEBUG] Ошибка при выравнивании: {e}")
        return [0] * len(pdb_seq)
    pdb_catalytic_sites = [0] * len(pdb_seq)
    aligned_u = alignment.aligned[0]  # Список кортежей (start, end) для Uniprot
    aligned_p = alignment.aligned[1]  # Список кортежей (start, end) для PDB
    for (u_start, u_end), (p_start, p_end) in zip(aligned_u, aligned_p):
        block_length = u_end - u_start
        if (p_end - p_start) != block_length:
            # Если длины блока не совпадают, пропускаем этот блок
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
            print(f"[DEBUG] Ошибка при разборе chain_ranges '{chain}': {e}")
    return chain_dict

# ---------------------- Основная функция обработки батча ----------------------
def process_batch(data_batch, batch_num, output_dir):
    print(f"[INFO] Начало обработки батча {batch_num}, белков в батче: {len(data_batch)}")
    output = []  # для аннотаций
    proteins_table = dict()
    processed = 0

    for idx, result in enumerate(data_batch):
        try:
            primary_accession = result.get('primaryAccession', 'unknown')
            print(f"[DEBUG] Обработка белка {primary_accession} ({idx+1}/{len(data_batch)})")
            uniprot_sequence = result['sequence']['value']
            features = result.get('features', [])
            # Инициализируем catalytic_sites (0 для каждого остатка)
            uniprot_catalytic_sites = [0] * len(uniprot_sequence)
            # Извлекаем EC номер с помощью исправленной функции
            ec_number = extract_ec_number(result.get('proteinDescription', {}))
            if ec_number is None:
                print(f"[DEBUG] EC number не найден для {primary_accession}")
                ec_number = "not found"
            try:
                full_name = result['proteinDescription']['recommendedName']['fullName']['value']
            except Exception as e:
                print(f"[DEBUG] Полное имя не найдено для {primary_accession}: {e}")
                full_name = "not found"
                
            evidence_codes = set()
            for feature in features:
                if feature.get("type") == "Active site":
                    for ev in feature.get("evidences", []):
                        code = ev.get("evidenceCode")
                        if code:
                            evidence_codes.add(code)
                            
            # Записываем информацию о белке в таблицу
            proteins_table[primary_accession] = {
                "uniprot_id": primary_accession,
                "uniprot_sequence": uniprot_sequence,
                "EC_number": ec_number,
                "full_name": full_name,
                "pdb_ids": [],
                "evidence_codes": list(evidence_codes),
                "batch_number": batch_num
            }

            # Обработка активных сайтов
            for feature in features:
                if feature.get('type') == 'Active site':
                    try:
                        start = feature['location']['start']['value']
                        end = feature['location']['end']['value']
                        for pos in range(start, end + 1):
                            # Проверка выхода за границы
                            if pos - 1 < len(uniprot_catalytic_sites):
                                uniprot_catalytic_sites[pos - 1] = 1
                    except Exception as e:
                        print(f"[DEBUG] Ошибка при обработке active site для {primary_accession}: {e}")

            # Вывод аннотации для Uniprot последовательности (цепь A)
            output_sequence(output, protein_id=primary_accession, chain='A',
                            positions=range(1, len(uniprot_sequence) + 1),
                            sequence=uniprot_sequence, catalytic_sites=uniprot_catalytic_sites)

            # Обработка cross references для PDB
            cross_references = result.get('uniProtKBCrossReferences', [])
            for cross_reference in cross_references:
                if cross_reference.get('database') == 'PDB':
                    try:
                        pdb_id = cross_reference['id']
                        # Специфический фикс: если pdb_id == "8V2I", заменяем на "9BP6"
                        if pdb_id == "8V2I":
                            pdb_id = "9BP6"
                        proteins_table[primary_accession]["pdb_ids"].append(pdb_id)
                        # Пытаемся получить информацию о цепях из свойства с индексом 2
                        props = cross_reference.get('properties', [])
                        if len(props) < 3:
                            continue
                        chain_ranges_str = props[2].get('value', '')
                        reference_chains_properties = parse_chain_ranges(chain_ranges_str)
                        for chain_id, (fr, to) in reference_chains_properties.items():
                            # Преобразуем в индексацию с нуля
                            fr, to = fr - 1, to - 1
                            sequences = analyze_PDB_reference(pdb_id, [chain_id])
                            if not sequences:
                                continue
                            for pdb_chain_id, pdb_sequence, pdb_positions in sequences:
                                # Маппим catalytic sites для участка белка
                                pdb_catalytic_sites = map_catalytic_sites(
                                    uniprot_seq=uniprot_sequence[fr:to],
                                    pdb_seq=pdb_sequence,
                                    uniprot_catalytic_sites=uniprot_catalytic_sites[fr:to]
                                )
                                output_sequence(output, protein_id=pdb_id, chain=pdb_chain_id,
                                                positions=pdb_positions, sequence=pdb_sequence,
                                                catalytic_sites=pdb_catalytic_sites)
                    except Exception as e:
                        print(f"[DEBUG] Ошибка при обработке PDB для {primary_accession}, pdb_id {cross_reference.get('id')}: {e}")

            processed += 1

        except Exception as e:
            print(f"[ERROR] Пропуск белка {result.get('primaryAccession', 'unknown')} из-за ошибки: {e}")
            continue

    # Сохраняем файлы батча
    annotations_file = os.path.join(output_dir, f"batch{batch_num}_annotations.pkl")
    table_file = os.path.join(output_dir, f"batch{batch_num}_table.json")
    try:
        with open(annotations_file, 'wb') as f:
            pickle.dump(output, f)
        with open(table_file, 'w') as f:
            json.dump(proteins_table, f, indent=4)
        print(f"[INFO] Батч {batch_num} успешно сохранён. Обработано белков: {processed}")
    except Exception as e:
        print(f"[ERROR] Ошибка при сохранении файлов для батча {batch_num}: {e}")

# ---------------------- Основной блок ----------------------
def main():
    input_file = "/home/iscb/wolfson/annab4/uniprot_files/filtered_all_protein.json"
    try:
        with open(input_file) as f:
            results = json.load(f)
    except Exception as e:
        print(f"[FATAL] Не удалось открыть входной файл {input_file}: {e}")
        return

    total = len(results)
    num_batches = 100
    batch_size = (total + num_batches - 1) // num_batches  # округление вверх

    print(f"[INFO] Всего белков: {total}. Будет обработано {num_batches} батчей, по ~{batch_size} белков в каждом.")
    
    output_dir = os.path.join("/home/iscb/wolfson/annab4/DB/all_proteins", "batches")
    os.makedirs(output_dir, exist_ok=True)
    
    for i in range(0, num_batches):
        batch_start = i * batch_size
        batch_end = min((i + 1) * batch_size, total)
        data_batch = results[batch_start:batch_end]
        print(f"[INFO] Обработка батча {i+1}: белки с {batch_start} до {batch_end}")
        process_batch(data_batch, i + 1, output_dir)
        print(f"[INFO] Завершён батч {i+1}\n")

if __name__ == "__main__":
    main()
