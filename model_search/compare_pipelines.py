import pickle, sys
sys.path.insert(0, '/home/iscb/wolfson/annab4/main_scannet/ScanNet_Ub')

import utilities.io_utils as io_utils
import utilities.dataset_utils as dataset_utils

PIPELINE_DIR = '/home/iscb/wolfson/annab4/main_scannet/ScanNet_Ub/pipelines/'
SPLIT_DIR    = '/home/iscb/wolfson/annab4/DB/all_proteins/cross_validation_chem/weight_based_v9/'

for i in range(1, 6):
    origins, _, _, _ = dataset_utils.read_labels(f'{SPLIT_DIR}split{i}.txt')
    origins_set = set(origins)

    esm_file  = f'{PIPELINE_DIR}catalytic_sites_ESM2_split{i}_ESM2_3B_MLP_pipeline_ESM2.data'
    scan_file = (f'{PIPELINE_DIR}catalytic_sites_split{i}_esm2_3B_layer36'
                f'_pipeline_ScanNet_aa-esm2_atom-valency_frames-triplet_sidechain_Beff-500.data')

    esm_env  = io_utils.load_pickle(esm_file)
    scan_env = io_utils.load_pickle(scan_file)

    esm_failed  = set(esm_env['failed_samples'])
    scan_failed = set(scan_env['failed_samples'])

    esm_ok  = set(origins[j] for j in range(len(origins)) if j not in esm_failed)
    scan_ok = set(origins[j] for j in range(len(origins)) if j not in scan_failed)

    only_esm  = esm_ok  - scan_ok   # ESM2 pipeline kept, ScanNet dropped
    only_scan = scan_ok - esm_ok    # ScanNet kept, ESM2 dropped

    print(f'\n=== split{i} ===')
    print(f'  total: {len(origins)}')
    print(f'  ESM2 ok: {len(esm_ok)}  |  ScanNet ok: {len(scan_ok)}')
    print(f'  only in ESM2 (ScanNet dropped): {len(only_esm)}')
    print(f'  only in ScanNet (ESM2 dropped): {len(only_scan)}')
    if only_esm:
        print(f'  examples only_esm: {list(only_esm)[:5]}')
    if only_scan:
        print(f'  examples only_scan: {list(only_scan)[:5]}')