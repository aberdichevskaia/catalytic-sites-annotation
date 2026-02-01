import os

def parse_annotation_file(path):
    ann = {}
    with open(path) as f:
        current_id = None
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                current_id = line[1:]
                ann[current_id] = []
            else:
                cols = line.split()
                chain = cols[0]
                pos = int(cols[1])
                aa = cols[2]
                lab = int(cols[3])  
                ann[current_id].append((chain, pos, aa, lab))
    return ann

def catalytic_label_sum(arr):
    return sum([label for chain, pos, aa, label in arr])

def save_annotations_ordered(ids, data, output_file):
    with open(output_file, 'w') as f:
        for id_chain in ids:
            f.write(f">{id_chain}\n")
            for chain, pos, aa, lab in data[id_chain]:
                f.write(f"{chain} {pos} {aa} {lab}\n")

def write_diff_info(ids, before, after, output_file):
    with open(output_file, "w") as f:
        for id_chain in ids:
            # Находим позиции (resnum), где до пропагации был 1
            before_list = before[id_chain]
            after_list = after[id_chain]
            before_ones = [pos for (chain, pos, aa, lab) in before_list if lab == 1]
            # Где раньше был 0, а после стал 1
            propagated_ones = [pos for (i, (chain, pos, aa, lab)) in enumerate(after_list)
                               if lab == 1 and before_list[i][3] == 0]
            f.write(f"{id_chain}:\n")
            f.write(f"- before: {before_ones}\n")
            f.write(f"- propagated: {propagated_ones}\n\n")

def write_diff_info_with_aa(ids, before, after, output_file):
    with open(output_file, "w") as f:
        for id_chain in ids:
            before_list = before[id_chain]
            after_list  = after[id_chain]
            
            before_entries = [
                f"{chain} {pos} {aa}"
                for chain, pos, aa, lab in before_list
                if lab == 1
            ]
            
            propagated_entries = [
                f"{chain} {pos} {aa}"
                for (chain, pos, aa, lab), (_, _, _, lab0) in zip(after_list, before_list)
                if lab == 1 and lab0 == 0
            ]

            f.write(f"{id_chain}:\n")
            f.write(f"- before: {before_entries}\n")
            f.write(f"- propagated: {propagated_entries}\n\n")


def calculate_catalytic_cnt(ann_dict):
    catalytic_cnt = 0
    for id, arr in ann_dict.items():
        catalytic_cnt += sum([label for chain, pos, aa, label in arr])
    return catalytic_cnt

ANNOT_DIR = "/home/iscb/wolfson/annab4/DB/all_proteins/cross_validation_chem/weight_based_v2"

#for al_type in ["3di_gaps", "TMAlign", "3di"]:
    #print("Alignment type:", al_type)
for ratio in [0.2, 0.3]:#, 0.5, 0.6, 0.7]:
    print("Propagation ration:", ratio)
    PROPAGATED_DIR = f"/home/iscb/wolfson/annab4/DB/all_proteins/MSA_propagation/propagated_v8_3di_{ratio}"

    propagated_filepath = os.path.join(PROPAGATED_DIR, "propagated_all.txt")

    all_ann = {}
    for split in os.listdir(ANNOT_DIR):
        if not split.endswith(".txt"):
            continue
        all_ann.update(parse_annotation_file(os.path.join(ANNOT_DIR, split)))

    catalytic_cnt = calculate_catalytic_cnt(all_ann)
    print(f"Number of positive lables before propagation: {catalytic_cnt}")

    propagated_ann = parse_annotation_file(propagated_filepath)
    catalytic_cnt_propagated = calculate_catalytic_cnt(propagated_ann)
    print(f"Number of positive lables propagation: {catalytic_cnt_propagated}")

    print()

    improved_ids = []
    for id_chain in all_ann:
        if id_chain in propagated_ann:
            cnt_before = catalytic_label_sum(all_ann[id_chain])
            cnt_after  = catalytic_label_sum(propagated_ann[id_chain])
            if cnt_after > cnt_before:
                improved_ids.append(id_chain)

    print(f"Number of improved chains: {len(improved_ids)}")

    write_diff_info_with_aa(improved_ids, all_ann, propagated_ann, os.path.join(PROPAGATED_DIR, "diff_with_aa.txt"))
    print("Saved diff_with_aa.txt")

    write_diff_info(improved_ids, all_ann, propagated_ann, os.path.join(PROPAGATED_DIR, "diff.txt"))
    print("Saved diff.txt")

    save_annotations_ordered(improved_ids, all_ann, os.path.join(PROPAGATED_DIR, "before.txt"))
    save_annotations_ordered(improved_ids, propagated_ann, os.path.join(PROPAGATED_DIR, "after.txt"))
    print("Done! Saved before.txt and after.txt")
    print()
