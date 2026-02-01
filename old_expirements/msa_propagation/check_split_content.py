def parse_annotation_file(path):
    """
    Read per‐chain annotation file into:
      { "UniProtID_chain": { residue_number: (AA, label), ... }, ... }
    """
    ann = {}
    with open(path) as f:
        current = None
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                current = line[1:]
                ann[current] = {}
            else:
                chain, resnum, aa, lab = line.split()
                ann[current][int(resnum)] = (aa, int(lab))
    return ann


before_prop = "/home/iscb/wolfson/annab4/DB/all_proteins/cross_validation_chem/cleaned/split3.txt"
after_prop = "/home/iscb/wolfson/annab4/DB/all_proteins/MSA_propagation/propagated_3di_0.2/split3.txt"

ann_before_prop = parse_annotation_file(before_prop)
ann_after_prop = parse_annotation_file(after_prop)

print(ann_after_prop.keys() == ann_before_prop.keys())

print()

print(set(ann_before_prop.keys()) - set(ann_after_prop.keys() ))