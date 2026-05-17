def compute_index_mapping(gapped_a, gapped_b) -> dict:
    """Map ungapped indices from sequence A to sequence B via their gapped alignment strings."""
    mapping = {}
    i, j = 0, 0
    for a, b in zip(gapped_a, gapped_b):
        if a != "-" and b != "-":
            mapping[i] = j
            i += 1
            j += 1
        elif a != "-":
            i += 1
        elif b != "-":
            j += 1
    return mapping
