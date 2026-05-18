import pickle


def acc_only(name: str) -> str:
    """Extract the base accession from a display name by splitting on the first '_'.

    Examples:
        'P81877_F1'    -> 'P81877'
        'A0A0K2S4Q6-1' -> 'A0A0K2S4Q6-1'  (no underscore -> unchanged)
    """
    return name.split("_", 1)[0].upper()


def parse_batch_file(path: str) -> dict:
    """Parse an annotation pickle file into {chain_id: list_of_annotation_lines}.

    Handles both bytes and str lines inside the pickle. Chain IDs are returned
    without the leading '>' and with surrounding whitespace stripped.
    """
    with open(path, 'rb') as f:
        data = pickle.load(f)
    out = {}
    curr = None
    annots = []
    for line in data:
        is_header = (
            (isinstance(line, bytes) and line.startswith(b'>')) or
            (isinstance(line, str) and line.startswith('>'))
        )
        if is_header:
            if curr is not None:
                out[curr] = annots.copy()
            curr = line.lstrip(b'>').decode() if isinstance(line, bytes) else line[1:].strip()
            annots = []
        else:
            annots.append(line)
    if curr is not None:
        out[curr] = annots.copy()
    return out
