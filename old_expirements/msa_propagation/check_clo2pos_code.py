import json

# ----------------------------------------
# A3M parsing and col2pos helpers
# ----------------------------------------
def parse_a3m_header(header: str) -> int:
    parts = header[1:].split()
    if len(parts) < 8:
        # probably a centroid sequence
        return 0
    try:
        return int(parts[7])
    except ValueError:
        return 0
    
def a3m_to_aligned_and_col2pos(a3mseq, original_seq, start_index=0):
    aligned_seq, col2pos = [],{}
    seq_len = len(original_seq)
    col_counter = -1
    seq_counter = -1
    for i,aa in enumerate(a3mseq):
        if aa != '-':
            seq_counter +=1
        if aa.isupper() or aa == '-':
            col_counter += 1
            aligned_seq.append(aa)
            #if aa != '-' and 0 <= seq_counter + start_index < seq_len:
            if aa != '-':
                col2pos[col_counter] = seq_counter + start_index      
    return "".join(aligned_seq), col2pos

def compute_aligned_and_mapping(
    msa_seq,
    orig_seq,
    orig_start):
    aligned_chars = []
    col2pos = {}
    orig_i = orig_start
    aligned_i = 0
    L = len(orig_seq)

    for c in msa_seq:
        if c == "-":
            # gap
            aligned_chars.append("-")
            aligned_i += 1
        elif c.isalpha():
            if c.islower():
                orig_i += 1
            else:
                # match
                if 0 <= orig_i < L:
                    aligned_chars.append(c)
                    col2pos[aligned_i] = orig_i
                orig_i += 1
                aligned_i += 1

    return "".join(aligned_chars), col2pos

def read_a3m(fn: str):
    headers, ids, seqs = [], [], []
    with open(fn) as f:
        cur = []
        for l in f:
            l = l.rstrip()
            if not l:
                continue
            if l.startswith(">"):
                headers.append(l)
                ids.append(l[1:].split()[0])
                if cur:
                    seqs.append("".join(cur))
                    cur = []
            else:
                cur.append(l)
        if cur:
            seqs.append("".join(cur))
    return headers, ids, seqs

PROTEIN_TABLE = "/home/iscb/wolfson/annab4/DB/all_protein_table_modified.json"
with open(PROTEIN_TABLE) as f:
    protein_table = json.load(f)
uniprot_seqs = {
    uid: d["uniprot_sequence"]
    for uid, d in protein_table.items()
}

filepath = "/home/iscb/wolfson/annab4/DB/all_proteins/MSA_propagation/msa_dir/28992.a3m"
headers, ids, raw = read_a3m(filepath)
for header, hid, seq in zip(headers, ids, raw):
    full = uniprot_seqs.get(hid)
    if full is None:
        raise KeyError(f"UniProt sequence for {hid} not found")
    start = parse_a3m_header(header)
    #a_seq, mapping = compute_aligned_and_mapping(seq, full, start)
    a_seq, mapping = a3m_to_aligned_and_col2pos(seq, full, start)
    #print(hid)
    print(a_seq)


