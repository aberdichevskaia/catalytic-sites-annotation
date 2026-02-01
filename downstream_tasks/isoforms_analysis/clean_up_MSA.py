import re, sys

inp = "/home/iscb/wolfson/annab4/MSA_O00255-3_0_A.fasta"
out = "/home/iscb/wolfson/annab4/MSA_O00255-3_0_A_fixed.fasta"

def read_fasta(path):
    name, seq = None, []
    with open(path) as f:
        for line in f:
            line=line.strip()
            if not line: 
                continue
            if line.startswith(">"):
                if name is not None:
                    yield name, "".join(seq)
                name, seq = line, []
            else:
                seq.append(line)
        if name is not None:
            yield name, "".join(seq)

lens = set()
recs = []
for h, s in read_fasta(inp):
    s = s.replace('.', '-')          # на всякий случай
    s = re.sub(r'[a-z]', '', s)      # ключевое: убрать A3M insertions
    recs.append((h, s))
    lens.add(len(s))

print("Unique lengths after fix:", sorted(lens)[:10], " ... total", len(lens))
# если всё равно разные — можно фильтровать по длине первой (query)
L = len(recs[0][1])
recs2 = [(h,s) for h,s in recs if len(s)==L]
print(f"Keeping {len(recs2)}/{len(recs)} sequences with length == {L}")

with open(out, "w") as f:
    for h, s in recs2:
        f.write(h+"\n")
        for i in range(0, len(s), 80):
            f.write(s[i:i+80]+"\n")
print("Wrote", out)