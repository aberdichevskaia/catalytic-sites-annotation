alpha = list("ACDEFGHIKLMNPQRSTVWY")
with open("match1_minus1.mat", "w") as f:
    # Первая строка — заголовок с алфавитом
    f.write("  " + " ".join(alpha) + "\n")
    # Каждая строка — остаток + 20 чисел
    for aa in alpha:
        scores = [1 if aa==bb else -1 for bb in alpha]
        f.write(aa + " " + " ".join(f"{s:2d}" for s in scores) + "\n")
