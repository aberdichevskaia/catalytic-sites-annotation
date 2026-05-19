import pickle
from pprint import pprint

paths = [
    "/home/iscb/wolfson/annab4/scannet_retrains/ablate01_esm2_3B_CataloDB_fold1_v1/test_results.pkl",
    "/home/iscb/wolfson/annab4/scannet_retrains/ablate01_esm2_3B_CataloDB_fold2_v1/test_results.pkl",
    "/home/iscb/wolfson/annab4/scannet_retrains/ablate01_esm2_3B_CataloDB_fold3_v1/test_results.pkl",
    "/home/iscb/wolfson/annab4/scannet_retrains/ablate01_esm2_3B_CataloDB_fold4_v1/test_results.pkl",
    "/home/iscb/wolfson/annab4/scannet_retrains/ablate01_esm2_3B_CataloDB_fold5_v1/test_results.pkl",
]

sets = []
for p in paths:
    with open(p, "rb") as f:
        obj = pickle.load(f)
    ids = list(map(str, obj["ids"]))
    print("\nFILE:", p)
    print("subset:", obj.get("subset"))
    print("n_ids:", len(ids))
    print("first 10 ids:")
    pprint(ids[:10])
    sets.append(set(ids))

common = set.intersection(*sets)
print("\nCOMMON ACROSS ALL 5:", len(common))

for i in range(len(sets)):
    for j in range(i + 1, len(sets)):
        print(f"intersection {i+1}-{j+1}: {len(sets[i] & sets[j])}")