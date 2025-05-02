# scripts/stage5/merge_manual_labels.py

import csv

SEED_CSV   = "data/seed_relation_set_fast.csv"
MANUAL_CSV = "data/manual_relation_labels.csv"
OUT_CSV    = "data/seed_relation_merged.csv"

# load seed into dict keyed by (c,p)
seed = {}
with open(SEED_CSV) as f:
    reader = csv.DictReader(f)
    for r in reader:
        seed[(r["claim_idx"],r["premise_idx"])] = r["label"]

# overwrite with manual
with open(MANUAL_CSV) as f:
    reader = csv.DictReader(f)
    for r in reader:
        seed[(r["claim_idx"],r["premise_idx"])] = r["label"]

# write merged
with open(OUT_CSV, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["claim_idx","premise_idx","label"])
    for (c,p), label in seed.items():
        w.writerow([c,p,label])
print(f"Merged seed+manual → {OUT_CSV}")