# scripts/stage5/step7_sample_active.py

import argparse
import csv
from heapq import nsmallest

# Paths
SCORES_CSV = "data/relation_scores_all.csv"
CLAIMS_TXT = "data/claims_highconf.txt"
PREMS_TXT  = "data/premises_highconf.txt"
OUTPUT_CSV = "data/to_annotate_relation.csv"

def main(n):
    # Load all claims & premises into lists
    claims   = [l.strip() for l in open(CLAIMS_TXT, encoding="utf-8")]
    premises = [l.strip() for l in open(PREMS_TXT, encoding="utf-8")]

    # Build heap of (confidence, row)
    heap = []
    with open(SCORES_CSV, encoding="utf-8") as fin:
        reader = csv.DictReader(fin)
        for r in reader:
            c_i = int(r["claim_idx"])
            p_i = int(r["premise_idx"])
            support = float(r["support_prob"])
            attack  = float(r["attack_prob"])
            none    = float(r["none_prob"])
            # consider only support/attack
            if none >= max(support, attack):
                continue
            conf = max(support, attack)
            heap.append((conf, c_i, p_i, support, attack, none))

    # Pick the n lowest-confidence
    lowest = nsmallest(n, heap, key=lambda x: x[0])

    # Write out with sentences
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as fout:
        writer = csv.writer(fout)
        writer.writerow([
            "claim_idx", "premise_idx",
            "claim_sentence", "premise_sentence",
            "support_prob", "attack_prob", "none_prob", "confidence"
        ])
        for conf, c_i, p_i, sup, atk, non in lowest:
            writer.writerow([
                c_i,
                p_i,
                claims[c_i],
                premises[p_i],
                f"{sup:.4f}",
                f"{atk:.4f}",
                f"{non:.4f}",
                f"{conf:.4f}",
            ])

    print(f"Sampled {len(lowest)} pairs into {OUTPUT_CSV}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=1000,
                   help="Number of low-confidence pairs to sample")
    args = p.parse_args()
    main(args.n)