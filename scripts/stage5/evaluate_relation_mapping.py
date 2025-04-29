#!/usr/bin/env python3
"""
Evaluate Support/Attack/None classification on the dev set.
Compares data/relation_dev_gold.csv (gold) vs. data/relation_dev_pred.csv (predicted).
"""

import pandas as pd
from sklearn.metrics import classification_report

def main():
    # Load gold & pred
    gold = pd.read_csv("data/relation_dev_gold.csv")
    pred = pd.read_csv("data/relation_dev_pred.csv")

    # Ensure no NaNs and consistent string types
    gold_rel = gold["relation"].fillna("None").astype(str)
    pred_rel = pred["pred_label_str"].fillna("None").astype(str)

    print("=== Relation Mapping (Before Negation) ===\n")
    print(classification_report(
        gold_rel,
        pred_rel,
        labels=["Support","Attack","None"],
        zero_division=0
    ))

if __name__ == "__main__":
    main()