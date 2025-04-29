#!/usr/bin/env python3
"""
scripts/stage4/merge_active.py

Merge newly annotated active‐learning labels into the existing gold‐augmented dataset.

Inputs:
  --gold_csv    Path to your existing gold_augmented.csv (or v2)  
  --new_csv     Path to your newly annotated CSV (to_annotate_1k.csv)  
Outputs:
  --merged_csv  Path where the combined CSV will be written
"""

import argparse
import pandas as pd

def main():
    parser = argparse.ArgumentParser(
        description="Merge active‐learning labels into the gold dataset"
    )
    parser.add_argument(
        '--gold_csv', required=True,
        help='Path to existing gold_augmented.csv'
    )
    parser.add_argument(
        '--new_csv', required=True,
        help='Path to newly annotated CSV (to_annotate_1k.csv)'
    )
    parser.add_argument(
        '--merged_csv', required=True,
        help='Path to write the merged output CSV'
    )
    args = parser.parse_args()

    # Load files
    gold = pd.read_csv(args.gold_csv)
    new  = pd.read_csv(args.new_csv)

    # Concatenate and drop duplicates (so any overlaps are handled)
    combined = pd.concat([gold, new], ignore_index=True).drop_duplicates()

    # Save
    combined.to_csv(args.merged_csv, index=False)
    print(f"Merged {len(gold)} gold + {len(new)} new → {len(combined)} total examples")
    
if __name__ == "__main__":
    main()