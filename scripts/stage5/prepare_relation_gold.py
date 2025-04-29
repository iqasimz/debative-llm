#!/usr/bin/env python3
"""
Prepare a unified relation‐gold file from M-Arg and Nixon-vs-Kennedy CSV/TSV files.

Outputs:
  data/relation_gold.csv  with columns: sentence_i, sentence_j, relation
"""

import argparse
import pandas as pd

def clean_rel(r):
    """
    Normalize relation labels to exactly 'Support', 'Attack', or 'None'.
    """
    r = str(r).strip().lower()
    if r in ('support', 'supports', 'pro'):
        return 'Support'
    if r in ('attack', 'attacks', 'con'):
        return 'Attack'
    # unify different “no‐relation” labels
    if r in ('neither', 'no_relation', 'no relation', 'no-relation', 'none'):
        return 'None'
    # fallback
    return 'None'

def load_marg(path):
    """
    Load M-Arg CSV and extract sentence_1, sentence_2, relation_gold (or relation).
    """
    df = pd.read_csv(path, sep=None, engine='python')
    rel_col = 'relation_gold' if 'relation_gold' in df.columns else 'relation'
    df2 = df[['sentence_1', 'sentence_2', rel_col]].rename(
        columns={
            'sentence_1':'sentence_i',
            'sentence_2':'sentence_j',
            rel_col      :'relation'
        }
    )
    df2['relation'] = df2['relation'].map(clean_rel)
    return df2

def load_nixon(path):
    """
    Load Nixon-Kennedy TSV/CSV and extract argument1, argument2, relation.
    """
    sep = '\t' if path.lower().endswith('.tsv') else ','
    df = pd.read_csv(path, sep=sep)
    df2 = df[['argument1', 'argument2', 'relation']].rename(
        columns={
            'argument1':'sentence_i',
            'argument2':'sentence_j',
            'relation' :'relation'
        }
    )
    df2['relation'] = df2['relation'].map(clean_rel)
    return df2

def main():
    parser = argparse.ArgumentParser(
        description="Merge M-Arg and Nixon-Kennedy into relation_gold.csv"
    )
    parser.add_argument(
        '--marg_csv', required=True,
        help='Path to M-Arg CSV file'
    )
    parser.add_argument(
        '--nixon_csv', required=True,
        help='Path to Nixon-vs-Kennedy TSV/CSV file'
    )
    parser.add_argument(
        '--output_csv', default='data/relation_gold.csv',
        help='Path to write unified relation_gold.csv'
    )
    args = parser.parse_args()

    # Load and normalize each dataset
    print(f"Loading M-Arg from {args.marg_csv}...")
    marg_df = load_marg(args.marg_csv)
    print(f" → {len(marg_df)} pairs")

    print(f"Loading Nixon-Kennedy from {args.nixon_csv} (sep auto-detect)...")
    nixon_df = load_nixon(args.nixon_csv)
    print(f" → {len(nixon_df)} pairs")

    # Combine and dedupe
    merged = pd.concat([marg_df, nixon_df], ignore_index=True)
    before = len(merged)
    merged = merged.drop_duplicates(subset=['sentence_i', 'sentence_j', 'relation'])
    after = len(merged)
    print(f"Dropped {before-after} duplicate rows → {after} unique pairs")

    # Save
    merged.to_csv(args.output_csv, index=False)
    print(f"Wrote unified relation_gold to {args.output_csv}")

if __name__ == '__main__':
    main()