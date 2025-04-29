#!/usr/bin/env python3
"""
Augment Support/Attack/None pairs via EDA, back-translation & MLM paraphrasing.
Outputs a larger CSV for training.
"""
import os
import sys
# Ensure the current directory (scripts/stage5) is on the module path
sys.path.append(os.path.dirname(__file__))

import argparse
import random
import pandas as pd
from tqdm import tqdm
from augment_utils import eda_augment, backtranslate_pair, mlm_paraphrase_pair

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gold_csv',      required=True,
                        help="Input balanced relation CSV")
    parser.add_argument('--eda',           type=int, default=1,
                        help="Number of EDA variants per pair")
    parser.add_argument('--bt',            type=int, default=1,
                        help="Number of back-translation variants per pair")
    parser.add_argument('--mlm',           type=int, default=1,
                        help="Number of MLM paraphrase variants per pair")
    parser.add_argument('--max_aug_factor',type=int, default=3,
                        help="Maximum augmentations per pair")
    parser.add_argument('--output_csv',    required=True,
                        help="Output augmented CSV")
    args = parser.parse_args()

    df = pd.read_csv(args.gold_csv)
    out_rows = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Augmenting"):
        si, sj, lbl = row['sentence_i'], row['sentence_j'], row['relation']
        # keep the original
        out_rows.append({'sentence_i': si, 'sentence_j': sj, 'relation': lbl})

        # generate augmentations
        augs = []
        for _ in range(args.eda):
            augs.append(eda_augment(si, sj))
        for _ in range(args.bt):
            augs.append(backtranslate_pair(si, sj))
        for _ in range(args.mlm):
            augs.append(mlm_paraphrase_pair(si, sj))

        # sample up to max_aug_factor
        random.shuffle(augs)
        for aug_si, aug_sj in augs[:args.max_aug_factor]:
            out_rows.append({
                'sentence_i': aug_si,
                'sentence_j': aug_sj,
                'relation': lbl
            })

    # build DataFrame and dedupe
    out_df = pd.DataFrame(out_rows)
    out_df = out_df.drop_duplicates(subset=['sentence_i','sentence_j','relation'])
    out_df.to_csv(args.output_csv, index=False)
    print(f"Wrote augmented → {args.output_csv} ({len(out_df)} rows)")

if __name__ == '__main__':
    main()