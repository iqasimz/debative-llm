#!/usr/bin/env python3
"""
scripts/stage4/pseudo_label.py

Use the teacher model to generate soft labels for each sentence,
and extract a high-confidence subset.

Inputs:
  --model_dir   Directory of the saved teacher (tokenizer + model)
  --input_txt   Plain text file, one sentence per line
Outputs:
  --soft_csv    CSV with columns: sentence,claim_prob,premise_prob
  --highconf_csv CSV subset with max(conf) >= threshold
"""

import argparse
import pandas as pd
import torch
from transformers import BertTokenizerFast, BertForSequenceClassification
from tqdm import tqdm

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model_dir',    required=True)
    p.add_argument('--input_txt',    required=True)
    p.add_argument('--soft_csv',     required=True)
    p.add_argument('--highconf_csv', required=True)
    p.add_argument('--threshold',    type=float, default=0.9)
    p.add_argument('--batch',        type=int,   default=32)
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load sentences
    with open(args.input_txt, encoding='utf8') as f:
        sents = [l.strip() for l in f if l.strip()]
    print(f"Loaded {len(sents)} sentences from {args.input_txt}")

    # Load model + tokenizer
    tokenizer = BertTokenizerFast.from_pretrained(args.model_dir)
    model     = BertForSequenceClassification.from_pretrained(args.model_dir).to(device)
    model.eval()

    rows = []
    highconf = []

    # Batch inference
    for i in tqdm(range(0, len(sents), args.batch), desc="Pseudo-labeling"):
        batch_sents = sents[i:i+args.batch]
        enc = tokenizer(batch_sents, padding=True, truncation=True, return_tensors='pt').to(device)
        with torch.no_grad():
            logits = model(**enc).logits
            probs  = torch.softmax(logits, dim=1).cpu()

        for sent, prob in zip(batch_sents, probs):
            claim_p, premise_p = prob.tolist()
            rows.append((sent, claim_p, premise_p))
            if max(claim_p, premise_p) >= args.threshold:
                label = "Claim" if claim_p > premise_p else "Premise"
                highconf.append((sent, label, max(claim_p, premise_p)))

    # Save soft labels
    df_soft = pd.DataFrame(rows, columns=['sentence','claim_prob','premise_prob'])
    df_soft.to_csv(args.soft_csv, index=False)
    print(f"Saved soft labels to {args.soft_csv}")

    # Save high-confidence subset
    df_hc = pd.DataFrame(highconf, columns=['sentence','pred_label','confidence'])
    df_hc.to_csv(args.highconf_csv, index=False)
    print(f"Saved {len(df_hc)} high-confidence labels to {args.highconf_csv}")

if __name__ == '__main__':
    main()