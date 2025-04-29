#!/usr/bin/env python3
"""
Select the top-K most uncertain sentences according to the student model.
Outputs data/uncertain_5k.txt (one sentence per line).
"""

import argparse, torch
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification
)
from tqdm import tqdm

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model_dir',  required=True, help='trained student model dir')
    p.add_argument('--input_txt',  required=True, help='all sentences file (one per line)')
    p.add_argument('--output_txt', required=True, help='where to save uncertain top-K')
    p.add_argument('--k',          type=int, default=5000)
    p.add_argument('--batch',      type=int, default=32)
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else
                          'mps'  if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")

    tok   = DistilBertTokenizerFast.from_pretrained(args.model_dir)
    model = DistilBertForSequenceClassification.from_pretrained(args.model_dir).to(device)
    model.eval()

    # Load sentences
    with open(args.input_txt, encoding='utf8') as f:
        sents = [l.strip() for l in f if l.strip()]
    print(f"Loaded {len(sents)} sentences")

    margins = []
    # Batch inference
    for i in tqdm(range(0, len(sents), args.batch), desc="Uncertainty"):
        batch = sents[i:i+args.batch]
        enc   = tok(batch, padding=True, truncation=True,
                    return_tensors='pt').to(device)
        with torch.no_grad():
            logits = model(**enc).logits
            probs  = torch.softmax(logits, dim=1).cpu()
        for sent, p in zip(batch, probs):
            margins.append((abs(p[0]-p[1]).item(), sent))

    # pick lowest-K margins
    margins.sort(key=lambda x: x[0])
    with open(args.output_txt, 'w', encoding='utf8') as out:
        for _, sent in margins[:args.k]:
            out.write(sent.replace('\n',' ') + '\n')
    print(f"Wrote top {args.k} uncertain sentences to {args.output_txt}")

if __name__=="__main__":
    main()