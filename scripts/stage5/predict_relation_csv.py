#!/usr/bin/env python3
"""
Predict Support/Attack/None on an arbitrary CSV of sentence pairs.
Inputs:
  --model_dir:   path to your trained BERT relation model
  --gold_csv:    CSV with columns ['sentence_i','sentence_j']
  --output_csv:  where to write predictions (same length as gold_csv)
  --batch:       batch size for inference
"""

import argparse
import pandas as pd
import torch
from tqdm import tqdm
from transformers import BertTokenizerFast, BertForSequenceClassification

LABEL_MAP = {0: "Support", 1: "Attack", 2: "None"}

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir",  required=True)
    p.add_argument("--gold_csv",   required=True)
    p.add_argument("--output_csv", required=True)
    p.add_argument("--batch",      type=int, default=32)
    args = p.parse_args()

    # Load data
    df = pd.read_csv(args.gold_csv)
    texts = list(zip(df["sentence_i"], df["sentence_j"]))

    # Model + tokenizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = BertTokenizerFast.from_pretrained(args.model_dir)
    model = BertForSequenceClassification.from_pretrained(args.model_dir).to(device)
    model.eval()

    preds = []
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), args.batch), desc="Predicting"):
            batch_pairs = texts[i : i + args.batch]
            # concatenate with [SEP]
            batch_enc = tok(
                [f"{a} [SEP] {b}" for a, b in batch_pairs],
                truncation=True,
                padding=True,
                return_tensors="pt",
            ).to(device)
            logits = model(**batch_enc).logits
            batch_preds = torch.argmax(logits, dim=1).cpu().tolist()
            preds.extend(batch_preds)

    # Attach predictions
    df["pred_label"] = preds
    df["pred_label_str"] = df["pred_label"].map(LABEL_MAP)

    # Save
    df.to_csv(args.output_csv, index=False)
    print(f"→ wrote {len(df)} predictions to {args.output_csv}")

if __name__ == "__main__":
    main()