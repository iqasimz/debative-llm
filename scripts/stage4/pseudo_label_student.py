#!/usr/bin/env python3
"""
scripts/stage4/pseudo_label_student.py

Use a distilled student model to pseudo-label a full corpus:
- Emits soft probabilities for all sentences.
- Emits high-confidence subset at a given threshold.
"""
import argparse
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir',    required=True, help='Path to student model directory')
    parser.add_argument('--input_txt',    required=True, help='Plain-text file of sentences, one per line')
    parser.add_argument('--soft_csv',     required=True, help='Output CSV for all soft labels')
    parser.add_argument('--highconf_csv',  required=True, help='Output CSV for high-confidence subset')
    parser.add_argument('--threshold',    type=float, default=0.9, help='Confidence threshold')
    parser.add_argument('--batch',        type=int, default=32)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else
                          'mps'  if torch.backends.mps.is_available() else
                          'cpu')
    print(f"Using device: {device}")

    tokenizer = DistilBertTokenizerFast.from_pretrained(args.model_dir)
    model     = DistilBertForSequenceClassification.from_pretrained(
                    args.model_dir, num_labels=2
                ).to(device)
    model.eval()

    # Load sentences
    with open(args.input_txt, 'r') as f:
        sentences = [line.rstrip('\n') for line in f]

    # Tokenize and create DataLoader
    encodings = tokenizer(
        sentences,
        padding='longest',
        truncation=True,
        return_tensors='pt'
    )
    dataset = TensorDataset(encodings['input_ids'], encodings['attention_mask'])
    loader  = DataLoader(dataset, batch_size=args.batch)

    all_probs = []
    with torch.no_grad():
        for input_ids, attention_mask in tqdm(loader, desc='Labeling'):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            probs = torch.softmax(logits, dim=1).cpu().tolist()
            all_probs.extend(probs)

    # Save full soft labels
    df_soft = pd.DataFrame({
        'sentence': sentences,
        'claim_prob': [p[0] for p in all_probs],
        'premise_prob': [p[1] for p in all_probs]
    })
    df_soft.to_csv(args.soft_csv, index=False)
    print(f"Saved all soft labels ({len(df_soft)}) to {args.soft_csv}")

    # Filter high-confidence
    df_hc = df_soft[df_soft[['claim_prob','premise_prob']].max(axis=1) >= args.threshold]
    df_hc.to_csv(args.highconf_csv, index=False)
    print(f"Saved {len(df_hc)} high-confidence labels to {args.highconf_csv}")

if __name__ == '__main__':
    main()