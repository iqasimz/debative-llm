#!/usr/bin/env python3
"""
Evaluate the DistilBERT student on your held-out gold dev set.
Prints precision, recall, F1 for Claim vs. Premise and overall accuracy.
"""
import argparse
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from sklearn.metrics import classification_report, accuracy_score
from tqdm import tqdm

def load_gold(path, tokenizer, max_len=128):
    df     = pd.read_csv(path)
    sent_c = next(c for c in df.columns if 'sentence' in c.lower())
    lbl_c  = next(c for c in df.columns if 'label' in c.lower())
    texts  = df[sent_c].tolist()
    labels = [0 if str(l).strip().lower()=='claim' else 1
              for l in df[lbl_c]]
    enc = tokenizer(texts, padding='max_length', truncation=True,
                    max_length=max_len, return_tensors='pt')
    return TensorDataset(enc['input_ids'], enc['attention_mask'], torch.tensor(labels))

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--gold_csv', required=True)
    p.add_argument('--model_dir', required=True)
    p.add_argument('--batch',    type=int, default=32)
    args = p.parse_args()

    device    = torch.device('cuda' if torch.cuda.is_available() else
                             'mps'  if torch.backends.mps.is_available() else
                             'cpu')
    tokenizer = DistilBertTokenizerFast.from_pretrained(args.model_dir)
    model     = DistilBertForSequenceClassification.from_pretrained(
                    args.model_dir, num_labels=2
                ).to(device)
    model.eval()

    dev_ds  = load_gold(args.gold_csv, tokenizer)
    loader  = DataLoader(dev_ds, batch_size=args.batch)

    preds, trues = [], []
    with torch.no_grad():
        for ids, mask, lbl in tqdm(loader, desc="Evaluating"):
            ids, mask = ids.to(device), mask.to(device)
            logits   = model(ids, attention_mask=mask).logits
            batch_p  = torch.argmax(logits, dim=1).cpu().tolist()
            preds.extend(batch_p)
            trues.extend(lbl.tolist())

    report = classification_report(trues, preds,
                target_names=['Claim','Premise'], zero_division=0)
    acc    = accuracy_score(trues, preds)
    print(report)
    print(f"Overall accuracy: {acc:.4f}")

if __name__ == '__main__':
    main()