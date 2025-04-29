#!/usr/bin/env python3
"""
Train a teacher model (BERT-base) on augmented claim/premise data.
Saves the best checkpoint by balanced F1 on the dev set.
Includes class-weighted loss to address imbalance.
"""

import argparse
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset, Subset
from torch.optim import AdamW
from transformers import (
    BertTokenizerFast,
    BertForSequenceClassification,
    get_linear_schedule_with_warmup
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from tqdm import tqdm
from collections import Counter

def load_data(path, tokenizer, max_len=128):
    df = pd.read_csv(path)
    # dynamically find sentence column
    sent_col = next(c for c in df.columns if 'sentence' in c.lower())
    texts = df[sent_col].tolist()
    # dynamically find label column
    label_col = next(c for c in df.columns if 'label' in c.lower())
    # CASE-INSENSITIVE label mapping: claim=0, premise=1
    labels = [0 if str(l).strip().lower()=='claim' else 1 for l in df[label_col]]
    enc = tokenizer(
        texts,
        truncation=True,
        padding='max_length',
        max_length=max_len,
        return_tensors='pt'
    )
    return enc, torch.tensor(labels, dtype=torch.long)

def compute_metrics(preds, labels):
    f1_claim   = f1_score(labels, preds, pos_label=0, zero_division=0)
    f1_premise = f1_score(labels, preds, pos_label=1, zero_division=0)
    return (f1_claim + f1_premise) / 2

def train_epoch(model, loader, optimizer, scheduler, loss_fn, device):
    model.train()
    total_loss = 0.0
    for batch in tqdm(loader, desc="Train"):
        input_ids, attention_mask, labels = [b.to(device) for b in batch]
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        loss   = loss_fn(logits, labels)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
    return total_loss / len(loader)

def eval_epoch(model, loader, device):
    model.eval()
    preds, true = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Eval"):
            input_ids, attention_mask, labels = [b.to(device) for b in batch]
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            preds.extend(torch.argmax(logits, dim=1).cpu().tolist())
            true.extend(labels.cpu().tolist())
    return compute_metrics(preds, true)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input',   required=True, help='Path to gold_augmented_v2_clean.csv')
    parser.add_argument('--model',   default='bert-base-uncased')
    parser.add_argument('--batch',   type=int, default=16)
    parser.add_argument('--epochs',  type=int, default=4)
    parser.add_argument('--lr',      type=float, default=3e-5)
    parser.add_argument('--output',  required=True, help='Directory to save model')
    args = parser.parse_args()

    device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = BertTokenizerFast.from_pretrained(args.model)

    enc, labels = load_data(args.input, tokenizer)
    dataset     = TensorDataset(enc['input_ids'], enc['attention_mask'], labels)
    
    counts = Counter(labels.tolist())
    print(f"Label counts (0=claim,1=premise): {counts}")

    train_idx, dev_idx = train_test_split(
        list(range(len(dataset))),
        test_size=0.1,
        stratify=labels,
        random_state=42
    )
    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=args.batch, shuffle=True)
    dev_loader   = DataLoader(Subset(dataset, dev_idx),   batch_size=args.batch, shuffle=False)

    total = sum(counts.values())
    w0    = total / counts[0]
    w1    = total / counts[1]
    loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor([w0, w1], device=device))

    model     = BertForSequenceClassification.from_pretrained(args.model, num_labels=2).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr)
    total_steps = len(train_loader) * args.epochs
    scheduler   = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )

    best_f1 = 0.0
    for epoch in range(1, args.epochs+1):
        print(f"=== Epoch {epoch} ===")
        tr_loss = train_epoch(model, train_loader, optimizer, scheduler, loss_fn, device)
        print(f"Train loss: {tr_loss:.4f}")
        dv_f1   = eval_epoch(model, dev_loader, device)
        print(f"Dev balanced F1: {dv_f1:.4f}")
        if dv_f1 > best_f1:
            best_f1 = dv_f1
            model.save_pretrained(args.output)
            tokenizer.save_pretrained(args.output)
            print(f"→ Saved best model (F1={dv_f1:.4f}) to {args.output}")

if __name__ == '__main__':
    main()