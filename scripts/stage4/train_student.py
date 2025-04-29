#!/usr/bin/env python3
"""
scripts/stage4/train_student.py

Train a student model (DistilBERT) via knowledge distillation from a teacher.
Uses:
  - Hard labels from gold_augmented_v2_clean.csv
  - Soft targets from pseudo_highconf_v2.csv
Supports:
  --max_soft      Subsample the soft dataset for speed
  --num_workers   Parallel DataLoader workers
"""

import argparse
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset, Subset
from torch.optim import AdamW
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    get_linear_schedule_with_warmup
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from tqdm import tqdm

def load_hard(path, tokenizer, max_len=128):
    df    = pd.read_csv(path)
    texts = df['sentence'].tolist()
    # CASE‐INSENSITIVE mapping for Claim/Premise
    labels = [0 if str(l).strip().lower()=='claim' else 1 for l in df['label']]
    enc   = tokenizer(
        texts,
        padding='max_length',
        truncation=True,
        max_length=max_len,
        return_tensors='pt'
    )
    return TensorDataset(enc['input_ids'], enc['attention_mask'], torch.tensor(labels, dtype=torch.long))

def load_soft(path, tokenizer, max_len=128, max_soft=None, seed=42):
    """
    Loads soft targets. Handles two cases:
      1) CSV has explicit probabilities for claim/premise columns.
      2) CSV has `pred_label` and `confidence` columns for high-conf only.
    """
    df = pd.read_csv(path)
    # 1) subsample if requested
    if max_soft and len(df) > max_soft:
        df = df.sample(n=max_soft, random_state=seed).reset_index(drop=True)

    # Identify sentence column
    sent_col = next(c for c in df.columns if 'sentence' in c.lower())
    texts    = df[sent_col].tolist()

    # Case A: explicit claim/premise probability columns
    claim_cols   = [c for c in df.columns if 'claim' in c.lower() and 'pred' not in c.lower()]
    premise_cols = [c for c in df.columns if 'premise' in c.lower()]
    if claim_cols and premise_cols:
        claim_col   = claim_cols[0]
        premise_col = premise_cols[0]
        soft_vals   = df[[claim_col, premise_col]].astype(float).values

    # Case B: high-confidence predictions
    elif 'pred_label' in df.columns and 'confidence' in df.columns:
        lbls   = df['pred_label'].astype(str).str.strip().str.lower().tolist()
        confs  = df['confidence'].astype(float).tolist()
        soft_vals = []
        for l, c in zip(lbls, confs):
            if l == 'claim':
                soft_vals.append([c, 1.0 - c])
            else:
                soft_vals.append([1.0 - c, c])
    else:
        raise ValueError(
            f"Unable to find soft-label columns in {path}. "
            f"Expected prob columns or 'pred_label'+'confidence'."
        )

    soft = torch.tensor(soft_vals, dtype=torch.float)
    enc  = tokenizer(
        texts,
        padding='max_length',
        truncation=True,
        max_length=max_len,
        return_tensors='pt'
    )
    return TensorDataset(enc['input_ids'], enc['attention_mask'], soft)

def compute_metrics(preds, labels):
    f1c = f1_score(labels, preds, pos_label=0, zero_division=0)
    f1p = f1_score(labels, preds, pos_label=1, zero_division=0)
    return (f1c + f1p) / 2

def train_epoch(model, hard_loader, soft_loader, optimizer, scheduler, device, alpha, temp):
    model.train()
    ce_loss = torch.nn.CrossEntropyLoss()
    total_loss = 0.0

    # Hard-label phase
    for ids, mask, lbl in tqdm(hard_loader, desc="Train-hard"):
        ids, mask, lbl = ids.to(device), mask.to(device), lbl.to(device)
        logits = model(ids, attention_mask=mask).logits
        loss   = ce_loss(logits, lbl) * (1-alpha)
        total_loss += loss.item()
        loss.backward()
        optimizer.step(); scheduler.step(); optimizer.zero_grad()

    # Soft-label (distillation) phase
    for ids, mask, soft in tqdm(soft_loader, desc="Train-soft"):
        ids, mask, soft = ids.to(device), mask.to(device), soft.to(device)
        logits = model(ids, attention_mask=mask).logits
        logp   = torch.nn.functional.log_softmax(logits/temp, dim=1)
        loss   = torch.nn.functional.kl_div(logp, soft, reduction='batchmean') \
                 * (temp**2) * alpha
        total_loss += loss.item()
        loss.backward()
        optimizer.step(); scheduler.step(); optimizer.zero_grad()

    return total_loss / (len(hard_loader) + len(soft_loader))

def eval_epoch(model, loader, device):
    model.eval()
    preds, true = [], []
    with torch.no_grad():
        for ids, mask, lbl in tqdm(loader, desc="Eval"):
            ids, mask, lbl = ids.to(device), mask.to(device), lbl.to(device)
            logits = model(ids, attention_mask=mask).logits
            preds.extend(torch.argmax(logits, dim=1).cpu().tolist())
            true.extend(lbl.cpu().tolist())
    return compute_metrics(preds, true)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hard_csv',    required=True, help='Path to gold_augmented_v2_clean.csv')
    parser.add_argument('--soft_csv',    required=True, help='Path to pseudo_highconf_v2.csv')
    parser.add_argument('--model',       default='distilbert-base-uncased')
    parser.add_argument('--batch',       type=int, default=32)
    parser.add_argument('--epochs',      type=int, default=3)
    parser.add_argument('--lr',          type=float, default=5e-5)
    parser.add_argument('--alpha',       type=float, default=0.3, help='Weight for soft loss')
    parser.add_argument('--temperature', type=float, default=2.0)
    parser.add_argument('--max_soft',    type=int, help='Max number of soft examples per epoch')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--output',      required=True, help='Directory to save the student model')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else
                          'mps'  if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    tokenizer = DistilBertTokenizerFast.from_pretrained(args.model)

    hard_ds = load_hard(args.hard_csv, tokenizer)
    soft_ds = load_soft(args.soft_csv, tokenizer, max_soft=args.max_soft)

    # Split dev from hard set
    idxs = list(range(len(hard_ds)))
    lbls = [hard_ds[i][2].item() for i in idxs]
    train_idx, dev_idx = train_test_split(idxs, test_size=0.1, stratify=lbls, random_state=42)
    train_hard = Subset(hard_ds, train_idx)
    dev_ds     = Subset(hard_ds, dev_idx)

    loader_args = dict(
        batch_size=args.batch,
        num_workers=args.num_workers,
        pin_memory=(device.type=='cuda')
    )
    hard_loader = DataLoader(train_hard, shuffle=True, **loader_args)
    soft_loader = DataLoader(soft_ds,   shuffle=True, **loader_args)
    dev_loader  = DataLoader(dev_ds,    shuffle=False, **loader_args)

    print(f"Sizes → hard:{len(train_hard)}, soft:{len(soft_ds)}, dev:{len(dev_ds)}")

    model     = DistilBertForSequenceClassification.from_pretrained(args.model, num_labels=2).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr)
    total_steps = (len(hard_loader) + len(soft_loader)) * args.epochs
    scheduler   = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1*total_steps),
        num_training_steps=total_steps
    )

    best_f1 = 0.0
    for epoch in range(1, args.epochs+1):
        print(f"\n=== Epoch {epoch} ===")
        train_loss = train_epoch(model, hard_loader, soft_loader,
                                 optimizer, scheduler, device,
                                 args.alpha, args.temperature)
        print(f"Train loss: {train_loss:.4f}")
        dev_f1 = eval_epoch(model, dev_loader, device)
        print(f"Dev balanced F1: {dev_f1:.4f}")
        if dev_f1 > best_f1:
            best_f1 = dev_f1
            model.save_pretrained(args.output)
            tokenizer.save_pretrained(args.output)
            print(f"→ Saved best student model (F1={dev_f1:.4f}) to {args.output}")

if __name__ == '__main__':
    main()