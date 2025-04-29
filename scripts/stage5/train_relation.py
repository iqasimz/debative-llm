#!/usr/bin/env python3
"""
Fine-tune BERT-base to predict Support / Attack / None between two sentences.
"""
import argparse
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
from transformers import BertTokenizerFast, BertForSequenceClassification, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tqdm import tqdm

def load_data(path, tokenizer, max_len=128):
    df = pd.read_csv(path)
    # Identify columns
    si = next(c for c in df.columns if 'sentence_i' in c.lower())
    sj = next(c for c in df.columns if 'sentence_j' in c.lower())
    rl = next(c for c in df.columns if 'relation' in c.lower())

    # 1) Clean & map the relation column
    df[rl] = df[rl].fillna('None').str.strip()
    mapping = {'Support': 0, 'Attack': 1, 'None': 2}
    labels = df[rl].map(mapping)

    # 2) Drop any rows with unmapped relations
    valid = labels.notna()
    if not valid.all():
        df = df[valid].reset_index(drop=True)
        labels = labels[valid]

    labels = labels.astype(int).tolist()

    # 3) Encode sentence pairs
    texts = [f"{a} [SEP] {b}" for a,b in zip(df[si], df[sj])]
    enc   = tokenizer(
        texts,
        padding='max_length',
        truncation=True,
        max_length=max_len,
        return_tensors='pt'
    )

    return TensorDataset(
        enc['input_ids'],
        enc['attention_mask'],
        torch.tensor(labels, dtype=torch.long)
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gold_csv', required=True)
    parser.add_argument('--model',    default='bert-base-uncased')
    parser.add_argument('--batch',    type=int, default=16)
    parser.add_argument('--epochs',   type=int, default=3)
    parser.add_argument('--lr',       type=float, default=3e-5)
    parser.add_argument('--output',   required=True)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = BertTokenizerFast.from_pretrained(args.model)
    dataset   = load_data(args.gold_csv, tokenizer)
    labels    = [dataset[i][2].item() for i in range(len(dataset))]

    # Train/dev split
    train_idx, dev_idx = train_test_split(
        list(range(len(dataset))),
        test_size=0.1,
        stratify=labels,
        random_state=42
    )
    train_ds = torch.utils.data.Subset(dataset, train_idx)
    dev_ds   = torch.utils.data.Subset(dataset, dev_idx)

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True)
    dev_loader   = DataLoader(dev_ds,  batch_size=args.batch)

    # Model, optimizer, scheduler
    model = BertForSequenceClassification.from_pretrained(
        args.model, num_labels=3
    ).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr)

    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )

    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        for batch in tqdm(train_loader, desc=f"Train Epoch {epoch}"):
            input_ids, attention_mask, labels = [t.to(device) for t in batch]
            loss = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            ).loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        # Evaluation
        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for batch in dev_loader:
                input_ids, attention_mask, labels = [t.to(device) for t in batch]
                logits = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                ).logits
                preds.extend(torch.argmax(logits, dim=1).cpu().tolist())
                trues.extend(labels.cpu().tolist())

        print(classification_report(
            trues, preds,
            target_names=['Support','Attack','None'],
            zero_division=0
        ))
        acc = sum(p == t for p,t in zip(preds,trues)) / len(trues)
        if acc > best_acc:
            best_acc = acc
            model.save_pretrained(args.output)
            tokenizer.save_pretrained(args.output)
            print(f"→ Saved best relation model (acc={acc:.4f}) to {args.output}")

if __name__ == '__main__':
    main()