#!/usr/bin/env python3
"""
Predict Support/Attack/None for every sentence‐pair within a window.
Writes out a CSV of (i, j, pred_label, confidence).
"""
import argparse
import csv
import torch
from transformers import BertTokenizerFast, BertForSequenceClassification
from tqdm import tqdm

def load_sentences(path):
    with open(path, 'r') as f:
        return [line.strip() for line in f if line.strip()]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', required=True, help="Relation model directory")
    parser.add_argument('--input_txt', required=True, help="One sentence per line")
    parser.add_argument('--output_csv', required=True)
    parser.add_argument('--window', type=int, default=2,
                        help="Max distance between sentences to consider")
    parser.add_argument('--batch', type=int, default=32)
    args = parser.parse_args()

    device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = BertTokenizerFast.from_pretrained(args.model_dir)
    model     = BertForSequenceClassification.from_pretrained(args.model_dir).to(device)
    model.eval()

    # 1) Load all sentences
    print("Loading sentences…")
    sentences = load_sentences(args.input_txt)
    n = len(sentences)
    print(f"→ {n} sentences loaded")

    # 2) Build all (i,j) pairs within the window
    print(f"Building index pairs with window={args.window}…")
    pairs = []
    for i in range(n):
        start = max(0, i - args.window)
        end   = min(n, i + args.window + 1)
        for j in range(start, end):
            if i == j: 
                continue
            pairs.append((i, j))
    print(f"→ {len(pairs)} total pairs to predict")

    # 3) Predict in batches
    print(f"Running inference in batches of {args.batch}…")
    with open(args.output_csv, 'w', newline='') as fout:
        writer = csv.writer(fout)
        writer.writerow(['i','j','pred_label','confidence'])

        for idx in tqdm(range(0, len(pairs), args.batch), desc="Predicting"):
            batch_pairs = pairs[idx:idx + args.batch]
            batch_s1 = [sentences[i] for (i,_) in batch_pairs]
            batch_s2 = [sentences[j] for (_,j) in batch_pairs]

            enc = tokenizer(batch_s1, batch_s2,
                            truncation=True, padding=True,
                            return_tensors='pt').to(device)

            with torch.no_grad():
                logits = model(**enc).logits
                probs  = torch.softmax(logits, dim=1)
                preds  = torch.argmax(probs, dim=1).tolist()
                confs  = probs.max(dim=1).values.tolist()

            for ((i,j), p, c) in zip(batch_pairs, preds, confs):
                writer.writerow([i, j, p, c])

    print(f"Done → predictions written to {args.output_csv}")

if __name__ == '__main__':
    main()