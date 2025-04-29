#!/usr/bin/env python3
"""
Prepare UKP Argumentative Essays dataset into a CSV for role classification.

Each sentence will have a label:
  - Claim
  - Premise
"""

import os
import pandas as pd
from transformers import AutoTokenizer
from tqdm import tqdm

BRAT_DIR = "data/brat-project-final"
OUTPUT_FILE = "data/ukp_prepared.csv"

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def get_sentences(text):
    # Use tokenizer's sentence splitter
    sentences = text.split("\n")
    sentences = [s.strip() for s in sentences if len(s.strip()) > 0]
    return sentences

def parse_ann_file(ann_path):
    claims = []
    premises = []
    with open(ann_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            if parts[0].startswith("T"):
                label = parts[1]
                text = " ".join(parts[4:])
                if "Claim" in label:
                    claims.append(text.strip())
                elif "Premise" in label:
                    premises.append(text.strip())
    return claims, premises

def main():
    data = []

    files = sorted(f for f in os.listdir(BRAT_DIR) if f.endswith(".txt"))
    for txt_file in tqdm(files, desc="Parsing essays"):
        base = txt_file[:-4]
        ann_path = os.path.join(BRAT_DIR, base + ".ann")
        txt_path = os.path.join(BRAT_DIR, txt_file)

        if not os.path.exists(ann_path):
            continue

        with open(txt_path, "r", encoding="utf-8") as f:
            text = f.read()

        sentences = get_sentences(text)
        claims, premises = parse_ann_file(ann_path)

        for sent in sentences:
            label = "Other"
            for claim in claims:
                if claim in sent:
                    label = "Claim"
                    break
            for premise in premises:
                if premise in sent:
                    label = "Premise"
                    break
            data.append({"sentence": sent, "label": label})

    df = pd.DataFrame(data)
    df = df[df.label != "Other"]
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"✅ Saved {len(df):,} sentences to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()