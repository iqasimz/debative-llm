# scripts/stage5/step6_apply_relation.py

import csv
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)
from datasets import load_dataset, Dataset
from tqdm.auto import tqdm

MODEL_DIR     = "models/relation_crossencoder"
CANDIDATES    = "data/seed_relation_set_fast.csv"
OUTPUT_PATH   = "data/relation_scores_all.csv"
BATCH_SIZE    = 32

def load_texts():
    claims   = [l.strip() for l in open("data/claims_highconf.txt", encoding="utf-8")]
    premises = [l.strip() for l in open("data/premises_highconf.txt", encoding="utf-8")]
    return claims, premises

def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model     = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).eval()
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    claims, premises = load_texts()
    rows = list(csv.DictReader(open(CANDIDATES, encoding="utf-8")))

    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as fout:
        writer = csv.writer(fout)
        writer.writerow(["claim_idx","premise_idx","support_prob","attack_prob","none_prob"])
        for i in tqdm(range(0, len(rows), BATCH_SIZE)):
            batch = rows[i:i+BATCH_SIZE]
            texts_a = [claims[int(r["claim_idx"])] for r in batch]
            texts_b = [premises[int(r["premise_idx"])] for r in batch]
            enc = tokenizer(texts_a, texts_b, padding=True, truncation=True, return_tensors="pt", max_length=256)
            enc = {k:v.to(device) for k,v in enc.items()}
            with torch.no_grad():
                logits = model(**enc).logits
                probs  = torch.softmax(logits, dim=-1).cpu().tolist()
            for r, p in zip(batch, probs):
                writer.writerow([r["claim_idx"], r["premise_idx"], *p])

if __name__=="__main__":
    main()