#!/usr/bin/env python3
import torch
from transformers import pipeline
from tqdm import tqdm

MODEL     = "valhalla/distilbart-mnli-12-3"
LABELS    = ["argumentative", "non-argumentative"]
INPUT     = "data/sentences.txt"
OUTPUT    = "data/arg_sents_mnli.txt"
BATCH     = 64

def main():
    # 1) Choose device: MPS if available else CPU
    device = 0 if torch.backends.mps.is_available() else -1
    print(f"Using device: {'mps' if device==0 else 'cpu'}")

    # 2) Initialize zero-shot pipeline
    classifier = pipeline(
        "zero-shot-classification",
        model=MODEL,
        device=device,
        batch_size=BATCH
    )

    # 3) Read all sentences
    with open(INPUT, "r", encoding="utf-8") as f:
        sents = [l.strip() for l in f if l.strip()]

    # 4) Classify in batches
    out = open(OUTPUT, "w", encoding="utf-8")
    for i in tqdm(range(0, len(sents), BATCH), desc="Zero‑shot MNLI"):
        batch = sents[i : i + BATCH]
        results = classifier(batch, LABELS)
        for sent, res in zip(batch, results):
            # If the top label is "argumentative", keep it
            if res["labels"][0] == "argumentative":
                out.write(sent + "\n")
    out.close()

    print(f"✅ Wrote argumentative sentences to {OUTPUT}")
    print("Count:", sum(1 for _ in open(OUTPUT)))

if __name__ == "__main__":
    main()