#!/usr/bin/env python3

import os
from sentence_transformers import SentenceTransformer
import numpy as np

INPUT_FILE = "data/arg_sents_combined.txt"
OUTPUT_FILE = "data/arg_embeddings1.npy"
MODEL_NAME = "all-mpnet-base-v2"  # very strong semantic model

def main():
    # Load sentences
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        sentences = [line.strip() for line in f if line.strip()]

    print(f"Loaded {len(sentences):,} sentences.")

    # Load model
    model = SentenceTransformer(MODEL_NAME)

    # Compute embeddings
    embeddings = model.encode(sentences, batch_size=64, show_progress_bar=True)

    # Save
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    np.save(OUTPUT_FILE, embeddings)

    print(f"✅ Saved embeddings to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()