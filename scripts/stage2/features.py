#!/usr/bin/env python3
import os
from pathlib import Path
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# ─── ensure we run from project root ─────────────────────────────
project_root = Path(__file__).parent.parent.parent.resolve()
os.chdir(project_root)

# ─── paths ────────────────────────────────────────────────────────
SENT_FILE  = project_root / "data" / "sentences.txt"
TFIDF_FILE = project_root / "data" / "tfidf.pkl"
SBERT_FILE = project_root / "data" / "sbert.npy"
# ───────────────────────────────────────────────────────────────────

def build_tfidf():
    print("Loading sentences for TF–IDF…")
    sents = []
    with open(SENT_FILE, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Reading sentences"):
            s = line.strip()
            if s:
                sents.append(s)

    print(f"Fitting TF–IDF on {len(sents)} sentences…")
    vec = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
    X = vec.fit_transform(sents)

    print("Saving TF–IDF vectorizer and sentences…")
    joblib.dump((vec, sents), TFIDF_FILE)
    print(f"✅ TF–IDF saved to {TFIDF_FILE}")

def build_sbert():
    print("Loading sentences for SBERT embeddings…")
    sents = []
    with open(SENT_FILE, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Reading sentences"):
            s = line.strip()
            if s:
                sents.append(s)

    print("Encoding sentences with SBERT (all-MiniLM-L6-v2)…")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(sents, show_progress_bar=True, batch_size=64)

    print("Saving SBERT embeddings…")
    np.save(SBERT_FILE, embeddings)
    print(f"✅ SBERT embeddings saved to {SBERT_FILE} (shape {embeddings.shape})")

if __name__ == "__main__":
    TFIDF_FILE.parent.mkdir(exist_ok=True)
    build_tfidf()
    build_sbert()