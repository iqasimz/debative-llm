# scripts/stage3/cluster_sentences_minibatch.py

import numpy as np
from sklearn.cluster import MiniBatchKMeans
import os

# ── Config ─────────────────────────
EMBED_FILE = "data/arg_embeddings1.npy"
OUT_FILE = "data/clusters_minibatch1.npy"
N_CLUSTERS = 2000  # Rough number, we can adjust later
BATCH_SIZE = 2048
# ───────────────────────────────────

def main():
    print("🚀 Loading embeddings...")
    X = np.load(EMBED_FILE)
    print(f"✅ Loaded {X.shape[0]:,} embeddings with dimension {X.shape[1]}.")

    print("🚀 Running MiniBatchKMeans...")
    kmeans = MiniBatchKMeans(n_clusters=N_CLUSTERS, batch_size=BATCH_SIZE, verbose=1)
    labels = kmeans.fit_predict(X)

    print(f"✅ Done clustering into {N_CLUSTERS} rough groups.")

    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
    np.save(OUT_FILE, labels)
    print(f"✅ Saved rough cluster labels to {OUT_FILE}")

if __name__ == "__main__":
    main()