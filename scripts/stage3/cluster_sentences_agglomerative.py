# scripts/stage3/cluster_sentences_agglomerative.py

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_distances
import os

# ── Config ─────────────────────────
EMBED_FILE = "data/arg_embeddings1.npy"
ROUGH_CLUSTER_FILE = "data/clusters_minibatch1.npy"
OUT_FILE = "data/clusters_final1.npy"
DISTANCE_THRESHOLD = 0.5  # Merge when sentences are close enough
# ───────────────────────────────────

def main():
    print("🚀 Loading embeddings and rough clusters...")
    X = np.load(EMBED_FILE)
    rough_labels = np.load(ROUGH_CLUSTER_FILE)

    print(f"✅ {X.shape[0]:,} sentences loaded.")

    final_labels = -np.ones(len(X), dtype=int)
    next_cluster_id = 0

    for cluster_id in np.unique(rough_labels):
        idx = np.where(rough_labels == cluster_id)[0]
        if len(idx) < 3:
            # Small clusters → assign directly
            final_labels[idx] = next_cluster_id
            next_cluster_id += 1
            continue

        X_sub = X[idx]
        dist_matrix = cosine_distances(X_sub)

        agg = AgglomerativeClustering(
            metric="precomputed",
            linkage="average",
            distance_threshold=DISTANCE_THRESHOLD,
            n_clusters=None
        )
        sub_labels = agg.fit_predict(dist_matrix)

        for sub_cluster in np.unique(sub_labels):
            idx_sub = idx[sub_labels == sub_cluster]
            final_labels[idx_sub] = next_cluster_id
            next_cluster_id += 1

    print(f"✅ Final number of clusters: {next_cluster_id:,}")

    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
    np.save(OUT_FILE, final_labels)
    print(f"✅ Saved final refined clusters to {OUT_FILE}")

if __name__ == "__main__":
    main()