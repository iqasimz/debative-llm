import numpy as np
import random
import matplotlib.pyplot as plt

# Load sentences
with open("data/arg_sents_combined.txt", "r", encoding="utf-8") as f:
    sentences = [l.strip() for l in f if l.strip()]

# Load cluster labels
clusters = np.load("data/clusters_final1.npy")

assert len(sentences) == len(clusters), "Mismatch between sentences and clusters!"

# Create a mapping: cluster_id -> list of sentences
from collections import defaultdict
cluster_map = defaultdict(list)
for sent, cid in zip(sentences, clusters):
    cluster_map[cid].append(sent)

# === Step 1: Sample random clusters ===
print("\n=== SAMPLE CLUSTERS ===\n")
sampled_clusters = random.sample(list(cluster_map.keys()), 10)  # Pick 10 random clusters

for cid in sampled_clusters:
    print(f"\n📦 Cluster {cid} (Size: {len(cluster_map[cid])} sentences):")
    for sent in cluster_map[cid][:5]:  # Print first 5 sentences for each cluster
        print(f" - {sent}")

# === Step 2: Basic cluster statistics ===
sizes = [len(sents) for sents in cluster_map.values()]
print("\n=== CLUSTER STATISTICS ===")
print(f"Total clusters: {len(cluster_map)}")
print(f"Min cluster size: {min(sizes)}")
print(f"Max cluster size: {max(sizes)}")
print(f"Mean cluster size: {np.mean(sizes):.2f}")
print(f"Median cluster size: {np.median(sizes):.2f}")

# === Step 3: Optional - Histogram ===
plt.hist(sizes, bins=50, color='skyblue')
plt.title("Cluster Size Distribution")
plt.xlabel("Number of sentences")
plt.ylabel("Number of clusters")
plt.yscale('log')  # use log scale for better visualization
plt.grid(True)
plt.show()