#!/usr/bin/env python3
"""
Cluster the top-K uncertain sentences and pick one per cluster for diversity.
Inputs:
  --input_txt   Path to data/uncertain_5k.txt
Outputs:
  --output_csv  Path to data/to_annotate_1k.csv
Options:
  --n_clusters  Number of clusters (default: 1000)
"""

import argparse
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input_txt',  required=True)
    p.add_argument('--output_csv', required=True)
    p.add_argument('--n_clusters', type=int, default=1000)
    args = p.parse_args()

    # 1) Load uncertain sentences
    with open(args.input_txt, encoding='utf8') as f:
        sents = [l.strip() for l in f if l.strip()]
    print(f"Loaded {len(sents)} sentences from {args.input_txt}")

    # 2) Embed with SBERT
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(sents, show_progress_bar=True)

    # 3) Cluster
    print(f"Clustering into {args.n_clusters} groups…")
    km = KMeans(n_clusters=args.n_clusters, random_state=42)
    clusters = km.fit_predict(embeddings)

    # 4) Pick one per cluster
    df = pd.DataFrame({'sentence': sents, 'cluster': clusters})
    sampled = df.groupby('cluster').first().reset_index(drop=True)

    # 5) Save for annotation
    sampled['label'] = ''  # empty column for annotator to fill
    sampled.to_csv(args.output_csv, index=False)
    print(f"Wrote {len(sampled)} sentences to {args.output_csv} for annotation.")

if __name__ == "__main__":
    main()