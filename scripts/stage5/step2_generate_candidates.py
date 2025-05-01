# scripts/stage5/step2_generate_candidates.py

import faiss
from sentence_transformers import SentenceTransformer
import csv

def load_list(path):
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

if __name__ == "__main__":
    # 1. Load high-confidence claims & premises
    claims   = load_list("data/claims_highconf.txt")
    premises = load_list("data/premises_highconf.txt")
    print(f"Loaded {len(claims)} claims and {len(premises)} premises.")

    # 2. Load bi-encoder model
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # 3. Encode premises & build FAISS index
    print("Encoding premises…")
    prem_emb = model.encode(premises,
                            convert_to_numpy=True,
                            batch_size=64,
                            show_progress_bar=True)
    # Normalize for cosine-sim
    faiss.normalize_L2(prem_emb)
    index = faiss.IndexFlatIP(prem_emb.shape[1])
    index.add(prem_emb)
    print(f"FAISS index ready with {index.ntotal} vectors.")

    # 4. Encode claims & retrieve top-K neighbors
    print("Encoding claims & searching…")
    claim_emb = model.encode(claims,
                             convert_to_numpy=True,
                             batch_size=64,
                             show_progress_bar=True)
    faiss.normalize_L2(claim_emb)

    K = 5
    D, I = index.search(claim_emb, K)  # D: cosine similarities, I: indices

    # 5. Flatten results
    candidates = [
        (ci, int(pj), float(score))
        for ci, (scores, ids) in enumerate(zip(D, I))
        for score, pj in zip(scores, ids)
    ]
    print(f"Generated {len(candidates)} claim→premise pairs.")

    # 6. Save to CSV
    with open("data/seed_candidates.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["claim_idx", "premise_idx", "sim"])
        writer.writerows(candidates)
    print("→ Saved data/seed_candidates.csv")