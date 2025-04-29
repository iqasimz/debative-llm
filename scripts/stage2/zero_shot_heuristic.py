
"""
Stage 2: Zero‑Shot + Heuristic classifier for argumentative vs non‑argumentative.

Usage:
  python scripts/stage2/zero_shot_heuristic.py
"""

import re, os, torch
from transformers import pipeline
from tqdm.auto import tqdm

# ── Config ───────────────────────────────────────────────────────────────────────
INPUT_FILE  = "data/sentences.txt"
OUTPUT_FILE = "data/arg_sents_heuristic.txt"

MODEL       = "valhalla/distilbart-mnli-12-3"
LABELS      = ["argumentative", "non-argumentative"]
THRESH      = 0.75            # only keep if MNLI score ≥ this

# any of these words strongly signal an argument
ARG_MARKER  = re.compile(r"\b(because|therefore|must|should|thus|if)\b", re.I)

BATCH_SIZE  = 8              # 🔥 70-min run used 64 batch size
MEGA_CHUNK  = 300             # 🔥 70-min run used 500 sentences per chunk
# ────────────────────────────────────────────────────────────────────────────────

def main():
    # 1) Load sentences
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"Missing {INPUT_FILE}")
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        sentences = [l.strip() for l in f if l.strip()]
    total = len(sentences)
    print(f"📄 Loaded {total:,} sentences.")

    # 2) Build MNLI pipeline
    device = 0 if torch.cuda.is_available() else -1
    device_name = "cuda" if device==0 else "cpu"
    print(f"🚀 Using device: {device_name}")
    clf = pipeline(
        "zero-shot-classification",
        model=MODEL,
        device=device,
        batch_size=BATCH_SIZE,
        truncation=True,
        fp16=True if device == 0 else False    # 🔥 half-precision when using GPU
    )

    # 3) Inference + Heuristic Filtering
    kept = []
    for start in tqdm(range(0, total, MEGA_CHUNK), desc="Chunks"):
        chunk = sentences[start : start + MEGA_CHUNK]
        results = clf(chunk, LABELS)
        for sent, res in zip(chunk, results):
            score = res["scores"][res["labels"].index("argumentative")]
            if score >= THRESH or ARG_MARKER.search(sent):
                kept.append(sent)
        if device == 0:
            torch.cuda.empty_cache()

    # 4) Write results
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
        out.write("\n".join(kept))

    print(f"\n✅ Extraction complete.")
    print(f"✅ {len(kept):,} argumentative sentences saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()