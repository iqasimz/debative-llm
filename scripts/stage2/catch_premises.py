import re, os, torch
from transformers import pipeline
from tqdm.auto import tqdm

# ── Config ────────────────────────────────────────────
INPUT_FILE  = "data/original_sentences.txt"
OUTPUT_FILE = "data/premises_caught.txt"

MODEL       = "valhalla/distilbart-mnli-12-3"
LABELS      = ["argumentative", "non-argumentative"]
THRESH_LOW  = 0.6
THRESH_HIGH = 0.75

ARG_MARKER  = re.compile(r"\b(because|therefore|thus|hence|must|should|since|if|as a result|implies|consequently|suggests|due to)\b", re.I)

BATCH_SIZE  = 8
MEGA_CHUNK  = 300
# ───────────────────────────────────────────────────────

def main():
    # Load sentences
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"Missing {INPUT_FILE}")
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        sentences = [l.strip() for l in f if l.strip()]
    total = len(sentences)
    print(f"📄 Loaded {total:,} leftover sentences.")

    # Build MNLI pipeline
    device = 0 if torch.cuda.is_available() else -1
    device_name = "cuda" if device==0 else "cpu"
    print(f"🚀 Using device: {device_name}")
    clf = pipeline(
        "zero-shot-classification",
        model=MODEL,
        device=device,
        batch_size=BATCH_SIZE,
        truncation=True,
        fp16=True if device == 0 else False
    )

    kept = []
    pbar = tqdm(total=total, desc="🛠️ Mining premises", dynamic_ncols=True)

    for start in range(0, total, MEGA_CHUNK):
        chunk = sentences[start : start + MEGA_CHUNK]
        results = clf(chunk, LABELS)

        for sent, res in zip(chunk, results):
            score = res["scores"][res["labels"].index("argumentative")]
            if THRESH_LOW <= score < THRESH_HIGH or ARG_MARKER.search(sent):
                kept.append(sent)

        if device == 0:
            torch.cuda.empty_cache()
        pbar.update(len(chunk))
        pbar.set_postfix_str(f"Kept: {len(kept):,}")

    pbar.close()

    # Save results
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
        out.write("\n".join(kept))

    print(f"\n✅ Mining complete.")
    print(f"✅ {len(kept):,} new premises saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()