#!/usr/bin/env python3
import os
from pathlib import Path
import spacy
from tqdm import tqdm

# ─── Ensure we run from project root ────────────────────────────
project_root = Path(__file__).parent.parent.parent.resolve()
os.chdir(project_root)

# ─── Inputs & outputs ───────────────────────────────────────────
INPUT    = project_root / "data" / "combined_corpus.txt"
OUTPUT   = project_root / "data" / "sentences.txt"
MIN_LEN  = 20   # drop very short fragments
# ─────────────────────────────────────────────────────────────────

def main():
    # 1) load spaCy English model
    print("Loading spaCy model…")
    nlp = spacy.load("en_core_web_sm", disable=["tagger","parser","ner"])
    nlp.add_pipe("sentencizer")  # only need sentence boundary detection

    # 2) prepare output
    OUTPUT.parent.mkdir(exist_ok=True)
    out_f = OUTPUT.open("w", encoding="utf-8")

    # 3) read+split
    total_in = 0
    total_out = 0
    with INPUT.open("r", encoding="utf-8") as in_f:
        for line in tqdm(in_f, desc="Splitting into sentences"):
            total_in += 1
            text = line.strip()
            if not text:
                continue
            doc = nlp(text)
            for sent in doc.sents:
                s = sent.text.strip()
                if len(s) >= MIN_LEN:
                    out_f.write(s + "\n")
                    total_out += 1

    out_f.close()
    print(f"\n✅ Processed {total_in} paragraphs → {total_out} sentences")
    print(f"   Wrote sentences to {OUTPUT}")

if __name__ == "__main__":
    main()