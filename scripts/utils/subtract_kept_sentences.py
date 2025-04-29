# scripts/utils/subtract_kept_sentences.py

import os

# Paths
original_file = "data/sentences.txt"
kept_file = "data/arg_sents_heuristic.txt"
output_file = "data/sentences_leftover.txt"

# 1) Load original and kept sentences
print("📄 Loading files...")
all_sentences = set(l.strip() for l in open(original_file, encoding="utf-8") if l.strip())
kept_sentences = set(l.strip() for l in open(kept_file, encoding="utf-8") if l.strip())

print(f"Total original sentences: {len(all_sentences):,}")
print(f"Total kept sentences: {len(kept_sentences):,}")

# 2) Subtract
leftover_sentences = all_sentences - kept_sentences
print(f"🧹 Sentences after subtraction (leftover): {len(leftover_sentences):,}")

# 3) Save
os.makedirs(os.path.dirname(output_file), exist_ok=True)
with open(output_file, "w", encoding="utf-8") as f:
    for sent in leftover_sentences:
        f.write(sent + "\n")

print(f"✅ Leftover sentences saved to {output_file}")