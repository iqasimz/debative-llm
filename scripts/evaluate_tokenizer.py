#!/usr/bin/env python3
import os
import random
from pathlib import Path
from collections import Counter

import pandas as pd
from tokenizers import Tokenizer

# ─── Force cwd to project root ─────────────────────────────────────
project_root = Path(__file__).parent.parent.resolve()
os.chdir(project_root)

# ─── Configuration ────────────────────────────────────────────────
TOKENIZER_FILE = project_root / "debate_tokenizer.json"
CORPUS_FILE    = project_root / "data" / "combined_corpus.txt"
SAMPLE_SIZE    = 1000     # number of lines to sample from the corpus
EXAMPLES       = [
    "The climate crisis demands immediate action.",
    "Is social media harmful to democracy?",
    "Renewable energy investments will pay off.",
    "We must protect data privacy in the digital age.",
    "Artificial intelligence can improve healthcare."
]
# ───────────────────────────────────────────────────────────────────

def inspect_vocab(tk):
    vocab = tk.get_vocab()  # token → id
    inv_vocab = {vid: tok for tok, vid in vocab.items()}
    size = len(vocab)
    special = [t for t in ["<s>","</s>","<unk>","<pad>","<mask>"] if t in vocab]
    first10 = [(i, inv_vocab[i]) for i in range(10)]
    last10  = [(i, inv_vocab[i]) for i in sorted(inv_vocab.keys())[-10:]]
    print(f"\n∘ Vocabulary size: {size}")
    print(f"∘ Special tokens: {special}")
    print("∘ First 10 tokens:", first10)
    print("∘ Last  10 tokens:", last10)

def sample_encodings(tk):
    print("\n=== Sample Encodings ===")
    for text in EXAMPLES:
        enc = tk.encode(text)
        print(f"\nInput: {text}")
        print("Tokens:", enc.tokens)
        print(" IDs:  ", enc.ids)

def corpus_stats(tk):
    print(f"\n=== Corpus Statistics (sample of {SAMPLE_SIZE} lines) ===")
    # load lines
    with open(CORPUS_FILE, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]
    sample = random.sample(lines, min(SAMPLE_SIZE, len(lines)))

    unk = 0
    lengths = []
    token_counter = Counter()
    for line in sample:
        enc = tk.encode(line)
        tokens = enc.tokens
        unk += tokens.count("<unk>")
        lengths.append(len(tokens))
        token_counter.update(tokens)

    total_tokens = sum(lengths)
    avg_len = sum(lengths) / len(lengths)
    unk_rate = unk / total_tokens * 100
    coverage = len(token_counter) / tk.get_vocab_size() * 100

    print(f"• Total tokens:      {total_tokens}")
    print(f"• Unknown tokens:    {unk} ({unk_rate:.2f}%)")
    print(f"• Avg tokens/line:   {avg_len:.2f}")
    print(f"• Unique coverage:   {coverage:.2f}% of vocab used in sample")

    # length distribution
    dist = Counter()
    for L in lengths:
        if L <= 5:      dist["1–5"] += 1
        elif L <= 10:   dist["6–10"] += 1
        elif L <= 20:   dist["11–20"] += 1
        else:           dist["21+"] += 1

    print("\n• Token‑length distribution (lines):")
    for bucket in ["1–5","6–10","11–20","21+"]:
        print(f"   {bucket:6} : {dist[bucket]}")

    # top 10 most frequent tokens in sample
    most = token_counter.most_common(10)
    print("\n• Top 10 tokens in sample:")
    for tok, cnt in most:
        print(f"   {tok!r:10} : {cnt}")

def main():
    print("Loading tokenizer…")
    tk = Tokenizer.from_file(str(TOKENIZER_FILE))

    inspect_vocab(tk)
    sample_encodings(tk)
    corpus_stats(tk)

if __name__ == "__main__":
    main()