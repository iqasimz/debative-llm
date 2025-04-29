#!/usr/bin/env python3
import os
from pathlib import Path
from tokenizers import Tokenizer, trainers, models, pre_tokenizers, normalizers
from tokenizers.processors import TemplateProcessing

# ─── ensure we always run from the project root ────────────────────
project_root = Path(__file__).parent.parent.resolve()
os.chdir(project_root)

# ─── config ────────────────────────────────────────────────────────
CORPUS_FILE = project_root / "data" / "combined_corpus.txt"
OUTPUT_FILE = project_root / "debate_tokenizer.json"
VOCAB_SIZE  = 8000
# ────────────────────────────────────────────────────────────────────

def main():
    if not CORPUS_FILE.exists():
        raise FileNotFoundError(f"Cannot find corpus at {CORPUS_FILE}")

    # 1) Initialize empty BPE tokenizer
    tokenizer = Tokenizer(models.BPE())

    # 2) Normalizer & pre‐tokenizer
    tokenizer.normalizer = normalizers.Sequence([
        normalizers.NFD(),
        normalizers.Lowercase(),
        normalizers.StripAccents()
    ])
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

    # 3) Trainer configuration
    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        min_frequency=2,
        special_tokens=["<s>", "</s>", "<unk>", "<pad>", "<mask>"],
    )

    # 4) Train on your combined corpus
    print(f"[INFO] Training BPE tokenizer on {CORPUS_FILE} with vocab size {VOCAB_SIZE}…")
    tokenizer.train([str(CORPUS_FILE)], trainer)

    # 5) Post‐processing: add BOS/EOS tokens
    tokenizer.post_processor = TemplateProcessing(
        single="<s> $A </s>",
        pair="<s> $A </s> <s> $B </s>",
        special_tokens=[
            ("<s>", tokenizer.token_to_id("<s>")),
            ("</s>", tokenizer.token_to_id("</s>")),
        ],
    )

    # 6) Save to JSON
    tokenizer.save(str(OUTPUT_FILE))
    print(f"✅ Saved tokenizer to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()