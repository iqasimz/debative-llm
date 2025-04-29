#!/usr/bin/env python3
"""
scripts/stage4/augment.py

Efficient augmentation script:
 - Pre-loads models once
 - Uses global model objects for augmentation
 - Avoids repeated disk loads
 - Supports back-translation, paraphrase, and MLM

Inputs:
  --input           Path to CSV of original gold examples (sentence,label)
  --output          Path where augmented CSV will be written (sentence,label)

Augmentation Options:
  --backtranslate N  Number of back-translation (En→De→En) augmentations per sentence
  --paraphrase N     Number of T5 paraphrase augmentations per sentence
  --mlm N            Number of MLM mask-fill augmentations per sentence
  --max_nf M         Max synthetic augmentations per sentence across all methods

Usage example:
  python scripts/stage4/augment.py \
    --input data/gold_combined.csv \
    --backtranslate 1 --paraphrase 1 --mlm 0 --max_nf 2 \
    --output data/gold_augmented.csv
"""

import argparse
import pandas as pd
import random
from tqdm import tqdm
import torch
from transformers import (
    MarianMTModel, MarianTokenizer,
    T5ForConditionalGeneration, T5Tokenizer,
    BertForMaskedLM, BertTokenizerFast
)

# Detect device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Pre-load models once
print(f"Loading models on {DEVICE}...")
# Back-translation models
BT_SRC = 'Helsinki-NLP/opus-mt-en-de'
BT_TGT = 'Helsinki-NLP/opus-mt-de-en'
bt_tokenizer = MarianTokenizer.from_pretrained(BT_SRC)
bt_model     = MarianMTModel.from_pretrained(BT_SRC).to(DEVICE)
dt_tokenizer = MarianTokenizer.from_pretrained(BT_TGT)
dt_model     = MarianMTModel.from_pretrained(BT_TGT).to(DEVICE)
# Paraphrasing model
pp_model_name = 't5-small'
pp_tokenizer  = T5Tokenizer.from_pretrained(pp_model_name)
pp_model      = T5ForConditionalGeneration.from_pretrained(pp_model_name).to(DEVICE)
# MLM model
mlm_name      = 'bert-base-uncased'
mlm_tokenizer = BertTokenizerFast.from_pretrained(mlm_name)
mlm_model     = BertForMaskedLM.from_pretrained(mlm_name).to(DEVICE)
print("Models loaded.\n")

def backtranslate(sent, n):
    """Perform English→German→English back-translation."""
    outs = []
    for _ in range(n):
        inputs = bt_tokenizer([sent], return_tensors='pt', padding=True).to(DEVICE)
        de_ids = bt_model.generate(**inputs, max_length=128)
        de_text = bt_tokenizer.batch_decode(de_ids, skip_special_tokens=True)
        inputs2 = dt_tokenizer(de_text, return_tensors='pt', padding=True).to(DEVICE)
        en_ids = dt_model.generate(**inputs2, max_length=128)
        outs.extend(dt_tokenizer.batch_decode(en_ids, skip_special_tokens=True))
    return outs

def paraphrase(sent, n):
    """Generate paraphrases using T5."""
    inputs = pp_tokenizer.encode("paraphrase: " + sent,
                                 return_tensors='pt', truncation=True, max_length=256).to(DEVICE)
    outputs = pp_model.generate(inputs, max_length=256,
                                num_beams=5, num_return_sequences=n, early_stopping=True)
    return [pp_tokenizer.decode(o, skip_special_tokens=True) for o in outputs]

def mlm_fill(sent, n):
    """Randomly mask tokens and fill with BERT MLM."""
    tokens = mlm_tokenizer.tokenize(sent)
    results = []
    for _ in range(n):
        tcopy = tokens.copy()
        mask_count = max(1, int(0.1 * len(tcopy)))
        for idx in random.sample(range(len(tcopy)), mask_count):
            tcopy[idx] = mlm_tokenizer.mask_token
        input_ids = mlm_tokenizer.encode(" ".join(tcopy), return_tensors='pt').to(DEVICE)
        with torch.no_grad():
            logits = mlm_model(input_ids).logits
        for pos in (input_ids[0] == mlm_tokenizer.mask_token_id).nonzero(as_tuple=True)[0]:
            top_id = torch.topk(logits[0, pos], 1).indices.item()
            tcopy[pos] = mlm_tokenizer.convert_ids_to_tokens(top_id)
        results.append(mlm_tokenizer.convert_tokens_to_string(tcopy))
    return results

def main():
    parser = argparse.ArgumentParser(
        description="Efficient data augmentation script (backtranslate, paraphrase, mlm)"
    )
    parser.add_argument('--input',           required=True, help='Input CSV (sentence,label)')
    parser.add_argument('--output',          required=True, help='Output augmented CSV')
    parser.add_argument('--backtranslate', type=int, default=0, help='Num back-translations per sentence')
    parser.add_argument('--paraphrase',    type=int, default=0, help='Num paraphrases per sentence')
    parser.add_argument('--mlm',           type=int, default=0, help='Num MLM fills per sentence')
    parser.add_argument('--max_nf',        type=int, default=2, help='Max augmentations per sentence')
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    print(f"Loaded {len(df)} examples from {args.input}\nStarting augmentation...")

    augmented = []
    for sent, label in tqdm(zip(df['sentence'], df['label']), total=len(df)):
        augmented.append((sent, label))
        added = 0
        for fn, cnt in [(backtranslate, args.backtranslate),
                        (paraphrase,    args.paraphrase),
                        (mlm_fill,      args.mlm)]:
            for aug_sent in fn(sent, cnt):
                if added < args.max_nf:
                    augmented.append((aug_sent, label))
                    added += 1

    aug_df = pd.DataFrame(augmented, columns=['sentence','label']) \
                   .drop_duplicates() \
                   .sample(frac=1, random_state=42)
    print(f"\nDeduplicated → {len(aug_df)} total examples\nSaving to {args.output}...")
    aug_df.to_csv(args.output, index=False)
    print("Augmentation complete.")

if __name__ == '__main__':
    main()