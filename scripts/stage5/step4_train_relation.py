# scripts/stage5/step4_train_relation.py

import os
import csv
import random
import torch
from datasets import Dataset, ClassLabel
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)
from sklearn.metrics import precision_recall_fscore_support

# ── CONFIG ─────────────────────────────────────────────────────────────────────
MODEL_NAME = "distilroberta-base"
OUTPUT_DIR = "models/relation_crossencoder"
SEED       = 42
EPOCHS     = 3
LR         = 2e-5

# Dropper batch size to avoid MPS OOM
PER_DEVICE_TRAIN_BATCH_SIZE = 8
GRADIENT_ACCUMULATION_STEPS = 4  # effective batch = 8 × 4 = 32
WEIGHT_DECAY               = 0.01

# On MPS, disable the high-watermark to allow larger allocations
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

# ── HELPERS ────────────────────────────────────────────────────────────────────
def load_seed(path="data/seed_relation_set_fast.csv"):
    claims   = [l.strip() for l in open("data/claims_highconf.txt", encoding="utf-8")]
    premises = [l.strip() for l in open("data/premises_highconf.txt", encoding="utf-8")]
    rows = []
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append({
                "text_a": claims[int(r["claim_idx"])],
                "text_b": premises[int(r["premise_idx"])],
                "label":    r["label"]
            })
    return rows

def prepare_dataset(rows):
    labels = ClassLabel(names=["support","attack","none"])
    for ex in rows:
        ex["label"] = labels.str2int(ex["label"])
    ds = Dataset.from_list(rows)
    return ds.train_test_split(test_size=0.1, seed=SEED), labels

def tokenize_fn(examples, tokenizer):
    return tokenizer(
        examples["text_a"],
        examples["text_b"],
        truncation=True,
        max_length=256,
    )

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    p, r, f1, _ = precision_recall_fscore_support(
        labels, preds, average=None, labels=[0,1,2]
    )
    return {
        "precision_support": p[0],
        "recall_support":    r[0],
        "f1_support":        f1[0],
        "precision_attack":  p[1],
        "recall_attack":     r[1],
        "f1_attack":         f1[1],
        "precision_none":    p[2],
        "recall_none":       r[2],
        "f1_none":           f1[2],
    }

# ── MAIN ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    random.seed(SEED)
    torch.manual_seed(SEED)

    print("1️⃣ Loading seed relation set…")
    rows = load_seed()
    splits, label_feat = prepare_dataset(rows)
    print(f" • Train size: {len(splits['train'])}")
    print(f" • Dev size:   {len(splits['test'])}")

    print("2️⃣ Preparing tokenizer & model…")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model     = AutoModelForSequenceClassification.from_pretrained(
                    MODEL_NAME, num_labels=label_feat.num_classes
                )

    print("3️⃣ Tokenizing datasets…")
    tokenized = splits.map(
        lambda x: tokenize_fn(x, tokenizer),
        batched=True,
        remove_columns=["text_a","text_b"]
    )
    tokenized.set_format("torch", columns=["input_ids","attention_mask","label"])

    print("4️⃣ Setting up Trainer…")
    data_collator = DataCollatorWithPadding(tokenizer)
    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        learning_rate=LR,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        weight_decay=WEIGHT_DECAY,
        logging_steps=500,
        save_steps=1000,
        save_total_limit=2,
        seed=SEED,
        # do not enable fp16 on MPS
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    print("5️⃣ Training…")
    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    print(f"✅ Model saved to {OUTPUT_DIR}")