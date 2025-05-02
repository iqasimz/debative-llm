# scripts/stage5/step5_evaluate_relation.py

import csv
from datasets import Dataset, ClassLabel
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)
from sklearn.metrics import precision_recall_fscore_support

MODEL_DIR  = "models/relation_crossencoder"
SEED       = 42
BATCH_SIZE = 16
LABELS     = ["support", "attack", "none"]

def load_dev(path="data/seed_relation_set_fast.csv"):
    claims   = [l.strip() for l in open("data/claims_highconf.txt", encoding="utf-8")]
    premises = [l.strip() for l in open("data/premises_highconf.txt", encoding="utf-8")]
    rows = []
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        all_rows = list(reader)
    # take last 10k as dev
    for r in all_rows[-10000:]:
        rows.append({
            "text_a": claims[int(r["claim_idx"])],
            "text_b": premises[int(r["premise_idx"])],
            "label":  r["label"]
        })
    return rows

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    p, r, f1, _ = precision_recall_fscore_support(
        labels, preds, average=None, labels=[0,1,2]
    )
    for i, lbl in enumerate(LABELS):
        print(f"{lbl:8} → P: {p[i]:.3f}  R: {r[i]:.3f}  F1: {f1[i]:.3f}")
    acc = (preds == labels).mean()
    print(f"Overall accuracy: {acc:.3f}")
    return {}

if __name__ == "__main__":
    print("▶️ Loading dev set…")
    rows = load_dev()
    labels = ClassLabel(names=LABELS)
    for ex in rows:
        ex["label"] = labels.str2int(ex["label"])
    ds = Dataset.from_list(rows)

    print("▶️ Loading tokenizer & model…")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model     = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)

    print("▶️ Tokenizing dev set…")
    ds = ds.map(
        lambda x: tokenizer(x["text_a"], x["text_b"], truncation=True, max_length=256),
        batched=True
    )
    ds.set_format("torch", columns=["input_ids","attention_mask","label"])

    print("▶️ Running evaluation…")
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir="eval_out",
            per_device_eval_batch_size=BATCH_SIZE,
            seed=SEED,
            do_eval=True,
            logging_steps=500,
            save_steps=1000,
        ),
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
    )
    trainer.evaluate(eval_dataset=ds)