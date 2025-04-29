# scripts/stage4/eval_role_classifier_balanced.py

import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# 📂 Path to your annotated sample
CSV_FILE = "data/sample_role_annotation_balanced.csv"

# ✅ Step 1: Load the sample
print("📄 Loading annotated sample...")
df = pd.read_csv(CSV_FILE)

# ✅ Step 2: Map labels
label_map = {"Claim": 0, "Premise": 1}
df["pred_id"] = df["predicted_role"].map(label_map)
df["true_id"] = df["true_label"].map(label_map)

# ✅ Step 3: Evaluate
accuracy = accuracy_score(df["true_id"], df["pred_id"])
precision = precision_score(df["true_id"], df["pred_id"], average="macro")
recall = recall_score(df["true_id"], df["pred_id"], average="macro")
f1 = f1_score(df["true_id"], df["pred_id"], average="macro")
conf_mat = confusion_matrix(df["true_id"], df["pred_id"])

# ✅ Step 4: Print results
print("\n=== Evaluation Results ===")
print(f"Accuracy:  {accuracy:.3f}")
print(f"Precision: {precision:.3f}")
print(f"Recall:    {recall:.3f}")
print(f"F1-Score:  {f1:.3f}")

print("\nConfusion Matrix:")
print(conf_mat)