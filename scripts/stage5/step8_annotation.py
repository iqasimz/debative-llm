import pandas as pd
from transformers import pipeline
from tqdm import tqdm

# 1) Load your CSV
df = pd.read_csv("data/to_annotate_relation.csv")

# 2) Instantiate the ZERO-SHOT classifier
classifier = pipeline(
    "zero-shot-classification",
    model="MoritzLaurer/deberta-v3-large-zeroshot-v2.0"
)

# 3) Define your candidate labels (use "none" to match your seed format)
candidate_labels = ["support", "attack", "none"]

# 4) Classification function
def classify_relation(claim, premise):
    result = classifier(
        premise,
        candidate_labels=candidate_labels,
        hypothesis_template="This statement {} the claim."
    )
    # Always take the top label, including "none"
    return result["labels"][0]

# 5) Iterate with progress bar
labels = []
for _, row in tqdm(df.iterrows(), total=len(df), desc="Annotating"):
    lbl = classify_relation(row["claim_sentence"], row["premise_sentence"])
    labels.append(lbl)

df["label"] = labels

# 6) Keep only IDs + label, then save **all** rows
final_df = df[["claim_idx", "premise_idx", "label"]]
final_df.to_csv("data/manual_relation_labels.csv", index=False)

print(f"\n✅ Done! Annotated {len(final_df)} rows (including 'none') and saved to data/manual_relation_labels.csv")