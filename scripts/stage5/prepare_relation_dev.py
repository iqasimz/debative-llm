#!/usr/bin/env python3
"""
Clean NaNs in ‘relation’ and split balanced gold into train/dev.
"""

import pandas as pd
from sklearn.model_selection import train_test_split

# 1) Load and fill NaNs
df = pd.read_csv("data/relation_gold_balanced.csv")
print("Original label counts (with NaN):")
print(df["relation"].value_counts(dropna=False))

df["relation"] = df["relation"].fillna("None")
print("\nAfter fillna('None'):")
print(df["relation"].value_counts())

# 2) Stratified split
train, dev = train_test_split(
    df,
    test_size=0.1,
    stratify=df["relation"],
    random_state=42
)

# 3) Save
train.to_csv("data/relation_train.csv", index=False)
dev.to_csv("data/relation_dev_gold.csv", index=False)
print(f"\nSaved train/dev sizes: {len(train)}/{len(dev)}")