#!/usr/bin/env python3
import os
from pathlib import Path
import pandas as pd

# ─── run from project root ───────────────────────────────────────
project_root = Path(__file__).parent.parent.resolve()
os.chdir(project_root)

INPUT_DIR  = project_root / "data" / "debate_dataset"
OUTPUT_TXT = project_root / "data" / "un_debates.txt"

def find_csv():
    csvs = list(INPUT_DIR.glob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No CSV found in {INPUT_DIR}")
    # prefer the one named “un-general-debates.csv”
    for c in csvs:
        if "un-general-debates" in c.name:
            return c
    return csvs[0]

def extract_paragraphs_from_text(text: str):
    # split on two or more line breaks
    paras = [p.strip().replace("\n", " ")
             for p in text.split("\n\n") if p.strip()]
    return paras

def main():
    INPUT_CSV = find_csv()
    print(f"[INFO] Loading {INPUT_CSV.name}")
    df = pd.read_csv(INPUT_CSV, low_memory=False)

    # detect all text-like columns
    text_cols = [col for col, dt in df.dtypes.items() if dt == "object"]
    print(f"[INFO] Found text columns: {text_cols}")

    all_paras = []
    for col in text_cols:
        # skip tiny metadata columns
        # only keep columns with >100 total characters
        total_len = df[col].dropna().str.len().sum()
        if total_len < 500:
            continue
        for cell in df[col].dropna().astype(str):
            paras = extract_paragraphs_from_text(cell)
            all_paras.extend(paras)

    print(f"[INFO] Extracted {len(all_paras)} paragraphs in total")

    OUTPUT_TXT.parent.mkdir(exist_ok=True)
    with open(OUTPUT_TXT, "w", encoding="utf-8") as out:
        for p in all_paras:
            out.write(p + "\n")

    print(f"✅ Written {len(all_paras)} paragraphs to {OUTPUT_TXT}")

if __name__ == "__main__":
    main()