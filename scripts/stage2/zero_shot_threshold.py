# debative-llm/scripts/stage2/zero_shot_heuristic_70min_megachunks.py

import os
import torch
from transformers import pipeline
from tqdm import tqdm
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True, help="Path to input sentences.txt")
    parser.add_argument("-o", "--output", required=True, help="Path to output file")
    args = parser.parse_args()

    input_file = args.input
    output_file = args.output

    # Device selection
    device = 0 if torch.cuda.is_available() else -1
    print("Running on", "GPU" if device==0 else "CPU")

    # Load model
    print("📦 Loading model in half-precision (fp16)...")
    classifier = pipeline(
        "zero-shot-classification",
        model="valhalla/distilbart-mnli-12-3",
        device=device,
        batch_size=64,         # <<<<<<<<<<<< Batch size
        truncation=True,
        max_length=512,
        torch_dtype=torch.float16 if device == 0 else None  # FP16 mode
    )
    labels = ["argumentative", "non-argumentative"]

    # Read sentences
    print("📄 Loading sentences...")
    with open(input_file, "r", encoding="utf-8") as f:
        sentences = [line.strip() for line in f if line.strip()]
    print(f"Total sentences loaded: {len(sentences):,}")

    # Mega-chunking
    chunk_size = 500      # <<<<<<<<<<<< Chunk size we used in 70-min run

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    print(f"🚀 Starting inference with chunk size = {chunk_size}...")
    with open(output_file, "w", encoding="utf-8") as out_file:
        for i in tqdm(range(0, len(sentences), chunk_size), desc="Zero-Shot MNLI"):
            mega_batch = sentences[i:i+chunk_size]
            preds = classifier(mega_batch, labels)

            for sent, pred in zip(mega_batch, preds):
                if pred["labels"][0] == "argumentative":
                    out_file.write(sent + "\n")

    print(f"✅ Done. Argumentative sentences written to {output_file}")

if __name__ == "__main__":
    main()