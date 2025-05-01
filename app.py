# app.py
# app.py
import os
import zipfile
import re

import streamlit as st
import pandas as pd
import torch
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
)
import gdown

# Heuristic single-word premise keywords
PREMISE_KEYWORDS = {"because", "if", "since", "due to", "therefore"}
# Heuristic multi-word premise phrases
PHRASE_KEYWORDS  = {"given that", "only when"}

# Paths & Drive IDs
MODEL_DIR  = "models"
ROLE_ZIP   = os.path.join(MODEL_DIR, "role_student_v2_highconf.zip")
ROLE_DIR   = os.path.join(MODEL_DIR, "role_student_v2_highconf")
# TODO: replace with your actual Google Drive file ID
ROLE_ID    = "1kOUNmrq-rJvgeF3Xy-8rrOvslC1Wv6Ng"


def fetch_and_unpack(path_dir: str, path_zip: str, drive_id: str):
    """Download from Google Drive and unzip into path_dir, if not already."""
    if not os.path.isdir(path_dir):
        os.makedirs(path_dir, exist_ok=True)
        url = f"https://drive.google.com/uc?id={drive_id}"
        # download zip file
        gdown.download(url, path_zip, quiet=False)
        # unzip
        with zipfile.ZipFile(path_zip, "r") as z:
            z.extractall(path_dir)
    return path_dir

@st.cache_resource
def load_models():
    # 1. Fetch & unpack the role tagger
    role_path = fetch_and_unpack(ROLE_DIR, ROLE_ZIP, ROLE_ID)

    # 2. Load from the unpacked directory
    c_tok = DistilBertTokenizerFast.from_pretrained(role_path)
    c_mod = DistilBertForSequenceClassification.from_pretrained(role_path).eval()
    return (c_tok, c_mod)

# Load once
(claim_tok, claim_mod) = load_models()

st.title("🔵 Debative-LLM Demo")
input_text = st.text_area("Enter sentences (one per line)", height=250)

if st.button("Analyze"):
    sentences = [s.strip() for s in input_text.splitlines() if s.strip()]
    if not sentences:
        st.error("Need at least one sentence.")
        st.stop()

    # — Claim/Premise with softmax scores & improved heuristics —
    enc    = claim_tok(sentences, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = claim_mod(**enc)
        logits  = outputs.logits               # shape (B,2)
        probs   = torch.softmax(logits, dim=1) # shape (B,2)

    records = []
    for sent, prob in zip(sentences, probs):
        p_claim   = prob[0].item()
        p_premise = prob[1].item()

        # 1) Threshold-based initial choice
        label = "Claim" if p_claim > p_premise else "Premise"

        # 2) Heuristic overrides:
        low    = sent.lower()
        tokens = re.findall(r"\b\w+\b", low)

        # a) premise-keyword check
        cond_keyword = any(kw in tokens for kw in PREMISE_KEYWORDS) \
                    or any(ph in low for ph in PHRASE_KEYWORDS)

        # b) subordinate-clause detection (requires comma)
        has_comma_after = "," in low.split(" ", 1)[1] if " " in low else False
        cond_subord    = low.startswith(("when ", "if ", "although ")) and has_comma_after

        # c) “if … then …” without comma should still flip to Premise
        cond_if_then   = low.startswith("if ") and " then" in low

        # d) only when/although without comma stay Claim
        cond_exception = low.startswith(("when ", "although ")) and not has_comma_after

        # e) data-statement override: bare stats (e.g. “15% drop”) without subordinators → Claim
        cond_data      = bool(re.search(r'\b\d+%?\b', low)) and not (cond_subord or cond_if_then or cond_keyword)

        # final label adjustment
        if cond_exception or cond_data:
            label = "Claim"
        elif cond_if_then or (label == "Claim" and (cond_keyword or cond_subord)):
            label = "Premise"

        records.append({
            "Sentence":    sent,
            "P(Claim)":    f"{p_claim:.3f}",
            "P(Premise)":  f"{p_premise:.3f}",
            "Final Label": label
        })

    df_cp = pd.DataFrame.from_records(records)
    st.subheader("1️⃣ Claim vs. Premise (with softmax scores)")
    st.dataframe(df_cp, use_container_width=True)

    st.success("Done!")