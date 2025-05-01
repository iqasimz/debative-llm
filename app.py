import re
import streamlit as st
import pandas as pd
import torch
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
)

PREMISE_KEYWORDS = {"because", "if", "since", "due to", "therefore"}
PHRASE_KEYWORDS  = {"given that", "only when"}

@st.cache_resource
def load_models():
    model_id = "iqasimz/role_student_v2_highconf"
    tok = DistilBertTokenizerFast.from_pretrained(model_id)
    mod = DistilBertForSequenceClassification.from_pretrained(model_id).eval()
    return tok, mod

claim_tok, claim_mod = load_models()

st.title("🔵 Debative-LLM Demo")
input_text = st.text_area("Enter sentences (one per line)", height=250)

if st.button("Analyze"):
    sentences = [s.strip() for s in input_text.splitlines() if s.strip()]
    if not sentences:
        st.error("Need at least one sentence."); st.stop()

    enc = claim_tok(sentences, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        out   = claim_mod(**enc)
        probs = torch.softmax(out.logits, dim=1)

    records = []
    for sent, prob in zip(sentences, probs):
        p_c, p_p = prob[0].item(), prob[1].item()
        label = "Claim" if p_c > p_p else "Premise"
        low, tokens = sent.lower(), re.findall(r"\b\w+\b", sent.lower())
        cond_kw = any(kw in tokens for kw in PREMISE_KEYWORDS) or any(ph in low for ph in PHRASE_KEYWORDS)
        has_comma = "," in low.split(" ",1)[1] if " " in low else False
        cond_sub = low.startswith(("when ","if ","although ")) and has_comma
        cond_if  = low.startswith("if ") and " then" in low
        cond_ex  = low.startswith(("when ","although ")) and not has_comma
        cond_dt  = bool(re.search(r'\b\d+%?\b', low)) and not (cond_sub or cond_if or cond_kw)

        if cond_ex or cond_dt:
            label = "Claim"
        elif cond_if or (label=="Claim" and (cond_kw or cond_sub)):
            label = "Premise"

        records.append({
            "Sentence":    sent,
            "P(Claim)":    f"{p_c:.3f}",
            "P(Premise)":  f"{p_p:.3f}",
            "Final Label": label
        })

    df = pd.DataFrame(records)
    st.subheader("1️⃣ Claim vs. Premise")
    st.dataframe(df, use_container_width=True)
    st.success("Done!")