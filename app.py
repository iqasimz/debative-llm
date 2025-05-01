# app.py
import os
import re
import streamlit as st
import pandas as pd
import torch
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
)
import gdown

# Keys & paths
MODEL_DIR   = "models"
ROLE_DIR    = os.path.join(MODEL_DIR, "role_student_v2_highconf")
ROLE_FOLDER = "https://drive.google.com/drive/folders/1OhHRz99AKzQfYXc1iEGjItzxS3iwjAtJ?usp=share_link"

@st.cache_resource
def fetch_and_load_folder(drive_folder_url: str, local_dir: str):
    """Download an entire Drive folder into local_dir and load the model."""
    if not os.path.isdir(local_dir):
        os.makedirs(local_dir, exist_ok=True)
        # this will recursively download the folder contents
        gdown.download_folder(
            url=drive_folder_url,
            output=MODEL_DIR,
            use_cookies=False,
            quiet=False
        )
    # Now load the model from local_dir
    tok = DistilBertTokenizerFast.from_pretrained(local_dir)
    mod = DistilBertForSequenceClassification.from_pretrained(local_dir).eval()
    return tok, mod

@st.cache_resource
def load_models():
    c_tok, c_mod = fetch_and_load_folder(ROLE_FOLDER, ROLE_DIR)
    return (c_tok, c_mod)

# load once
claim_tok, claim_mod = load_models()

# -- rest of your Streamlit code unchanged --