# scripts/stage5/step3_score_candidates_fast.py

import csv
import random
import re
import logging

import torch
import spacy
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ── CONFIG ─────────────────────────────────────────────────────────────────────
MNLI_MODEL   = "valhalla/distilbart-mnli-12-3"
MNLI_THRESH  = 0.8
BOOST_SCORE  = 0.9
NONE_RATIO   = 2
CAP          = 100_000     # number of candidate pairs to score
SEED         = 42
BATCH_SIZE   = 64          # MNLI inference batch size

# Expanded cue lists
SUP_CUES = {
    "because","furthermore","moreover","in addition","additionally",
    "for example","for instance","indeed","thus","hence","therefore"
}
ATK_CUES = {
    "however","although","yet","nevertheless","notwithstanding",
    "on the other hand","in contrast","but","whereas","despite"
}

IF_THEN_RE = re.compile(r"\bif\b.+\bthen\b", re.IGNORECASE)
NEG_RE     = re.compile(r"\bnot\b|\bno\b|\bnever\b|\bnone\b", re.IGNORECASE)

# ── LOGGING ────────────────────────────────────────────────────────────────────
logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s",
                    level=logging.INFO)
log = logging.getLogger(__name__)

# ── HEURISTICS ─────────────────────────────────────────────────────────────────
def heuristic_boost(s2: str):
    """Return (support_boost, attack_boost) for a premise sentence."""
    t = s2.lower()
    sup = BOOST_SCORE if any(c in t for c in SUP_CUES) else 0.0
    atk = BOOST_SCORE if any(c in t for c in ATK_CUES) else 0.0

    if IF_THEN_RE.search(t):
        sup = max(sup, BOOST_SCORE)
    if NEG_RE.search(t):
        # flip the stronger cue
        if sup > atk:
            atk = max(atk, BOOST_SCORE * 0.8)
        else:
            sup = max(sup, BOOST_SCORE * 0.8)

    # subordinating conjunctions → support
    for tok in nlp(s2):
        if tok.dep_ in {"advcl","csubj","csubjpass"}:
            sup = max(sup, BOOST_SCORE)
    return sup, atk

# ── MAIN ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    random.seed(SEED)
    log.info("Loading data…")

    # 1. Load sentences
    with open("data/claims_highconf.txt", encoding="utf-8") as f:
        claims = [l.strip() for l in f if l.strip()]
    with open("data/premises_highconf.txt", encoding="utf-8") as f:
        premises = [l.strip() for l in f if l.strip()]

    # 2. Load candidate pairs
    with open("data/seed_candidates.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    random.shuffle(rows)
    rows = rows[:CAP]
    log.info(f"Capped to {len(rows)} candidate pairs for scoring.")

    # 3. Select device: MPS (macOS), then CUDA, else CPU
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    log.info(f"Using device: {device}")

    # 4. Load models
    log.info("Loading spaCy model…")
    nlp = spacy.load("en_core_web_sm", disable=["ner","textcat"])
    log.info(f"Loading MNLI model {MNLI_MODEL}…")
    tokenizer = AutoTokenizer.from_pretrained(MNLI_MODEL)
    model     = AutoModelForSequenceClassification.from_pretrained(MNLI_MODEL).to(device).eval()

    # 5. Batched scoring
    positives, none = [], []
    log.info("Scoring pairs in batches…")
    for i in tqdm(range(0, len(rows), BATCH_SIZE), desc="Batched scoring", unit="batch"):
        batch = rows[i : i + BATCH_SIZE]
        s1_list = [claims[int(r["claim_idx"])]   for r in batch]
        s2_list = [premises[int(r["premise_idx"])] for r in batch]

        # Tokenize + move to device
        inputs = tokenizer(
            s1_list,
            s2_list,
            truncation=True,
            padding="longest",
            return_tensors="pt",
            max_length=256
        ).to(device)

        with torch.no_grad():
            logits = model(**inputs).logits
            probs  = torch.softmax(logits, dim=-1).cpu()

        # Label each pair
        for r, p in zip(batch, probs):
            sup_s = p[2].item()  # entailment → support
            atk_s = p[0].item()  # contradiction → attack

            # heuristic boosts on the premise text
            hb_sup, hb_atk = heuristic_boost(premises[int(r["premise_idx"])])
            sup_s, atk_s = max(sup_s, hb_sup), max(atk_s, hb_atk)

            ci, pj = int(r["claim_idx"]), int(r["premise_idx"])
            if sup_s >= MNLI_THRESH and sup_s >= atk_s:
                positives.append((ci, pj, "support", sup_s))
            elif atk_s >= MNLI_THRESH and atk_s > sup_s:
                positives.append((ci, pj, "attack", atk_s))
            else:
                none.append((ci, pj, "none", 1.0))

    # 6. Balance ‘none’ examples
    random.shuffle(none)
    none = none[: len(positives) * NONE_RATIO]
    seed_set = positives + none
    random.shuffle(seed_set)
    log.info(f"Final seed set: {len(seed_set)} examples "
             f"({len(positives)} pos + {len(none)} none).")

    # 7. Save output
    out_path = "data/seed_relation_set_fast.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["claim_idx", "premise_idx", "label", "score"])
        writer.writerows(seed_set)
    log.info(f"Saved seed set to {out_path}")