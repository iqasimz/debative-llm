# step1_prepare_cp.py

import pandas as pd

def load_arg_sentences(path="data/arg_sents_combined.txt"):
    """Load all argumentative sentences from Stage 2."""
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

def load_highconf_roles(path="data/student_highconf_all.csv", conf_thr=0.9):
    """Split sentences into high-conf claims vs. premises."""
    df = pd.read_csv(path)
    # Filter by confidence threshold
    claims_df   = df[df.claim_prob   >= conf_thr]
    premises_df = df[df.premise_prob >= conf_thr]
    claims   = claims_df["sentence"].tolist()
    premises = premises_df["sentence"].tolist()
    return claims, premises

if __name__ == "__main__":
    # 1. Load your raw sentences (just to report count)
    all_sents = load_arg_sentences()
    print(f"Total argumentative sentences (Stage 2): {len(all_sents)}")

    # 2. Load & filter high-confidence roles
    claims, premises = load_highconf_roles()
    print(f"High-conf claims:   {len(claims)}")
    print(f"High-conf premises: {len(premises)}")

    # 3. Save to text files for the next steps
    pd.Series(claims).to_csv("data/claims_highconf.txt",
                             index=False, header=False, encoding="utf-8")
    pd.Series(premises).to_csv("data/premises_highconf.txt",
                               index=False, header=False, encoding="utf-8")
    print("→ Saved claims_highconf.txt and premises_highconf.txt")