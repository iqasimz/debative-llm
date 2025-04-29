# scripts/stage5/augment_utils.py

import random
import nltk
import torch
from nltk.corpus import wordnet
from transformers import MarianMTModel, MarianTokenizer, pipeline

# Download NLTK data if not already available
nltk.download('wordnet')
nltk.download('omw-1.4')

# Load MarianMT models for back-translation
mt_en_de_tokenizer = MarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-en-de')
mt_en_de_model     = MarianMTModel.from_pretrained('Helsinki-NLP/opus-mt-en-de')
mt_de_en_tokenizer = MarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-de-en')
mt_de_en_model     = MarianMTModel.from_pretrained('Helsinki-NLP/opus-mt-de-en')

# Fill-mask pipeline for MLM-based paraphrasing
mlm_fill = pipeline('fill-mask', model='bert-base-uncased', tokenizer='bert-base-uncased', top_k=5)

def eda_augment(s1: str, s2: str, alpha_sr=0.1, alpha_rd=0.1):
    """
    Easy Data Augmentation:
      - Synonym Replacement (alpha_sr)
      - Random Deletion (alpha_rd)
    Returns two augmented sentences.
    """
    def synonym_replacement(sentence, alpha):
        words = sentence.split()
        n = max(1, int(len(words) * alpha))
        new_words = words.copy()
        for _ in range(n):
            idx = random.randrange(len(new_words))
            synonyms = []
            for syn in wordnet.synsets(new_words[idx]):
                for lemma in syn.lemmas():
                    w = lemma.name().replace('_', ' ')
                    if w.lower() != new_words[idx].lower():
                        synonyms.append(w)
            if synonyms:
                new_words[idx] = random.choice(synonyms)
        return ' '.join(new_words)

    def random_deletion(sentence, alpha):
        words = sentence.split()
        if len(words) == 1:
            return sentence
        new_words = [w for w in words if random.random() > alpha]
        if not new_words:
            new_words = [random.choice(words)]
        return ' '.join(new_words)

    # Apply both operations
    return synonym_replacement(s1, alpha_sr), synonym_replacement(s2, alpha_sr)

def backtranslate_pair(s1: str, s2: str):
    """
    Back-translate each sentence via German to create paraphrases.
    """
    def translate(text, tokenizer, model):
        inputs = tokenizer(text, return_tensors='pt', truncation=True)
        translated = model.generate(**inputs)
        return tokenizer.decode(translated[0], skip_special_tokens=True)

    s1_bt = translate(translate(s1, mt_en_de_tokenizer, mt_en_de_model),
                      mt_de_en_tokenizer, mt_de_en_model)
    s2_bt = translate(translate(s2, mt_en_de_tokenizer, mt_en_de_model),
                      mt_de_en_tokenizer, mt_de_en_model)
    return s1_bt, s2_bt

def mlm_paraphrase_pair(s1: str, s2: str, mask_ratio=0.15):
    """
    Mask and fill tokens using a fill-mask pipeline to paraphrase.
    """
    def mlm_augment(sentence, ratio):
        words = sentence.split()
        n_masks = max(1, int(len(words) * ratio))
        masked = words.copy()
        mask_indices = random.sample(range(len(words)), n_masks)
        for idx in mask_indices:
            masked[idx] = mlm_fill.tokenizer.mask_token
        masked_text = ' '.join(masked)
        for _ in mask_indices:
            try:
                preds = mlm_fill(masked_text)
                masked_text = preds[0]['sequence']
            except Exception:
                break
        return masked_text

    return mlm_augment(s1, mask_ratio), mlm_augment(s2, mask_ratio)