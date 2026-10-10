"""Text cleaning used before TF-IDF.

Design note: we keep tech tokens like c++, c#, node.js, scikit-learn intact,
because generic cleaning would turn "c++" into "c" and destroy the signal.
"""
from __future__ import annotations

import re
from typing import List

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

URL_RE = re.compile(r"https?://\S+|www\.\S+")
EMAIL_RE = re.compile(r"\S+@\S+\.\S+")
PHONE_RE = re.compile(r"\+?\d[\d\s().-]{8,}\d")
# a token is letters/digits, optionally joined by . + # - (c++, c#, node.js, scikit-learn)
TOKEN_RE = re.compile(r"[a-z0-9]+(?:[.+#\-][a-z0-9]+)*[+#]*")


def simple_lemma(token: str) -> str:
    """Very small rule-based lemmatizer (plural / -ing / -ed endings).

    Protects tokens containing . + # - digits so tech terms are untouched.
    """
    if any(ch in token for ch in ".+#-") or any(ch.isdigit() for ch in token):
        return token
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 4 and token.endswith("sses"):
        return token[:-2]
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    if len(token) > 5 and token.endswith("ing"):
        return token[:-3]
    if len(token) > 4 and token.endswith("ed"):
        return token[:-2]
    return token


def clean_text(text: str) -> str:
    """Lowercase and strip URLs, emails and phone numbers."""
    text = text.lower()
    text = URL_RE.sub(" ", text)
    text = EMAIL_RE.sub(" ", text)
    text = PHONE_RE.sub(" ", text)
    return text


def tokenize(text: str) -> List[str]:
    """Clean, tokenize, drop stopwords and lemmatize."""
    tokens = TOKEN_RE.findall(clean_text(text))
    return [simple_lemma(t) for t in tokens if t not in ENGLISH_STOP_WORDS and len(t) > 1]


def preprocess(text: str) -> str:
    """Return the cleaned text as one space-joined string (TF-IDF input)."""
    return " ".join(tokenize(text))
