#!/usr/bin/env python3
"""
TF-IDF Text Processing Pipeline

Author: Cyril Zhang

This script implements an end-to-end NLP preprocessing pipeline and
computes TF-IDF scores from scratch for a collection of documents.

Pipeline:
1. Text cleaning and normalization
2. Stopword removal
3. Rule-based stemming
4. TF computation
5. IDF computation
6. TF-IDF scoring and ranking
"""

import re
import math
from pathlib import Path
from collections import Counter
from decimal import Decimal, ROUND_HALF_UP

# ===============================
# Configuration
DOC_LIST_FILE = "tfidf_docs.txt"
STOPWORDS_FILE = "stopwords.txt"

# ===============================
# Text Preprocessing Functions

def clean_text(text: str) -> str:
    """Remove links, non-word characters, extra spaces, and lowercase text."""
    text = re.sub(r"https?://\S+", "", text)          # remove URLs
    text = re.sub(r"[^\w\s]", " ", text)              # remove non-word chars
    text = re.sub(r"\s+", " ", text).strip()          # normalize spaces
    return text.lower()

def load_stopwords(path: Path) -> set:
    return set(line.strip() for line in path.open() if line.strip())

def remove_stopwords(tokens, stopwords):
    return [t for t in tokens if t not in stopwords]

def stem_token(token: str) -> str:
    """Apply simple rule-based stemming."""
    if token.endswith("ing"):
        return token[:-3]
    if token.endswith("ly"):
        return token[:-2]
    if token.endswith("ment"):
        return token[:-4]
    return token

def preprocess_document(text: str, stopwords: set) -> list:
    cleaned = clean_text(text)
    tokens = cleaned.split(" ")
    tokens = remove_stopwords(tokens, stopwords)
    tokens = [stem_token(t) for t in tokens if t]
    return tokens

# ===============================
# TF-IDF Functions

def compute_tf(tokens: list) -> dict:
    total_terms = len(tokens)
    counts = Counter(tokens)
    return {term: count / total_terms for term, count in counts.items()}

def compute_idf(documents: list) -> dict:
    total_docs = len(documents)
    doc_freq = Counter()

    for tokens in documents:
        for term in set(tokens):
            doc_freq[term] += 1

    return {
        term: math.log(total_docs / doc_freq[term]) + 1
        for term in doc_freq
    }

def compute_tfidf(documents: list) -> list:
    idf = compute_idf(documents)
    tfidf_results = []

    for tokens in documents:
        tf = compute_tf(tokens)
        tfidf = {
            term: float(
                Decimal(tf[term] * idf[term]).quantize(
                    Decimal("0.01"), rounding=ROUND_HALF_UP
                )
            )
            for term in tf
        }
        tfidf_results.append(tfidf)

    return tfidf_results

def top_k_terms(tfidf_map: dict, k=5):
    return sorted(
        tfidf_map.items(),
        key=lambda x: (-x[1], x[0])
    )[:k]

# ===============================
# File Utilities

def read_lines(path: Path):
    return [line.strip() for line in path.open() if line.strip()]

def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")

def write_text(path: Path, content: str):
    path.write_text(content, encoding="utf-8")

def format_output(pairs):
    return str([(word, score) for word, score in pairs])

# ===============================
# Main Execution

def main():
    base = Path(".")
    stopwords = load_stopwords(base / STOPWORDS_FILE)
    doc_names = read_lines(base / DOC_LIST_FILE)

    all_documents = []

    # Preprocessing
    for name in doc_names:
        text = read_text(base / name)
        tokens = preprocess_document(text, stopwords)
        write_text(base / f"preproc_{name}", " ".join(tokens))
        all_documents.append(tokens)

    # TF-IDF
    tfidf_results = compute_tfidf(all_documents)

    for name, tfidf in zip(doc_names, tfidf_results):
        top5 = top_k_terms(tfidf)
        write_text(base / f"tfidf_{name}", format_output(top5))

if __name__ == "__main__":
    main()
