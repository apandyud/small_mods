"""
Question clustering (text-only, English questions)
--------------------------------------------------
Two embedding backends:
  1) TF-IDF (word + character n-grams)  — no extra deps, strong baseline
  2) Sentence-BERT (if installed)       — better semantics; auto fallback if missing

Clustering methods:
  - Agglomerative with cosine distance + auto threshold search (silhouette)
  - Optional HDBSCAN if installed (set method="hdbscan")

Usage (minimal):
  python question_clustering_text_only.py --in questions.txt --out clusters.csv

Each line in questions.txt is one question.

You can also pass questions via Python API (see bottom for example).
"""

from __future__ import annotations
import argparse
import csv
import json
import math
import os
import re
import sys
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_distances
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA

# Optional deps
try:
    import hdbscan  # type: ignore
except Exception:
    hdbscan = None

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None  # type: ignore


# ---------------------------
# Text cleaning / preproc
# ---------------------------
_punct_norm = str.maketrans({
    "\u2013": "-",  # en dash
    "\u2014": "-",  # em dash
    "\u2019": "'",  # right single quote
    "\u2018": "'",
    "\u201c": '"',
    "\u201d": '"',
})

NUM_TOKEN = "<NUM>"
PCT_TOKEN = "<PCT>"

_num_re = re.compile(r"(?<!\w)(?:\d+[\d,]*\.?\d*)(?!\w)")
_pct_re = re.compile(r"(\d+[\d,]*\.?\d*)\s*%")


def clean_question(q: str) -> str:
    """Lightweight normalization suitable for English numeric QA questions."""
    s = q.strip().translate(_punct_norm)
    s = s.lower()
    # Normalize percentages before numbers so % doesn't double-replace
    s = _pct_re.sub(PCT_TOKEN, s)
    s = _num_re.sub(NUM_TOKEN, s)
    # Collapse extra whitespace
    s = re.sub(r"\s+", " ", s)
    return s


# ---------------------------
# Embedding backends
# ---------------------------
@dataclass
class EmbeddingConfig:
    backend: str = "tfidf"  # or "sbert"
    sbert_model: str = "all-MiniLM-L6-v2"


def embed_texts(texts: List[str], cfg: EmbeddingConfig) -> np.ndarray:
    if cfg.backend == "sbert":
        if SentenceTransformer is None:
            print("[warn] sentence-transformers not installed; falling back to TF-IDF.")
        else:
            model = SentenceTransformer(cfg.sbert_model)
            emb = model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
            return np.asarray(emb)
    # TF-IDF baseline (word + char n-grams)
    vect = TfidfVectorizer(
        preprocessor=None,
        tokenizer=None,
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=2,
        max_df=0.95,
        lowercase=False,
    )
    X_char = vect.fit_transform(texts)
    # Word-level TF-IDF to complement char-level
    vect_w = TfidfVectorizer(
        ngram_range=(1, 2),
        stop_words="english",
        min_df=2,
        max_df=0.95,
        lowercase=False,
    )
    X_word = vect_w.fit_transform(texts)
    # Concatenate sparse matrices
    from scipy.sparse import hstack

    X = hstack([X_char, X_word]).tocsr()
    # Convert to dense only for algorithms that need it; Agglo works with distances
    # We'll return a dense array of L2-normalized rows for safety.
    # NOTE: For large datasets, keep as sparse and compute distances on-the-fly.
    X_dense = X.astype(np.float32)
    # Row-normalize
    norms = np.sqrt(X_dense.multiply(X_dense).sum(axis=1)).A1 + 1e-12
    X_dense = X_dense.multiply(1.0 / norms[:, None]).toarray()
    return X_dense


# ---------------------------
# Clustering
# ---------------------------
@dataclass
class ClusterConfig:
    method: str = "agglomerative"  # "agglomerative" | "hdbscan"
    min_cluster_size: int = 8  # used by HDBSCAN or to bound threshold search
    max_k_guess: int = 30  # for threshold search upper bound heuristic


def cluster_agglomerative(X: np.ndarray, min_cluster_size: int = 8) -> Tuple[np.ndarray, float]:
    """Cosine-distance Agglomerative with automatic distance threshold via silhouette.
    Returns (labels, best_threshold).
    """
    D = cosine_distances(X)
    np.fill_diagonal(D, 0.0)

    # Search thresholds that yield between 2 and ~N/min_cluster_size clusters
    n = len(X)
    max_clusters = max(2, n // max(2, min_cluster_size))
    # Candidate thresholds: quantiles of pairwise distances
    tri = D[np.triu_indices(n, k=1)]
    qs = np.linspace(0.3, 0.9, 13)  # conservative range for cosine distances
    cands = np.quantile(tri, qs)

    best_s = -1.0
    best_labels = np.full(n, -1, dtype=int)
    best_thr = float(cands[0])

    for thr in cands:
        # Fit with distance_threshold → n_clusters=None triggers tree cut by threshold
        model = AgglomerativeClustering(
            affinity="precomputed",
            linkage="average",
            distance_threshold=thr,
            n_clusters=None,
        )
        labels = model.fit_predict(D)
        k = len(set(labels)) - (1 if -1 in labels else 0)
        if k < 2 or k > max_clusters:
            continue
        try:
            s = silhouette_score(1 - D, labels, metric="cosine")
        except Exception:
            continue
        if s > best_s:
            best_s, best_labels, best_thr = s, labels, float(thr)

    # Fallback: force 2 clusters if everything failed
    if best_s < 0:
        model = AgglomerativeClustering(affinity="precomputed", linkage="average", n_clusters=2)
        best_labels = model.fit_predict(D)
        best_thr = float(np.median(tri))

    return best_labels, best_thr


def cluster_hdbscan(X: np.ndarray, min_cluster_size: int = 8) -> Tuple[np.ndarray, Optional[float]]:
    if hdbscan is None:
        print("[warn] hdbscan not installed; falling back to agglomerative.")
        return cluster_agglomerative(X, min_cluster_size)
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=None, metric="euclidean")
    labels = clusterer.fit_predict(X)
    return labels, None


# ---------------------------
# Public API
# ---------------------------
@dataclass
class Result:
    labels: np.ndarray
    embedding_backend: str
    cluster_method: str
    extra: dict


def cluster_questions(
    questions: List[str],
    embedding_backend: str = "tfidf",
    cluster_method: str = "agglomerative",
    min_cluster_size: int = 8,
) -> Result:
    if not questions:
        raise ValueError("Empty question list.")
    cleaned = [clean_question(q) for q in questions]
    X = embed_texts(cleaned, EmbeddingConfig(backend=embedding_backend))
    if cluster_method == "hdbscan":
        labels, thr = cluster_hdbscan(X, min_cluster_size=min_cluster_size)
    else:
        labels, thr = cluster_agglomerative(X, min_cluster_size=min_cluster_size)

    return Result(
        labels=np.asarray(labels),
        embedding_backend=embedding_backend,
        cluster_method=cluster_method,
        extra={"threshold": thr},
    )


# ---------------------------
# I/O helpers
# ---------------------------

def read_lines(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [ln.rstrip("\n") for ln in f if ln.strip()]


def write_clusters_csv(path: str, questions: List[str], labels: np.ndarray) -> None:
    rows = [(int(lbl), q) for q, lbl in zip(questions, labels)]
    rows.sort(key=lambda x: (x[0], x[1]))
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["cluster", "question"])
        for lbl, q in rows:
            w.writerow([lbl, q])


def preview_clusters(questions: List[str], labels: np.ndarray, max_examples: int = 5) -> str:
    from collections import defaultdict
    bucket = defaultdict(list)
    for q, l in zip(questions, labels):
        bucket[int(l)].append(q)
    parts = []
    for k in sorted(bucket.keys()):
        ex = "\n".join("  - " + s for s in bucket[k][:max_examples])
        parts.append(f"Cluster {k} (n={len(bucket[k])}):\n{ex}")
    return "\n\n".join(parts)


# ---------------------------
# CLI
# ---------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="Input text file: one question per line")
    ap.add_argument("--out", dest="out", default="clusters.csv", help="Output CSV path")
    ap.add_argument("--backend", choices=["tfidf", "sbert"], default="tfidf")
    ap.add_argument("--method", choices=["agglomerative", "hdbscan"], default="agglomerative")
    ap.add_argument("--min_cluster_size", type=int, default=8)
    args = ap.parse_args()

    questions = read_lines(args.inp)
    res = cluster_questions(
        questions,
        embedding_backend=args.backend,
        cluster_method=args.method,
        min_cluster_size=args.min_cluster_size,
    )
    write_clusters_csv(args.out, questions, res.labels)

    # Console preview
    print(f"Embedding backend: {res.embedding_backend}")
    print(f"Clustering method: {res.cluster_method}")
    if res.extra.get("threshold") is not None:
        print(f"Chosen distance threshold: {res.extra['threshold']:.4f}")
    print()
    print(preview_clusters(questions, res.labels))


if __name__ == "__main__":
    main()

# ---------------------------
# Example (Python API)
# ---------------------------
# questions = [
#     "What is the percentage increase from 80 to 92?",
#     "What percent of 250 is 40?",
#     "By how much did sales grow year-over-year?",
#     "What is 17 + 24?",
#     "Compute the ratio of boys to girls if there are 12 boys and 8 girls.",
#     "How many percentage points is 12% vs 9%?",
# ]
# res = cluster_questions(questions, embedding_backend="tfidf", cluster_method="agglomerative")
# print(preview_clusters(questions, res.labels))
