"""Day 4 learning targets: tokenizer, BM25, RRF, and reranking.

No LLM calls. No dependencies required for English-token offline unit tests.
Install jieba for better Chinese segmentation; fallback is character n-grams.
"""
from __future__ import annotations
import math
import re
from collections import Counter

_PATTERN = re.compile(r"[a-zA-Z][a-zA-Z0-9_+.\-]*|\d+(?:\.\d+)?|[\u4e00-\u9fff]+")


def tokenize(text: str) -> list[str]:
    """Use same tokenizer for corpus AND queries.

    For Chinese, prefer jieba; otherwise (offline) use overlapping 1/2-grams.
    """
    tokens = []
    for part in _PATTERN.findall(text.lower()):
        if re.fullmatch(r"[\u4e00-\u9fff]+", part):
            try:
                import jieba
            except ImportError:
                tokens.extend(list(part))
                tokens.extend(part[i:i+2] for i in range(len(part)-1))
            else:
                tokens.extend(w for w in jieba.lcut(part) if w.strip())
        else:
            tokens.append(part)
    return tokens


class BM25:
    """Educational Okapi BM25 (self-contained; small-corpus use only)."""

    def __init__(self, docs: list[dict], k1: float = 1.5, b: float = 0.75):
        if k1 <= 0 or not 0 <= b <= 1:
            raise ValueError("k1 must be positive and b in [0, 1]")
        self.docs = docs
        self.k1, self.b = k1, b
        self.terms = [Counter(tokenize(row['text'])) for row in docs]
        self.lengths = [sum(freq.values()) for freq in self.terms]
        self.avgdl = sum(self.lengths) / len(docs) if docs else 0.0
        df = Counter(term for bag in self.terms for term in bag)
        n = len(docs)
        self.idf = {term: math.log(1 + (n - count + 0.5)/(count + 0.5))
                    for term, count in df.items()}

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        if top_k < 1:
            raise ValueError("top_k must be positive")
        terms = set(tokenize(query))
        results = []
        for doc, bag, dl in zip(self.docs, self.terms, self.lengths):
            score = 0.0
            for term in terms:
                tf = bag.get(term, 0)
                if tf:
                    denom = tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                    score += self.idf[term] * tf * (self.k1 + 1) / denom
            if score > 0:
                results.append({**doc, "bm25_score": float(score)})
        return sorted(results, key=lambda d: (-d['bm25_score'], d['id']))[:top_k]


def rrf_fuse(ranked_lists: list[list[dict]], top_k: int = 5, k: int = 60) -> list[dict]:
    """Reciprocal Rank Fusion: sum(1/(k+rank)), rank starts at 1.

    IDs, not Python object identity or list indexes, provide stable deduplication.
    """
    if k < 0 or top_k <= 0:
        raise ValueError("k must be nonnegative and top_k positive")
    combined: dict[str, dict] = {}
    for side, ranking in enumerate(ranked_lists):
        seen_in_side = set()
        for rank, hit in enumerate(ranking, start=1):
            doc_id = hit['id']
            if doc_id in seen_in_side:
                continue
            seen_in_side.add(doc_id)
            if doc_id not in combined:
                combined[doc_id] = {**hit, 'rrf_score': 0.0, 'sources': []}
            combined[doc_id]['rrf_score'] += 1.0 / (k + rank)
            combined[doc_id]['sources'].append(side)
    return sorted(combined.values(), key=lambda x: (-x['rrf_score'], x['id']))[:top_k]


def rerank_pairs(query: str, hits: list[dict], model) -> list[dict]:
    """Cross-Encoder: score each (query, document) pair; higher is better.

    `model` is injectable for testing, or a sentence_transformers.CrossEncoder.
    """
    if not hits:
        return []
    pairs = [(query, hit['text']) for hit in hits]
    scores = model.predict(pairs)
    if len(scores) != len(hits):
        raise RuntimeError("Reranker returned an unexpected number of scores")
    rows = [{**hit, 'rerank_score': float(score)} for hit, score in zip(hits, scores)]
    return sorted(rows, key=lambda row: (-row['rerank_score'], row['id']))


def hit_and_mrr(results: list[dict], relevant_ids: set[str], k: int = 5) -> tuple[float, float]:
    """Single-query Hit@K and reciprocal rank; IDs label true relevance."""
    if k <= 0:
        raise ValueError("k must be positive")
    for rank, hit in enumerate(results[:k], start=1):
        if hit['id'] in relevant_ids:
            return 1.0, 1.0 / rank
    return 0.0, 0.0
