"""Day 6 core: pure Python retrieval metrics; no model downloads required."""
from __future__ import annotations


def score_query(retrieved_ids: list[str], relevant_ids: set[str], k: int = 5) -> dict:
    """Recall@k, Hit@k, reciprocal rank; all relevant IDs must be labeled.

    Reject empty gold relevance sets: unanswerable questions have a separate test.
    """
    if k < 1 or not relevant_ids:
        raise ValueError("k must be positive and relevant_ids nonempty")
    if len(retrieved_ids) != len(set(retrieved_ids)):
        raise ValueError("retrieved IDs must be unique")
    top = retrieved_ids[:k]
    matched = set(top) & relevant_ids
    reciprocal_rank = 0.0
    for rank, doc_id in enumerate(top, start=1):
        if doc_id in relevant_ids:
            reciprocal_rank = 1.0 / rank
            break
    return {
        "recall": len(matched) / len(relevant_ids),
        "hit": float(bool(matched)),
        "rr": reciprocal_rank,
    }


def summarize(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError("No evaluation rows")
    return {
        "questions": len(rows),
        "recall_at_k": sum(r["recall"] for r in rows) / len(rows),
        "hit_at_k": sum(r["hit"] for r in rows) / len(rows),
        "mrr_at_k": sum(r["rr"] for r in rows) / len(rows),
    }
