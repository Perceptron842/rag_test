"""Evaluate Day5 SearchEngine with 20 curated question-topic labels.

Gold topics are manually assigned to this fixed 12-chunk teaching corpus.
Topic -> real Chroma IDs is resolved from the indexed documents, never guessed.
For a new corpus, write NEW relevance labels rather than reusing old topics.
"""
from __future__ import annotations
import argparse
import csv
import json
import time
from pathlib import Path
from metrics import score_query, summarize


def load_gold(path: str):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list) or not data:
        raise ValueError("Gold data must be a nonempty list")
    return data


def labels_from_documents(documents: list[dict]) -> dict[str, set[str]]:
    topic_index: dict[str, set[str]] = {}
    for doc in documents:
        topic = (doc.get("metadata") or {}).get("topic")
        if topic:
            topic_index.setdefault(str(topic), set()).add(str(doc["id"]))
    return topic_index


def run(engine, questions, modes: list[str], k: int = 5, candidate_k: int = 10):
    topic_index = labels_from_documents(engine.documents)
    rows = []
    for case in questions:
        relevant_ids = set()
        for topic in case["relevant_topics"]:
            ids = topic_index.get(topic)
            if not ids:
                raise ValueError(f"Topic {topic!r} not indexed; do not invent gold labels")
            relevant_ids.update(ids)
        for mode in modes:
            start = time.perf_counter()
            results = engine.search(case["question"], mode=mode, top_k=k, candidate_k=candidate_k)
            elapsed_ms = round(1000 * (time.perf_counter() - start), 2)
            ids = [str(doc["id"]) for doc in results]
            score = score_query(ids, relevant_ids, k)
            rows.append({"qid":case["id"],"question":case["question"],"mode":mode,
                         "k":k,"retrieved_ids":"|".join(ids),
                         "relevant_ids":"|".join(sorted(relevant_ids)),
                         "elapsed_ms":elapsed_ms,**score})
    summaries={}
    for mode in modes:
        this_mode=[row for row in rows if row["mode"]==mode]
        summaries[mode] = summarize(this_mode)
        summaries[mode]["avg_search_ms"] = round(sum(r["elapsed_ms"] for r in this_mode)/len(this_mode),2)
    return rows,summaries


def main():
    p=argparse.ArgumentParser(description="Day 6 retrieval evaluation")
    p.add_argument("--gold", default="data/gold_questions.json")
    p.add_argument("--db", default="chroma_db")
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--candidate-k", type=int, default=10)
    p.add_argument("--modes", nargs="+", choices=["vector","bm25","hybrid","rerank"],
                   default=["vector","bm25","hybrid"])
    p.add_argument("--out", default="outputs")
    args=p.parse_args()
    from app import SearchEngine
    engine=SearchEngine(db=args.db)
    rows,summary=run(engine,load_gold(args.gold),args.modes,args.top_k,args.candidate_k)
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    with (out/'retrieval_per_question.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0].keys())); writer.writeheader();writer.writerows(rows)
    (out/'retrieval_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))
    print(f"Saved to {out.resolve()}")


if __name__=='__main__':
    main()
