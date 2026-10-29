def rrf_fusion(result_lists, k=60):
    scores = {}

    for result in result_lists:
        for rank, doc_id in enumerate(result, start=1):
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank) 


    return sorted(
        scores,
        key = lambda doc_id : scores[doc_id],
        reverse=True
    ) 


vector = ["A", "B", "C"]
bm25 = ["B", "C", "D"]

print(rrf_fusion([vector, bm25]))