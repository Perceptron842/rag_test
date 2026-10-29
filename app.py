"""CLI: build day3-compatible Chroma index and compare BM25 / vector / hybrid / rerank."""
import argparse
import time
from functools import lru_cache
from pathlib import Path
from core import BM25, rrf_fuse, rerank_pairs
from retriever import (
    COLLECTION_NAME, DEFAULT_DB, MODEL_NAME, build_index, get_client, get_encoder,
)

DEFAULT_RERANKER = 'cross-encoder/mmarco-mMiniLMv2-L12-H384-v1'

# 获取cross-encoder rerank 来重新判断问题与每个候选文档的相关性
@lru_cache(maxsize=2)
def get_reranker(name):
    from sentence_transformers import CrossEncoder
    return CrossEncoder(name, device='cpu')

# 分批读取 Chroma 集合中的全部文档，将每条文档的 ID、正文和元数据整理成字典，最后返回一个列表
# collection是Chroma中的一个数据集合
def fetch_documents(collection, page_size=500):
    """Read exactly the same Chroma collection for BM25 and vector search."""
    docs = []
    n = collection.count()
    for offset in range(0, n, page_size):
        rows = collection.get(limit=page_size, offset=offset,
                              include=['documents', 'metadatas'])
        for doc_id, text, meta in zip(rows['ids'], rows['documents'], rows['metadatas']):
            docs.append({'id': doc_id, 'text': text, 'metadata': meta or {}})
    return docs


class SearchEngine:
    # 创建一个检索引擎，连接已有的 Chroma 集合，检查索引使用的模型，再读取文档并初始化 BM25 检索器
    def __init__(self, db=DEFAULT_DB, model=MODEL_NAME):
        # 获取chroma客户端
        client = get_client(db)
        try:
            # 获取chroma中的collection集合
            self.collection = client.get_collection(
                COLLECTION_NAME, 
                embedding_function=None,      # 不为这个集合绑定自动计算文本向量的函数。你的项目会自行计算查询向量，再传给 Chroma 执行检索
            )
        except Exception as exc:
            # 未先在chroma中构建索引的异常
            raise RuntimeError('No Chroma index: run python app.py index first') from exc
        self.model = model
        # 读取构建索引时记录的模型名称
        indexed_model = (self.collection.metadata or {}).get('model_name')
        # 构建文档索引的模型 == 当前处理用户问题的模型
        if indexed_model != model:
            raise ValueError(f'Index embedding model is {indexed_model!r}; use {indexed_model!r} or rebuild')
        self.documents = fetch_documents(self.collection)
        if not self.documents:
            raise ValueError('Collection is empty. Rebuild index with non-empty chunks.')
        self.bm25 = BM25(self.documents)

    def vector(self, question: str, top_k: int):
        encoder = get_encoder(self.model)
        # 将问题编码为向量
        qvec = encoder.encode([question], normalize_embeddings=True, convert_to_numpy=True)
        # 使用余弦相似度进行向量检索查询
        results = self.collection.query(query_embeddings=qvec.tolist(),
                                        n_results=min(top_k, len(self.documents)),
                                        include=['documents', 'metadatas', 'distances'])
        return [
            {'id': doc_id, 'text': text, 'metadata': meta or {},
             'cosine_similarity': 1.0 - float(distance)}
            for doc_id, text, meta, distance in zip(
                results['ids'][0], results['documents'][0],
                results['metadatas'][0], results['distances'][0])
        ]

    def search(self, question: str, mode='hybrid', top_k=5, candidate_k=10,
               rerank_model=DEFAULT_RERANKER):
        if not question.strip() or top_k <= 0 or candidate_k < top_k:
            raise ValueError('Question cannot be empty; require 0 < top_k <= candidate_k')
        if mode == 'bm25':
            return self.bm25.search(question, top_k)
        if mode == 'vector':
            return self.vector(question, top_k)
        vector_hits = self.vector(question, candidate_k)
        bm25_hits = self.bm25.search(question, candidate_k)
        # Fusion uses stable Chroma IDs, never merge incomparable raw BM25/cosine scores.
        fused = rrf_fuse([vector_hits, bm25_hits], top_k=candidate_k)
        if mode == 'hybrid':
            return fused[:top_k]
        if mode == 'rerank':
            return rerank_pairs(question, fused, get_reranker(rerank_model))[:top_k]
        raise ValueError(f'Unknown mode: {mode}')


def main():
    # 先定义程序能接收哪些命令     argparse是用来解析终端参数的模块
    parser = argparse.ArgumentParser(description='Day 4 hybrid retrieval')
    subs = parser.add_subparsers(dest='action', required=True)

    # 把数据建立成可检索的索引
    idx = subs.add_parser('index', help='Build the same Chroma index as Day 3')
    idx.add_argument('--input', default='data/sample_chunks.json')
    idx.add_argument('--db', default=DEFAULT_DB)
    idx.add_argument('--reset', action='store_true')
    idx.add_argument('--model', default=MODEL_NAME)

    # 定义搜索时的参数
    ask = subs.add_parser('search', help='Retrieve and compare rankings')
    ask.add_argument('--query', required=True)
    ask.add_argument('--db', default=DEFAULT_DB)
    ask.add_argument('--model', default=MODEL_NAME)
    ask.add_argument('--mode', choices=['bm25', 'vector', 'hybrid', 'rerank'], default='hybrid')
    ask.add_argument('--top-k', type=int, default=5)
    ask.add_argument('--candidate-k', type=int, default=10)
    ask.add_argument('--rerank-model', default=DEFAULT_RERANKER)

    # 把终端命令变成Python变量
    args = parser.parse_args()
    if args.action == 'index':
        build_index(args.input, db_path=args.db, reset=args.reset, model_name=args.model)
    else:
        started = time.perf_counter()
        engine = SearchEngine(db=args.db, model=args.model)
        hits = engine.search(args.query, args.mode, args.top_k,
                             args.candidate_k, args.rerank_model)
        print(f'Mode={args.mode}, results={len(hits)}, elapsed={time.perf_counter()-started:.2f}s')
        for rank, hit in enumerate(hits, 1):
            meta = hit.get('metadata') or {}
            score = next((f'{key}={hit[key]:.4f}' for key in ('rerank_score','rrf_score','bm25_score','cosine_similarity') if key in hit), '')
            print(f'#{rank} {score} source={meta.get("source")} page={meta.get("page")} id={hit["id"][:10]}')
            print(' ', hit['text'][:220].replace('\n',' '))


if __name__ == '__main__':
    main()
