"""Minimal Chinese semantic search with external embeddings and persistent Chroma.

Run from project root:
    python retriever.py index --input data/sample_chunks.json --reset
    python retriever.py search --query "如何用键查找字典的值？" --top-k 5

Requires: pip install -r requirements.txt
The first embedding model download needs network access.
"""
import argparse
import hashlib
import json
from functools import lru_cache
from pathlib import Path

MODEL_NAME = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
COLLECTION_NAME = 'day3_rag_chunks'
DEFAULT_DB = 'chroma_db'


def load_chunks(path):
    """Read Day 2 JSON schema (text/source/page) or our sample schema."""
    raw = json.loads(Path(path).read_text(encoding='utf-8'))
    if not isinstance(raw, list):
        raise ValueError('Expected a JSON array of chunks')
    chunks = []
    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f'Chunk {i} must be an object')
        text = item.get('text') or item.get('page_content')
        if not isinstance(text, str) or not text.strip():
            continue
        meta = item.get('metadata') if isinstance(item.get('metadata'), dict) else {}
        source = str(item.get('source') or meta.get('source') or 'unknown')
        page = item.get('page', meta.get('page'))
        index = item.get('chunk_index', meta.get('chunk_index', i))
        topic = str(item.get('topic') or meta.get('topic') or '')
        page = -1 if page is None else int(page)
        index = int(index)
        identity = f'{source}|{page}|{index}|{text}|{i}'
        chunk_id = hashlib.sha256(identity.encode('utf-8')).hexdigest()[:24]
        chunks.append({
            'id': chunk_id,
            'text': text,
            'metadata': {
                'source': source,
                'page': page,
                'chunk_index': index,
                'topic': topic,
            },
        })
    if not chunks:
        raise ValueError('No non-empty chunks found')
    return chunks


@lru_cache(maxsize=2)
def get_encoder(model_name=MODEL_NAME):
    # Delayed import so offline math/tests work without installing model libraries.
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name, device='cpu')


@lru_cache(maxsize=4)
def get_client(db_path):
    import chromadb
    return chromadb.PersistentClient(path=str(Path(db_path).resolve()))


def build_index(input_path, db_path=DEFAULT_DB, reset=False, model_name=MODEL_NAME):
    chunks = load_chunks(input_path)
    client = get_client(db_path)
    if reset:
        from chromadb.errors import NotFoundError

        try:
            client.delete_collection(COLLECTION_NAME)
        except NotFoundError:
            # A fresh database has nothing to delete; continue creating it.
            pass
    existing_names = {c.name if hasattr(c, 'name') else str(c)
                      for c in client.list_collections()}
    if COLLECTION_NAME in existing_names:
        collection = client.get_collection(COLLECTION_NAME, embedding_function=None)
        prior = (collection.metadata or {}).get('model_name')
        if prior != model_name:
            raise ValueError(f'Existing DB model is {prior!r}; use --reset to rebuild')
    else:
        collection = client.create_collection(
            name=COLLECTION_NAME,
            metadata={'hnsw:space': 'cosine', 'model_name': model_name},
            embedding_function=None,
        )
    encoder = get_encoder(model_name)
    texts = [chunk['text'] for chunk in chunks]
    vectors = encoder.encode(texts, batch_size=8, normalize_embeddings=True,
                             convert_to_numpy=True, show_progress_bar=True)
    print('Embedding shape:', tuple(vectors.shape))
    # Keep chunks small; large datasets require batching on both encode & upsert.
    for start in range(0, len(chunks), 64):
        batch = chunks[start:start + 64]
        collection.upsert(
            ids=[c['id'] for c in batch],
            embeddings=vectors[start:start + 64].tolist(),
            documents=[c['text'] for c in batch],
            metadatas=[c['metadata'] for c in batch],
        )
    print(f'Index ready: {collection.count()} records; database: {Path(db_path).resolve()}')


def search(query, top_k=5, db_path=DEFAULT_DB, model_name=MODEL_NAME):
    if not query.strip():
        raise ValueError('Query must not be empty')
    if top_k <= 0:
        raise ValueError('top_k must be greater than zero')
    client = get_client(db_path)
    collection = client.get_collection(COLLECTION_NAME, embedding_function=None)
    prior = (collection.metadata or {}).get('model_name')
    if prior != model_name:
        raise ValueError(f'Index used {prior!r}; use the same embedding model')
    count = collection.count()
    if count == 0:
        return []
    encoder = get_encoder(model_name)
    qvec = encoder.encode([query], normalize_embeddings=True, convert_to_numpy=True)
    results = collection.query(
        query_embeddings=qvec.tolist(),
        n_results=min(top_k, count),
        include=['documents', 'metadatas', 'distances'],
    )
    rows = []
    for i, text in enumerate(results['documents'][0]):
        metadata = results['metadatas'][0][i] or {}
        distance = float(results['distances'][0][i])
        rows.append({
            'rank': i + 1,
            'similarity': 1 - distance,  # collection was created with cosine distance
            'distance': distance,
            'text': text,
            'source': metadata.get('source'),
            'page': metadata.get('page'),
            'chunk_index': metadata.get('chunk_index'),
            'topic': metadata.get('topic'),
        })
    return rows


def main():
    parser = argparse.ArgumentParser(description='Day 3 local vector search')
    sub = parser.add_subparsers(dest='action', required=True)
    idx = sub.add_parser('index')
    idx.add_argument('--input', default='data/sample_chunks.json')
    idx.add_argument('--reset', action='store_true', help='Delete/rebuild existing collection')
    idx.add_argument('--db', default=DEFAULT_DB)
    idx.add_argument('--model', default=MODEL_NAME)
    ask = sub.add_parser('search')
    ask.add_argument('--query', required=True)
    ask.add_argument('--top-k', type=int, default=5)
    ask.add_argument('--db', default=DEFAULT_DB)
    ask.add_argument('--model', default=MODEL_NAME)
    args = parser.parse_args()
    if args.action == 'index':
        build_index(args.input, args.db, args.reset, args.model)
    else:
        hits = search(args.query, args.top_k, args.db, args.model)
        for hit in hits:
            page = '-' if hit['page'] in (None, -1) else hit['page']
            print(f"\n#{hit['rank']} cosine={hit['similarity']:.4f} "
                  f"source={hit['source']} page={page} chunk={hit['chunk_index']}")
            print(hit['text'][:300])


if __name__ == '__main__':
    main()
