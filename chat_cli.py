"""Day 5 command line RAG: index -> ask -> optional interactive chat.

Examples:
 python chat_cli.py index --input data/sample_chunks.json --reset
 python chat_cli.py ask --question 'Python字典怎样按键查值？' --mock
 python chat_cli.py chat
"""
import argparse
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from rag_pipeline import MockGenerator, OpenAIGenerator, run_rag
from app import SearchEngine
from retriever import build_index, DEFAULT_DB, MODEL_NAME

# This local CLI uses the project's provider settings, including its API key.
load_dotenv(Path(__file__).with_name('.env'), override=True)


def make_generator(mock):
    if mock:
        return MockGenerator()
    return OpenAIGenerator(
        model=os.getenv('OPENAI_MODEL', ''),
        api_key=os.getenv('OPENAI_API_KEY', ''),
        base_url=os.getenv('OPENAI_BASE_URL') or None,
    )


def ask_once(question, args, engine, generator):
    result = run_rag(
        question, engine, generator, mode=args.mode,
        top_k=args.top_k, candidate_k=args.candidate_k,
        min_cosine=None if args.no_gate else args.min_cosine,
    )
    print('\n【回答】\n' + result.answer)
    if result.sources:
        print('\n【检索来源（不代表自动核验正确性）】')
        for source in result.sources:
            print(f"[{source['number']}] {source['source']} | {source['excerpt'][:80]}")
    if args.show_prompt and result.evidence:
        print('\n【发送给模型的证据片段】\n' + result.evidence)
    print('\n状态：', result.status)


def main():
    p = argparse.ArgumentParser(description='Day 5: retrieval-augmented generation')
    sub = p.add_subparsers(dest='action', required=True)
    index = sub.add_parser('index', help='Build Day 3/4 compatible Chroma index')
    index.add_argument('--input', default='data/sample_chunks.json')
    index.add_argument('--db', default=DEFAULT_DB)
    index.add_argument('--model', default=MODEL_NAME)
    index.add_argument('--reset', action='store_true')
    for action in ('ask', 'chat'):
        ap = sub.add_parser(action)
        ap.add_argument('--db', default=DEFAULT_DB)
        ap.add_argument('--model', default=MODEL_NAME)
        ap.add_argument('--mode', choices=('vector', 'bm25', 'hybrid', 'rerank'), default='hybrid')
        ap.add_argument('--top-k', type=int, default=5)
        ap.add_argument('--candidate-k', type=int, default=10)
        ap.add_argument('--min-cosine', type=float, default=0.35)
        ap.add_argument('--no-gate', action='store_true', help='Disable cosine relevance heuristic (for experiments only)')
        ap.add_argument('--mock', action='store_true', help='Use offline mock LLM')
        ap.add_argument('--show-prompt', action='store_true')
        if action == 'ask':
            ap.add_argument('--question', '-q', required=True)
    args = p.parse_args()
    try:
        if args.action == 'index':
            build_index(args.input, db_path=args.db,
                        reset=args.reset, model_name=args.model)
            return
        engine = SearchEngine(db=args.db, model=args.model)
        generator = make_generator(args.mock)
        if args.action == 'ask':
            ask_once(args.question, args, engine, generator)
        else:
            print('RAG ready. Type exit to quit.')
            while True:
                question = input('\n你：').strip()
                if question.lower() in ('exit', 'quit', 'q'):
                    break
                if not question:
                    continue
                ask_once(question, args, engine, generator)
    except (KeyboardInterrupt, EOFError):
        print('\n已退出')
    except Exception as exc:
        # This is a local teaching app. Production code should use structured logs
        # and classify errors without leaking credentials or full response payloads.
        from openai import (AuthenticationError, RateLimitError, APIConnectionError,
                            APITimeoutError, APIStatusError)
        if isinstance(exc, AuthenticationError):
            msg = 'API 密钥无效（401），请检查 .env。'
        elif isinstance(exc, RateLimitError):
            msg = '遇到额度或限流问题（429）。'
        elif isinstance(exc, APITimeoutError):
            msg = 'API 请求超时。'
        elif isinstance(exc, APIConnectionError):
            msg = '无法连接 API，请检查网络、代理或 BASE_URL。'
        elif isinstance(exc, APIStatusError):
            msg = f'API 返回 HTTP {exc.status_code}：请检查模型 ID 和接口兼容性。'
        else:
            msg = str(exc)
        print('运行失败：' + msg, file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
