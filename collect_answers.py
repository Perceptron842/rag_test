"""Generate answers for manual assessment; makes REAL LLM calls (paid).

No mock scores are called factual evaluation. Requires indexed Day5 corpus and API credentials.
"""
import argparse,csv,os,json
from pathlib import Path
from dotenv import load_dotenv
from rag_pipeline import OpenAIGenerator,run_rag
from app import SearchEngine


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--db',default='chroma_db')
    p.add_argument('--limit',type=int,default=5)
    p.add_argument('--mode',choices=['vector','bm25','hybrid','rerank'],default='hybrid')
    p.add_argument('--output',default='outputs/answer_review.csv')
    args=p.parse_args()
    if args.limit<1: raise ValueError('limit must be positive')
    load_dotenv(Path(__file__).with_name('.env'))
    llm=OpenAIGenerator(model=os.getenv('OPENAI_MODEL',''),api_key=os.getenv('OPENAI_API_KEY',''),base_url=os.getenv('OPENAI_BASE_URL') or None)
    engine=SearchEngine(db=args.db)
    gold=json.loads(Path('data/gold_questions.json').read_text(encoding='utf8'))
    rows=[]
    for case in gold[:args.limit]:
        result=run_rag(case['question'],engine,llm,mode=args.mode,top_k=5,candidate_k=10,min_cosine=None)
        rows.append({'qid':case['id'],'question':case['question'],
                     'reference_answer':case['reference_answer'],
                     'generated_answer':result.answer,'source_ids':'|'.join(s['id'] for s in result.sources),
                     'evidence':result.evidence,'status':result.status,
                     'correctness':'','faithfulness':'','error_type':'','notes':''})
        print('Generated',case['id'])
    path=Path(args.output);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print('Review each answer manually and fill correctness/faithfulness as 0 or 1:',path)

if __name__=='__main__': main()
