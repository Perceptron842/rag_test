"""Day 7 offline checks: corpus labels, config hygiene, and minimal retrieval/generation wiring.

No GPU, embedding download or API key required.
"""
import json
from pathlib import Path
from metrics import score_query
from rag_pipeline import run_rag, MockGenerator

BASE = Path(__file__).resolve().parent


def verify_data():
    chunks=json.loads((BASE/'data/sample_chunks.json').read_text(encoding='utf-8'))
    gold=json.loads((BASE/'data/gold_questions.json').read_text(encoding='utf-8'))
    topics={c['topic'] for c in chunks}
    assert chunks and gold, '语料和题目均不能为空'
    assert len({x['id'] for x in gold}) == len(gold), '问题ID重复'
    for x in gold:
        assert x['question'].strip() and x['relevant_topics'], '问题缺少文本或标签'
        assert set(x['relevant_topics']) <= topics, f'无法匹配的标签：{x["id"]}'
    print(f'✓ 教学语料 {len(chunks)} Chunk，标注题 {len(gold)} 条，主题标签一致')


def verify_pipeline():
    good={'id':'test1','text':'Python字典使用键值对，可以使用data[key]读取值。',
          'metadata':{'source':'test.md','page':1},'cosine_similarity':0.8}
    class Retriever:
        def __init__(self, hits): self.hits=hits
        def search(self, q, mode='hybrid', top_k=5, candidate_k=10):return self.hits[:top_k]
    valid=run_rag('字典如何读取值？',Retriever([good]),MockGenerator())
    assert valid.status=='mock' and valid.sources[0]['source']=='test.md，第1页'
    empty=run_rag('未知问题',Retriever([]),MockGenerator())
    assert empty.status=='insufficient_evidence' and not empty.sources
    assert score_query(['B','A'],{'A'},2)['rr']==0.5
    print('✓ 模拟问答、来源追溯、无证据拒答、MRR 计算均正常')


def verify_hygiene():
    gitignore=(BASE/'.gitignore').read_text(encoding='utf-8')
    for p in ['.env','chroma_db/','*.safetensors']:
        assert p in gitignore, f'.gitignore 缺少 {p}'
    assert (BASE/'.env.example').exists(), '缺少示例配置'
    print('✓ 存在 .env.example 和基本 Git 忽略配置（不代表已完成安全审计）')


if __name__=='__main__':
    verify_data()
    verify_pipeline()
    verify_hygiene()
    print('Day 7 离线自检通过')
