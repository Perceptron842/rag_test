"""Create an honest Markdown summary from REAL retrieval_summary.json results.

Does not generate fake scores and does not perform model evaluation.
"""
from pathlib import Path
import json, argparse


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--input',default='outputs/retrieval_summary.json')
    p.add_argument('--output',default='outputs/PROJECT_REPORT.md')
    args=p.parse_args()
    file=Path(args.input)
    if not file.exists():
        raise SystemExit('没有评测数据。请先运行 evaluate_retrieval.py，不会凭空填写分数。')
    metrics=json.loads(file.read_text(encoding='utf-8'))
    if not metrics: raise SystemExit('评测结果为空')
    lines=['# RAG 检索评测汇总（真实运行数据）','',
           '来源：本机 `evaluate_retrieval.py` 输出。仅适用于当前语料与标注。','',
           '| 方法 | 问题数 | Recall@K | Hit@K | MRR@K | 平均耗时(ms) |',
           '| --- | ---: | ---: | ---: | ---: | ---: |']
    for name,stats in metrics.items():
        def v(key):
            value=stats.get(key)
            return f'{value:.4f}' if isinstance(value,(int,float)) else '未统计'
        lines.append(f'| {name} | {stats.get("questions", "-")} | {v("recall_at_k")} | {v("hit_at_k")} | {v("mrr_at_k")} | {v("avg_search_ms")} |')
    lines += ['', '## 失败案例（手工补充）','',
              '1. 问题、期望相关文档、实际检索结果、原因、改进与复测：待填写','',
              '2. 待填写','', '3. 待填写','',
              '## 局限','',
              '本项目默认使用少量教学 Chunk 与多种问题改写，不能推论生产场景效果；模型生成的正确性和引用支持性未由该脚本评测。','']
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text('\n'.join(lines), encoding='utf-8')
    print(f'已生成 {out}')

if __name__=='__main__':main()
