# RAG 知识库问答系统（学习作品 / Day 7 整理版）

> 这是一个 **教学原型**，尚未验证生产级准确率、安全性或性能。请把你的真实结果写入报告，不要将模拟响应当作 LLM 真实回答。

## 功能
- 中文文本 Chunk → 多语种 SentenceTransformer Embedding → Chroma 持久化
- Vector / BM25 / Hybrid（RRF），可选 Cross-Encoder Rerank
- LLM Chat Completions 问答与来源候选、无检索结果拒答、启发式相关性门槛
- 无需 API Key 的 `--mock` 模式，人工答案评审与 Recall@K、Hit@K、MRR@K 评测
- 命令行问答与可选 Streamlit UI

## 目录与架构
详见 [ARCHITECTURE.md](ARCHITECTURE.md)。核心模块：`retriever.py`、`core.py`、`app.py`、`rag_pipeline.py`、`metrics.py`。

'''
retriever.py Embedding+Chroma
core.py BM25+RRF+Rerank
app.py 统一检索接口
rag_pipeline.py RAG主流程
chat_cli.py 命令行交互
web_ui.py Streamlit界面
metrics.py 检索评测指标
'''

## 快速复现（Windows PowerShell）

```powershell
conda create -n rag_dev python=3.11 -y
conda activate rag_dev
python -m pip install -r requirements.txt
python check_project.py
python -m unittest discover -s tests -v
```

以上离线自检不下载模型、不调用 LLM，但安装依赖可能需要联网。

首次索引（需下载 Embedding 模型）：

```powershell
python chat_cli.py index --input data/sample_chunks.json --reset
```

`--reset` 仅用于教学索引，会重建对应 collection。**不要对需要保留的数据使用**。

运行模拟问答（真实检索 + 模拟生成，不使用 LLM API）：

```powershell
python chat_cli.py ask -q "Python字典如何根据key取值？" --mock --show-prompt
```

真实 LLM：

```powershell
Copy-Item .env.example .env
notepad .env
python chat_cli.py ask -q "Python字典如何根据key取值？"
python chat_cli.py chat
```

`.env` 必须填真实可用的 `OPENAI_API_KEY`、`OPENAI_MODEL`。兼容网关还需正确填写 `OPENAI_BASE_URL`，并支持 Chat Completions；这一步可能产生费用。**千万不要把密钥提交到 GitHub**。

可选界面：

```powershell
python -m pip install streamlit
streamlit run web_ui.py
```

## 可复现评测（在相同 12 Chunk 教学数据索引上）

```powershell
python evaluate_retrieval.py --top-k 5 --candidate-k 10 --modes vector bm25 hybrid
python make_report.py
```

将生成 `outputs/retrieval_per_question.csv`、`outputs/retrieval_summary.json`、`outputs/PROJECT_REPORT.md`。**分数完全来自本机真实运行**，如果没跑评测，生成报告会直接报错。

生成回答评测（真实 API，可能收费；需人工判定）：

```powershell
python collect_answers.py --limit 5 --mode hybrid
# 手工填写 outputs/answer_review.csv 中 correctness / faithfulness
python review_answers.py --input outputs/answer_review.csv
```

更多方法见 [PROJECT_REPORT_TEMPLATE.md](PROJECT_REPORT_TEMPLATE.md)，面试口述题见 [INTERVIEW.md](INTERVIEW.md)。

## Day 7 验收清单

- [ ] 清楚解释索引阶段与在线问答阶段
- [ ] 不看答案写出简化的 `build_evidence()` / `build_messages()` / RRF 融合
- [ ] 运行 `check_project.py` 和全部 unittest
- [ ] 完成至少一条 Mock 问答，并核对来源
- [ ] 配置密钥后完成真实问答，检查引用是否支持结论（可选，需网络/额度）
- [ ] 跑出 Vector / BM25 / Hybrid 的真实评测数据
- [ ] 记录至少 3 个**自己的**失败案例、修复与复测情况
- [ ] 完成 `leetcode_11.py` 并能够解释双指针

## 常见故障
- 模型下载失败：检查模型仓库访问、代理、证书或改用已下载的本地模型路径（保持索引模型一致）。
- API 401/429/404：分别检查密钥、额度及模型/接口兼容。
- 检索器找不到 Collection：先索引或用 `--db` 指向已有数据库。
- 改变文档、Embedding 模型或分块设置后：重新索引并重新核对 gold labels。

## 局限与后续
- 默认 12 Chunk + 20 个问题改写仅适合教学，不能代表真实生产质量。
- `min_cosine=0.35` 是启发式阈值；BM25/RRF 得分不能当作余弦相似度。
- 答案来源列表不是自动事实验证，仍可能发生错误引用。
- 未做持续交互记忆、用户权限、数据更新一致性、生产部署。
- 第 2 周将基于此项目进入 Tool Calling、LangGraph 工作流和可靠性设计。

请在 GitHub 仓库提供真实运行截图、自己的设计决策和失败分析，不要上传任何 API Key 或未经授权的 PDF。
