# 项目架构与技术选型

```text
知识库文档 / data/*.json
       ↓
Chunk 清洗及来源元数据（Day 2，见输入格式）
       ↓
Embedding（SentenceTransformer） + Chroma（离线持久化）
       ↓
用户问题 → Vector Search + BM25 → RRF → 可选 CrossEncoder Rerank
       ↓
检索结果 / metadata + 来源标号
       ↓
启发式相关性检查（不能保证文档真的相关）
       ↓
Context / Prompt → LLM Chat Completions API
       ↓
回答 + 来源候选（需人工或程序进一步核实引用）
```

## 为什么这样选型？
- 中文教学样例：使用小型多语种 Embedding，在 CPU 上便于入门。
- Chroma：本地持久化简单，方便在 Windows 上运行；大规模服务需重新评估方案。
- BM25 + Vector：词面精确匹配与语义相似性互补；不保证在任何数据集上都提升。
- RRF：两路评分量纲不同，按排名而非直接累加原分值。
- Rerank：只重排召回候选，需要额外计算开销；召回不到的答案无法被救回。
- Prompt + 来源：生成阶段限制证据使用；不能自动消除幻觉，也不能证明引用正确。

## 关键文件
- `retriever.py`：Embedding 与 Chroma
- `core.py`：BM25、RRF、可选重排
- `app.py`：整合检索器、对外提供 `.search()`
- `rag_pipeline.py`：构造 Context、Prompt、生成与拒答判断
- `chat_cli.py` / `web_ui.py`：交互入口
- `metrics.py` / `evaluate_retrieval.py`：离线评测

## 当前不包括
文档上传/解析 UI、鉴权、多租户、容器化部署、LLM 生成的自动事实核验、独立的无答案拒答准确率测试。项目是教学原型，不应声称已生产部署。
