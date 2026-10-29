"""Optional Streamlit UI. Install: pip install streamlit"""
import os
import streamlit as st
from dotenv import load_dotenv
from app import SearchEngine
from rag_pipeline import run_rag, MockGenerator, OpenAIGenerator

load_dotenv()
st.set_page_config(page_title='Day 5 RAG', layout='centered')
st.title('Day 5：RAG 知识库问答')
st.caption('先执行 python chat_cli.py index ... 构建索引')

@st.cache_resource
def load_engine():
    return SearchEngine(db=os.getenv('RAG_DB', 'chroma_db'))

question = st.text_area('输入问题', 'Python字典如何通过key查询value？')
mode = st.selectbox('检索方式', ['hybrid', 'vector', 'bm25', 'rerank'])
mock = st.checkbox('离线模拟（不调用 LLM）', value=True)
min_cos = st.slider('余弦相关性阈值（教学示例，需评测校准）', 0.0, 0.8, 0.35, step=0.05)
if st.button('发送问题', type='primary'):
    try:
        engine = load_engine()
        generator = MockGenerator() if mock else OpenAIGenerator(
            os.getenv('OPENAI_MODEL', ''), os.getenv('OPENAI_API_KEY', ''),
            os.getenv('OPENAI_BASE_URL') or None)
        result = run_rag(question, engine, generator, mode=mode, min_cosine=min_cos)
        st.subheader('回答')
        st.write(result.answer)
        with st.expander('查看检索来源'):
            for s in result.sources:
                st.write(f"[{s['number']}] {s['source']} — {s['excerpt']}")
        with st.expander('查看参考片段'):
            st.code(result.evidence)
    except Exception as exc:
        st.error('运行失败，请检查索引、模型下载、API 配置或终端错误日志。')
