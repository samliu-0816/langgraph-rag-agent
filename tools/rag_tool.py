import os
from dotenv import load_dotenv

# 1. 强制在导入其他库之前加载环境变量
load_dotenv()

from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.tools import tool
from langchain_core.documents import Document

# 2. 初始化 Embedding (现在能读到 Key 了)
embeddings = DashScopeEmbeddings(model="text-embedding-v1")

INDEX_PATH = "faiss_index"

def get_vector_store():
    """加载或创建向量索引"""
    if os.path.exists(INDEX_PATH):
        print("正在加载本地向量索引...")
        return FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    else:
        print("本地索引不存在，正在创建示例索引...")
        docs = [
            Document(page_content="LangGraph 是一个用于构建有状态、多角色 LLM 应用的库。"),
            Document(page_content="RAG (检索增强生成) 技术可以通过引入外部知识库来减少 LLM 的幻觉。"),
            Document(page_content="这个 Agent 项目使用了 DashScope Embeddings 和 FAISS。"),
            Document(page_content="项目架构包含：FastAPI 后端, Tavily 联网搜索, 和 Docker 容器化部署。"),
        ]
        vector_store = FAISS.from_documents(docs, embeddings)
        vector_store.save_local(INDEX_PATH)
        return vector_store

vector_store = get_vector_store()

@tool
def search_internal_knowledge(query: str) -> str:
    """
    当用户询问关于 LangGraph、项目架构、技术细节或内部业务时，使用此工具。
    """
    try:
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})
        results = retriever.invoke(query)
        return "\n\n".join([f"[内部文档]: {doc.page_content}" for doc in results])
    except Exception as e:
        return f"检索错误: {str(e)}"