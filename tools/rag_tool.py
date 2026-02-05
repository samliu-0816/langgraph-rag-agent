import os
import logging
from dotenv import load_dotenv

# 1. 获取当前模块的 Logger
logger = logging.getLogger(__name__)

load_dotenv()

from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.tools import tool
from langchain_pinecone import PineconeVectorStore

# 2. 初始化配置
embeddings = DashScopeEmbeddings(model="text-embedding-v2")
index_name = os.getenv("PINECONE_INDEX_NAME")

# 3. 尝试连接 Pinecone
try:
    if index_name:
        vector_store = PineconeVectorStore.from_existing_index(
            index_name=index_name,
            embedding=embeddings
        )
        logger.info(f"[RAG] 成功连接到 Pinecone 索引: {index_name}")
    else:
        logger.warning("[RAG] 未配置 PINECONE_INDEX_NAME，RAG 功能将不可用")
        vector_store = None
except Exception as e:
    logger.error(f"[RAG] 连接 Pinecone 失败: {e}", exc_info=True)
    vector_store = None


@tool
def search_internal_knowledge(query: str) -> str:
    """
    当用户询问关于 LangGraph、项目架构、技术细节或内部业务知识时，必须使用此工具。
    """
    logger.info(f"[RAG] 正在检索: {query}")

    if not vector_store:
        logger.warning("[RAG] 数据库未连接，无法检索")
        return "内部知识库暂时不可用。"

    try:
        # 检索 Top 3
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})
        results = retriever.invoke(query)

        if not results:
            logger.info("[RAG] 未找到相关文档")
            return "内部知识库中没有找到相关内容。"

        logger.info(f"[RAG] 检索成功，找到 {len(results)} 条相关文档")
        return "\n\n".join([f"[内部文档]: {doc.page_content}" for doc in results])

    except Exception as e:
        logger.error(f"[RAG] 检索过程发生异常: {str(e)}", exc_info=True)
        return f"检索出错: {str(e)}"