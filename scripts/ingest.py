import os
import logging
from dotenv import load_dotenv

# 1. 引入加载器和分割器
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.embeddings import DashScopeEmbeddings
from langchain_pinecone import PineconeVectorStore

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("IngestScript")

load_dotenv()


def ingest_data():
    api_key = os.getenv("PINECONE_API_KEY")
    index_name = os.getenv("PINECONE_INDEX_NAME")

    # 定义一个文件路径
    data_path = "data"

    if not api_key or not index_name:
        logger.critical("配置缺失: 请检查环境变量")
        return

    # --- 第一步：加载文件 (Load) ---
    logger.info(f"正在扫描 {data_path} 目录下的文件...")

    # glob="**/*.pdf" 表示递归查找所有子文件夹里的 pdf
    pdf_loader = DirectoryLoader(data_path, glob="**/*.pdf", loader_cls=PyPDFLoader)
    txt_loader = DirectoryLoader(data_path, glob="**/*.txt", loader_cls=TextLoader)

    # 执行加载
    pdf_docs = pdf_loader.load()
    txt_docs = txt_loader.load()

    # 合并所有文档
    all_docs = pdf_docs + txt_docs

    if not all_docs:
        logger.warning("未找到任何文档，请检查 data 目录！")
        return

    logger.info(f"成功加载 {len(all_docs)} 个文档对象")

    # --- 第二步：文本切分 (Split) ---
    # 为什么要切分？因为 Embedding 模型一次处理的长度有限（Token限制），
    # 而且切分成小块能让检索更精准（不会因为一段话太长而丢失重点）。
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # 每个切片大约 1000 个字符
        chunk_overlap=200,  # 切片之间重叠 200 个字符（防止上下文在切分处断裂）
    )

    # 执行切分
    split_docs = text_splitter.split_documents(all_docs)
    logger.info(f"切分完成，共生成 {len(split_docs)} 个数据段 (Chunks)")

    # --- 第三步：向量化并入库 (Embed & Store) ---
    logger.info(f"准备连接 Pinecone，索引: {index_name}")

    embeddings = DashScopeEmbeddings(model="text-embedding-v2")

    try:
        # 这里的 batch_size 是为了防止一次发给 Pinecone 太多数据导致超时
        PineconeVectorStore.from_documents(
            documents=split_docs,
            embedding=embeddings,
            index_name=index_name,
            batch_size=100,
        )
        logger.info(f"成功将 {len(split_docs)} 条数据注入 Pinecone！")

    except Exception as e:
        logger.error(f"数据注入失败: {e}", exc_info=True)


if __name__ == "__main__":
    # 确保你有一个叫 data 的文件夹，并且里面放了文件
    if not os.path.exists("data"):
        os.makedirs("data")
        print("已创建 data 文件夹，请把文件放进去再运行。")
    else:
        ingest_data()