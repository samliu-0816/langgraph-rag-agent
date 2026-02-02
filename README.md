# 🤖 LangGraph RAG Agent

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![LangGraph](https://img.shields.io/badge/LangGraph-Stateful-orange)
![RAG](https://img.shields.io/badge/Architecture-RAG-green)

## 📖 简介 (Introduction)

这是一个基于 **LangGraph** 和 **RAG (检索增强生成)** 架构构建的智能 Agent 模板。它旨在解决传统 LLM 应用中"无状态"和"知识幻觉"的问题。

本项目展示了如何构建一个具备以下能力的 Agent：
1.  **长期记忆**: 使用 `MemorySaver` 持久化对话状态（Checkpointer）。
2.  **私有知识库**: 集成 **FAISS** 向量库和 **DashScope (通义千问)** Embeddings。
3.  **图与工作流**: 使用 LangGraph 的图结构（Graph）精细控制 Agent 的决策流程。

## 🚀 核心特性 (Features)

- **State Management**: 使用 LangGraph 的 `StateGraph` 管理多轮对话上下文。
- **RAG Integration**: 自定义工具 `search_internal_knowledge` 连接本地向量数据。
- **Model Agnostic**: 支持 OpenAI 或 DashScope 等多种 LLM 后端。
- **Extensible**: 易于扩展 Function Calling 和其他 ToolNode。

## 🛠️ 技术栈 (Tech Stack)

- **框架**: [LangChain](https://www.langchain.com/), [LangGraph](https://langchain-ai.github.io/langgraph/)
- **向量数据库**: FAISS
- **Embeddings**: DashScope (Aliyun)
- **环境管理**: Python 3.10+ / Docker (Optional)


