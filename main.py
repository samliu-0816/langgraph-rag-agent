import os
import operator
from typing import Annotated, Sequence, TypedDict

# 加载环境变量
from dotenv import load_dotenv

load_dotenv()

# FastAPI 相关
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

# LangChain/Graph 相关
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_community.chat_models import ChatTongyi
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

# --- 导入工具 ---
# 1. 内部 RAG 工具
from tools.rag_tool import search_internal_knowledge
# 2. 外部搜索工具 (Tavily)
from langchain_community.tools.tavily_search import TavilySearchResults

# --- 1. FastAPI 初始化 ---
app = FastAPI(
    title="LangGraph Agent API",
    description="支持联网(Tavily)和内部知识库(RAG)的智能体服务",
    version="1.0.0"
)


class ChatRequest(BaseModel):
    query: str
    thread_id: str = "default_user"


class ChatResponse(BaseModel):
    response: str
    thread_id: str


# --- 2. Agent 构建 ---

# LLM 模型
llm = ChatTongyi(model="qwen-turbo", temperature=0)

# 初始化 Tavily (限制返回3条结果)
tavily_search = TavilySearchResults(max_results=3)

# 工具列表：既能查内部，又能查外部
tools = [search_internal_knowledge, tavily_search]

llm_with_tools = llm.bind_tools(tools)


# 状态定义
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]


# 节点逻辑
def call_model(state: AgentState):
    messages = state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


def should_continue(state: AgentState):
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:
        return "tools"
    return END


# 图构建
workflow = StateGraph(AgentState)
workflow.add_node("agent", call_model)
workflow.add_node("tools", ToolNode(tools))

workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", should_continue)
workflow.add_edge("tools", "agent")

# 编译图 (启用记忆)
checkpointer = MemorySaver()
agent_app = workflow.compile(checkpointer=checkpointer)


# --- 3. API 路由 ---

@app.get("/health")
async def health_check():
    return {"status": "ok", "service": "langgraph-agent-api"}


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    对话接口：自动判断是否需要联网或查库
    """
    config = {"configurable": {"thread_id": request.thread_id}}
    inputs = {"messages": [HumanMessage(content=request.query)]}

    try:
        # 调用 LangGraph
        final_state = agent_app.invoke(inputs, config=config)
        last_message = final_state["messages"][-1]

        return ChatResponse(
            response=last_message.content,
            thread_id=request.thread_id
        )
    except Exception as e:
        # 实际生产中建议记录日志
        raise HTTPException(status_code=500, detail=str(e))


# 本地调试启动入口
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)