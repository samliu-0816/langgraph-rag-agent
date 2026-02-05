import os
import logging
import operator
from typing import Annotated, Sequence, TypedDict

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_community.chat_models import ChatTongyi
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

# 导入工具
from tools.rag_tool import search_internal_knowledge
from langchain_community.tools.tavily_search import TavilySearchResults

# ==========================================
# 1. 全局日志配置 (Logging Setup)
# ==========================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # 输出到控制台
        logging.FileHandler("app.log", encoding='utf-8')  # 输出到文件
    ]
)
logger = logging.getLogger(__name__)

# ==========================================
# 2. FastAPI 初始化
# ==========================================
app = FastAPI(title="LangGraph Agent API", version="1.0.0")


class ChatRequest(BaseModel):
    query: str
    thread_id: str = "default_user"


class ChatResponse(BaseModel):
    response: str
    thread_id: str


# ==========================================
# 3. Agent 构建
# ==========================================

# 初始化 LLM (固定使用通义千问)
llm = ChatTongyi(model="qwen-turbo", temperature=0)

# 初始化工具 (Tavily + RAG)
tavily_tool = TavilySearchResults(max_results=3)
tools = [search_internal_knowledge, tavily_tool]

# 绑定工具
llm_with_tools = llm.bind_tools(tools)


# 定义状态
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]


# 节点逻辑
def call_model(state: AgentState):
    messages = state["messages"]
    # 记录日志
    logger.info(f"🤖 [Agent] 正在调用 LLM (Qwen-Turbo)...")
    try:
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}
    except Exception as e:
        logger.error(f"[Agent] LLM 调用失败: {e}", exc_info=True)
        raise e


def should_continue(state: AgentState):
    messages = state["messages"]
    last_message = messages[-1]

    if last_message.tool_calls:
        logger.info(f"[Agent] 决策: 调用工具 ({len(last_message.tool_calls)} 个)")
        return "tools"

    logger.info("[Agent] 决策: 结束对话，生成回复")
    return END


# 构建图
workflow = StateGraph(AgentState)
workflow.add_node("agent", call_model)
workflow.add_node("tools", ToolNode(tools))

workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", should_continue)
workflow.add_edge("tools", "agent")

checkpointer = MemorySaver()
agent_app = workflow.compile(checkpointer=checkpointer)


# ==========================================
# 4. API 路由
# ==========================================

@app.on_event("startup")
async def startup_event():
    logger.info("系统启动完成，正在监听端口 8000...")


@app.get("/health")
async def health_check():
    return {"status": "ok", "service": "langgraph-agent"}


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    logger.info(f"[API] 收到请求 | Thread: {request.thread_id} | Query: {request.query}")

    config = {"configurable": {"thread_id": request.thread_id}}
    inputs = {"messages": [HumanMessage(content=request.query)]}

    try:
        final_state = agent_app.invoke(inputs, config=config)
        last_message = final_state["messages"][-1]

        logger.info(f"[API] 请求处理完成，返回 {len(last_message.content)} 字符")

        return ChatResponse(
            response=last_message.content,
            thread_id=request.thread_id
        )
    except Exception as e:
        logger.error(f" [API] 处理异常: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)