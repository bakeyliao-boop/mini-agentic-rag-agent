from langchain.agents import create_agent
from langchain.agents.middleware.tool_call_limit import ToolCallLimitMiddleware
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.checkpoint.memory import InMemorySaver


KNOWLEDGE_AGENT_SYSTEM_PROMPT = """你是一个基于本地知识库回答问题的助手。

你可以使用 ls、search 和 read 三个工具：
- ls 用于浏览知识库目录。
- search 只用于定位候选内容，返回结果不能直接作为回答证据。
- read 用于读取并核实 Markdown 原文。

处理目录或范围问题时，应使用 ls 逐层定位目标目录。通过 ls 找到具体目录后，必须将该目录作为 search 的 path 参数。
回答知识类问题前，必须使用 read 核实原文；证据不足时应明确说明无法从知识库确定答案。
"""


def build_knowledge_agent(
    chat_model: BaseChatModel,
    tools: list[BaseTool],
) -> object:
    """使用指定对话模型和知识库工具创建 LangChain Agent。"""

    return create_agent(
        model=chat_model,
        tools=tools,
        system_prompt=KNOWLEDGE_AGENT_SYSTEM_PROMPT,
        middleware=[
            ToolCallLimitMiddleware(
                run_limit=6,
                exit_behavior="error",
            )
        ],
        checkpointer=InMemorySaver(),
    )


def extract_tool_traces(
    messages: list[BaseMessage],
) -> list[dict[str, object]]:
    """从 Agent 消息中提取工具名称、参数和执行状态。"""

    traces: list[dict[str, object]] = []
    traces_by_id: dict[str, dict[str, object]] = {}

    for message in messages:
        if isinstance(message, AIMessage):
            for tool_call in message.tool_calls:
                tool_call_id = tool_call["id"]
                trace = {
                    "step": len(traces) + 1,
                    "tool_call_id": tool_call_id,
                    "name": tool_call["name"],
                    "args": tool_call["args"],
                    "status": "missing_result",
                }
                traces.append(trace)
                traces_by_id[tool_call_id] = trace

        if isinstance(message, ToolMessage):
            trace = traces_by_id.get(message.tool_call_id)
            if trace is not None:
                trace["status"] = message.status

    return traces
