"""知识库 Agent 的构建、调用限制与运行数据提取。"""

import json

from langchain.agents import create_agent
from langchain.agents.middleware import (
    AgentMiddleware,
    AgentState,
    hook_config,
)
from langchain.agents.middleware.tool_call_limit import ToolCallLimitMiddleware
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.checkpoint.memory import InMemorySaver

from rag_core.models import GroundedAnswer
from rag_core.agentic.prompts import (
    KNOWLEDGE_AGENT_PROMPT_VERSION,
    KNOWLEDGE_AGENT_SYSTEM_PROMPT,
)

KNOWLEDGE_TOOL_NAMES = frozenset({"ls", "glob", "search", "read", "grep"})


class KnowledgeToolCallLimitMiddleware(ToolCallLimitMiddleware):
    """只限制知识库工具，不把结构化回答当作知识库工具。"""

    def _matches_tool_filter(self, tool_call: dict[str, object]) -> bool:
        """判断一次调用是否属于知识库工具，包括 grep。"""

        return tool_call.get("name") in KNOWLEDGE_TOOL_NAMES


class KnowledgeRetrievalStopMiddleware(AgentMiddleware):
    """search 返回 none 时，在模型再次生成前确定性结束 Agent。

    search 只返回候选，none 表示没有可信证据；
    继续让模型改写查询只会浪费 Token 和时延，
    因此由程序直接终止本轮运行，最终由运行时降级为 insufficient。
    """

    @hook_config(can_jump_to=["end"])
    def before_model(
        self,
        state: AgentState,
        runtime: object,
    ) -> dict[str, object] | None:
        """检查最近一次工具结果，search 无证据时跳到结束。"""

        messages = state.get("messages", [])
        if not messages:
            return None

        last_message = messages[-1]
        if not isinstance(last_message, ToolMessage):
            return None
        if last_message.name != "search":
            return None

        content = last_message.content
        if not isinstance(content, str):
            return None
        try:
            payload = json.loads(content)
        except (TypeError, ValueError):
            return None
        if not isinstance(payload, dict):
            return None

        if payload.get("retrieval_status") == "none":
            return {"jump_to": "end"}
        return None


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
            KnowledgeToolCallLimitMiddleware(
                run_limit=6,
                exit_behavior="end",
            ),
            KnowledgeRetrievalStopMiddleware(),
        ],
        checkpointer=InMemorySaver(),
        response_format=GroundedAnswer,
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
                if tool_call["name"] not in KNOWLEDGE_TOOL_NAMES:
                    continue

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


def extract_token_usage(
    messages: list[BaseMessage],
) -> dict[str, object]:
    """累加 Agent 每轮模型消息中的 token 使用量。"""

    input_tokens = 0
    output_tokens = 0
    total_tokens = 0
    reasoning_tokens = 0

    for message in messages:
        if not isinstance(message, AIMessage):
            continue

        usage_metadata = message.usage_metadata or {}
        input_tokens += int(usage_metadata.get("input_tokens", 0))
        output_tokens += int(usage_metadata.get("output_tokens", 0))
        total_tokens += int(usage_metadata.get("total_tokens", 0))

        output_details = usage_metadata.get("output_token_details") or {}
        reasoning_tokens += int(output_details.get("reasoning", 0))

    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "output_token_details": {
            "reasoning": reasoning_tokens,
        },
    }
