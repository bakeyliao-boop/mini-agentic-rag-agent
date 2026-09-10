"""知识库 Agent 的构建、调用限制与运行数据提取。"""

import json
from typing import Literal

from langchain.agents import create_agent
from langchain.agents.middleware import (
    AgentMiddleware,
    AgentState,
    hook_config,
)
from langchain.agents.middleware.tool_call_limit import ToolCallLimitMiddleware
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.checkpoint.memory import InMemorySaver

from rag_core.models import GroundedAnswer
from rag_core.agentic.prompts import (
    KNOWLEDGE_AGENT_PROMPT_VERSION,
    KNOWLEDGE_AGENT_SYSTEM_PROMPT,
)

KNOWLEDGE_TOOL_NAMES = frozenset({"ls", "glob", "search", "read", "grep"})
RetrievalPolicy = Literal["free", "locate_first"]


def _location_attempt_completed(messages) -> bool:
    """仅检查当前用户轮次是否完成定位尝试，不判断命中是否相关。"""
    names = {}
    completed = False
    for message in messages:
        if isinstance(message, HumanMessage):
            names = {}
            completed = False
        elif isinstance(message, AIMessage):
            names.update({call["id"]: call["name"] for call in message.tool_calls})
        elif isinstance(message, ToolMessage) and message.status != "error":
            name = message.name or names.get(message.tool_call_id)
            if name in {"search", "grep"}:
                completed = True
    return completed


class LocateFirstMiddleware(AgentMiddleware):
    """可选策略：首次 read 前先完成一次内容定位，grep 空命中也恢复读取。

    不识别题目、文件名或关键词，不建立行号白名单；只对启用此策略的请求生效。
    新用户轮次重新判断，避免上一轮定位结果自动放开下一轮的 read。
    search 返回 none 时仍由既有停止中间件结束，不在这里改变该策略。
    """

    def _model_request(self, request):
        if _location_attempt_completed(request.state.get("messages", [])):
            return request
        return request.override(tools=[
            tool for tool in request.tools
            if (tool.get("name", tool.get("function", {}).get("name")) if isinstance(tool, dict) else tool.name) != "read"
        ])

    def _read_rejection(self, request):
        if request.tool_call["name"] == "read" and not _location_attempt_completed(request.state.get("messages", [])):
            return ToolMessage(
                content="当前检索策略要求先调用 search 或 grep 尝试定位，再使用 read 读取原文。",
                name="read", tool_call_id=request.tool_call["id"], status="error",
            )
        return None

    def wrap_model_call(self, request, handler):
        return handler(self._model_request(request))

    async def awrap_model_call(self, request, handler):
        return await handler(self._model_request(request))

    def wrap_tool_call(self, request, handler):
        rejection = self._read_rejection(request)
        return rejection if rejection is not None else handler(request)

    async def awrap_tool_call(self, request, handler):
        rejection = self._read_rejection(request)
        return rejection if rejection is not None else await handler(request)


class KnowledgeToolCallLimitMiddleware(ToolCallLimitMiddleware):
    """限制知识库工具；用完预算后保留最终结构化答复的机会。"""

    def _budget_model_request(self, request):
        """沿用父类的真实运行计数，不根据历史消息数估算剩余预算。"""
        if self.run_limit is None:
            return request
        count_key = self.tool_name or "__all__"
        used = request.state.get("run_tool_call_count", {}).get(count_key, 0)
        if used >= self.run_limit:
            # 只撤下知识库工具。保留 response_format，LangChain 仍会绑定
            # GroundedAnswer；可以有据回答，也可以如实返回 insufficient。
            return request.override(tools=[])
        return request

    def wrap_model_call(self, request, handler):
        return handler(self._budget_model_request(request))

    async def awrap_model_call(self, request, handler):
        return await handler(self._budget_model_request(request))

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
    *,
    retrieval_policy: RetrievalPolicy = "free",
) -> object:
    """使用指定对话模型和知识库工具创建 LangChain Agent。"""

    if retrieval_policy not in ("free", "locate_first"):
        raise ValueError("unknown retrieval_policy")
    middleware = [
        KnowledgeToolCallLimitMiddleware(run_limit=6, exit_behavior="end"),
        KnowledgeRetrievalStopMiddleware(),
    ]
    if retrieval_policy == "locate_first":
        middleware.append(LocateFirstMiddleware())
    return create_agent(
        model=chat_model,
        tools=tools,
        system_prompt=KNOWLEDGE_AGENT_SYSTEM_PROMPT,
        middleware=middleware,
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
