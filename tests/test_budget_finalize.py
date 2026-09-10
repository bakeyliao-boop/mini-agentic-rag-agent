"""预算耗尽后仍应生成结构化答复，不再向模型提供知识库工具。"""

import asyncio

import pytest
from langchain.agents import create_agent
from langchain.agents.middleware import ModelRequest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import StructuredTool

from rag_core.agentic.agent import KnowledgeToolCallLimitMiddleware
from rag_core.models import GroundedAnswer


def test_budget_exhaustion_removes_tools_without_changing_history():
    limiter = KnowledgeToolCallLimitMiddleware(run_limit=6, exit_behavior="end")
    messages = [HumanMessage(content="任意知识问题")]
    for used, expected_tools in ((5, ["read"]), (6, [])):
        request = ModelRequest(model=object(), messages=messages, tools=[{"name": "read"}], state={"run_tool_call_count": {"__all__": used}}, response_format=GroundedAnswer)
        result = limiter.wrap_model_call(request, lambda value: value)
        assert [tool["name"] for tool in result.tools] == expected_tools
        assert result.messages == messages
        assert result.response_format is GroundedAnswer
        assert request.tools == [{"name": "read"}]


def test_async_budget_exhaustion_uses_same_final_answer_phase():
    limiter = KnowledgeToolCallLimitMiddleware(run_limit=6, exit_behavior="end")
    request = ModelRequest(model=object(), messages=[], tools=[{"name": "read"}], state={"run_tool_call_count": {"__all__": 6}}, response_format=GroundedAnswer)
    async def handler(value):
        return value
    result = asyncio.run(limiter.awrap_model_call(request, handler))
    assert result.tools == []
    assert result.response_format is GroundedAnswer


@pytest.mark.parametrize("limit,prepared", [(6, False), (4, True)])
def test_real_graph_binds_only_grounded_answer_after_run_budget(limit, prepared):
    """使用真实计数状态，不能把注入的准备历史误算进剩余四次预算。"""
    bindings = []
    executed = []
    class FakeModel(FakeMessagesListChatModel):
        def bind_tools(self, tools, **kwargs):
            bindings.append([tool.name if hasattr(tool, "name") else tool.get("name", tool.get("function", {}).get("name")) for tool in tools])
            return self
    def read(path: str) -> dict:
        """离线读取桩，只记录执行次数。"""
        executed.append(path)
        return {"lines": []}
    responses = [AIMessage(content="", tool_calls=[{"name": "read", "args": {"path": "/任意.md"}, "id": f"r{n}", "type": "tool_call"}]) for n in range(limit)]
    responses.append(AIMessage(content="", tool_calls=[{"name": "GroundedAnswer", "args": {"answer_type": "insufficient", "answer": "原文不足，无法回答。", "evidence_ids": []}, "id": "final", "type": "tool_call"}]))
    model = FakeModel(responses=responses)
    agent = create_agent(model=model, tools=[StructuredTool.from_function(read)], middleware=[KnowledgeToolCallLimitMiddleware(run_limit=limit, exit_behavior="end")], response_format=GroundedAnswer)
    messages = [HumanMessage(content="问题")]
    if prepared:
        for name in ("glob", "grep"):
            messages.extend([
                AIMessage(content="", tool_calls=[{"name": name, "args": {}, "id": name, "type": "tool_call"}]),
                ToolMessage(content="{}", name=name, tool_call_id=name),
            ])
    result = agent.invoke({"messages": messages})
    assert len(executed) == limit
    assert all("read" in names for names in bindings[:-1])
    assert bindings[-1] == ["GroundedAnswer"]
    assert result["structured_response"].answer_type == "insufficient"
    assert not any(isinstance(message, ToolMessage) and message.status == "error" for message in result["messages"])
