"""可配置定位优先策略；不得依赖题目、文件名、关键词或固定行号。"""

import asyncio
import json
from importlib import import_module
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from langchain.agents.middleware import ModelRequest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage


def locator_result(name="grep", status="success"):
    return ToolMessage(name=name, tool_call_id="loc", content=json.dumps({"matches": []}), status=status)


def request(messages):
    return ModelRequest(model=object(), messages=messages, state={"messages": messages}, tools=[SimpleNamespace(name=name) for name in ("glob", "search", "grep", "read")])


def test_policy_hides_read_then_restores_it_even_after_empty_grep():
    module = import_module("rag_core.agentic.agent")
    policy = module.LocateFirstMiddleware()
    initial = request([HumanMessage(content="任意内容问题")])
    filtered = policy.wrap_model_call(initial, lambda value: value)
    assert [tool.name for tool in filtered.tools] == ["glob", "search", "grep"]
    assert initial.tools[-1].name == "read"
    after = request([HumanMessage(content="另一种措辞"), locator_result()])
    assert policy.wrap_model_call(after, lambda value: value).tools[-1].name == "read"


def test_policy_does_not_reuse_previous_turn_or_failed_locator():
    module = import_module("rag_core.agentic.agent")
    policy = module.LocateFirstMiddleware()
    for messages in ([HumanMessage(content="上轮"), locator_result(), HumanMessage(content="新问题")], [HumanMessage(content="问题"), locator_result(status="error")]):
        assert all(tool.name != "read" for tool in policy.wrap_model_call(request(messages), lambda value: value).tools)


def test_policy_checks_execution_not_only_model_schema():
    module = import_module("rag_core.agentic.agent")
    policy = module.LocateFirstMiddleware()
    calls = []
    tool_request = SimpleNamespace(tool_call={"name": "read", "id": "r1", "args": {"path": "/任意文件.md", "start_line": 37}}, state={"messages": [HumanMessage(content="内容问题")]})
    def handler(value):
        calls.append(value)
        return "executed"
    result = policy.wrap_tool_call(tool_request, handler)
    assert result.status == "error"
    assert calls == []
    tool_request.state["messages"].append(locator_result())
    assert policy.wrap_tool_call(tool_request, handler) == "executed"
    assert len(calls) == 1


def test_async_policy_matches_sync_behaviour():
    module = import_module("rag_core.agentic.agent")
    policy = module.LocateFirstMiddleware()
    async def handler(value):
        return value
    filtered = asyncio.run(policy.awrap_model_call(request([HumanMessage(content="问题")]), handler))
    assert all(tool.name != "read" for tool in filtered.tools)
    tool_request = SimpleNamespace(tool_call={"name": "read", "id": "r1"}, state={"messages": []})
    assert asyncio.run(policy.awrap_tool_call(tool_request, handler)).status == "error"


def test_policy_is_opt_in_and_does_not_change_six_call_budget(monkeypatch):
    module = import_module("rag_core.agentic.agent")
    options = []
    monkeypatch.setattr(module, "create_agent", lambda **kwargs: options.append(kwargs))
    module.build_knowledge_agent(object(), [])
    module.build_knowledge_agent(object(), [], retrieval_policy="locate_first")
    assert not any(isinstance(item, module.LocateFirstMiddleware) for item in options[0]["middleware"])
    assert any(isinstance(item, module.LocateFirstMiddleware) for item in options[1]["middleware"])
    assert options[1]["middleware"][0].run_limit == 6
    with pytest.raises(ValueError):
        module.build_knowledge_agent(object(), [], retrieval_policy="unknown")


@pytest.mark.parametrize("streaming", [False, True])
def test_runtime_passes_policy_to_both_entrypoints(tmp_path, monkeypatch, streaming):
    runner = import_module("rag_core.agentic.runner")
    received = []
    class FakeAgent:
        def invoke(self, *args, **kwargs):
            return {"messages": []}
        def stream(self, *args, **kwargs):
            yield {"type": "values", "data": {"messages": []}}
    def build(*args, **kwargs):
        received.append(kwargs)
        return FakeAgent()
    monkeypatch.setattr(runner, "build_knowledge_agent", build)
    runtime = runner.AgenticRuntime(tmp_path, object(), object(), [], retrieval_policy="locate_first")
    if streaming:
        list(runner.stream_agentic_question(runtime, "没有固定业务词", "test"))
    else:
        runner.run_agentic_question(runtime, "没有固定业务词", "test")
    assert received == [{"retrieval_policy": "locate_first"}]


def test_demo_policy_comes_from_configuration_not_question(tmp_path):
    demo = import_module("app.demo")
    runner = import_module("rag_core.agentic.runner")
    path = tmp_path / demo.DEMO_QUESTION_RELATIVE_PATH
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"id": "arbitrary", "question": "请解释维护手册中的注意事项", "retrieval_policy": "locate_first"}), encoding="utf-8")
    runtime = runner.AgenticRuntime(tmp_path, object(), object(), [])
    app = demo.create_demo_app(tmp_path, runtime_factory=lambda: runtime)
    with TestClient(app):
        assert app.state.pilot_runtime.retrieval_policy == "locate_first"
        assert app.state.pilot_runtime.vector_store is runtime.vector_store
    assert runtime.retrieval_policy == "free"


def test_real_agent_graph_changes_visible_tools_after_locator():
    """通过真实 Agent 图验证模型绑定的工具，而不只验证辅助函数。"""
    from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
    from langchain_core.tools import StructuredTool
    module = import_module("rag_core.agentic.agent")
    visible = []
    class FakeModel(FakeMessagesListChatModel):
        def bind_tools(self, tools, **kwargs):
            visible.append([tool.name if hasattr(tool, "name") else tool.get("name", tool.get("function", {}).get("name")) for tool in tools])
            return self
    def fake_glob() -> dict:
        """返回示例文件路径。"""
        return {"matches": [{"path": "/手册.md"}]}
    def fake_grep() -> dict:
        """无命中也应完成定位尝试，允许恢复阅读。"""
        return {"matches": [], "truncated": False}
    def fake_read() -> dict:
        """验证工具可以实际执行。"""
        return {"lines": []}
    responses = [AIMessage(content="", tool_calls=[{"name": name, "args": {}, "id": name, "type": "tool_call"}]) for name in ("glob", "grep", "read")]
    responses.append(AIMessage(content="", tool_calls=[{"name": "GroundedAnswer", "args": {"answer_type": "conversation", "answer": "结束", "evidence_ids": []}, "id": "answer", "type": "tool_call"}]))
    model = FakeModel(responses=responses)
    tools = [StructuredTool.from_function(func, name=name) for func, name in ((fake_glob, "glob"), (fake_grep, "grep"), (fake_read, "read"))]
    result = module.build_knowledge_agent(model, tools, retrieval_policy="locate_first").invoke({"messages": [("user", "任意内容问题")]}, config={"configurable": {"thread_id": "policy-test"}})
    assert "read" not in visible[0] and "read" not in visible[1]
    assert "read" in visible[2]
    traces = module.extract_tool_traces(result["messages"])
    assert [trace["status"] for trace in traces] == ["success"] * 3
