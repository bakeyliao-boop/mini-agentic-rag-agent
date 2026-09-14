"""路径已发现后的进展说明及重复目录调用护栏，不预置业务路径。"""

import asyncio
import json
from types import SimpleNamespace

import pytest
from langchain.agents.middleware import ModelRequest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from rag_core.agentic.agent import LocateFirstMiddleware
from rag_core.models import GroundedAnswer


def completed(name, args, payload, call_id):
    """构造一次模型调用及其成功返回。"""
    return [
        AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}]),
        ToolMessage(name=name, tool_call_id=call_id, content=json.dumps(payload), status="success"),
    ]


def discovered():
    return [HumanMessage(content="请核对两份任意资料")] + completed(
        "glob", {"target": "第一份", "target_type": "filename"},
        {"matches": [{"path": "/资源/第一份.md"}]}, "g1",
    )


def model_request(messages, tools=None, system=None):
    return ModelRequest(
        model=object(), messages=messages, state={"messages": messages},
        tools=tools if tools is not None else [SimpleNamespace(name=name) for name in ("ls", "glob", "search", "grep", "read")],
        system_message=system or SystemMessage(content="原有规则", id="original-system", additional_kwargs={"keep": True}),
        response_format=GroundedAnswer,
    )


def tool_request(messages, name, args):
    return SimpleNamespace(tool_call={"name": name, "args": args, "id": "next"}, state={"messages": messages})


def test_discovery_notice_keeps_new_resource_discovery_and_history():
    """发现第一份后仍允许不同glob发现第二份，不立即锁整个发现阶段。"""
    messages = discovered()
    original = model_request(messages)
    result = LocateFirstMiddleware().wrap_model_call(original, lambda item: item)
    assert [item.name for item in result.tools] == ["ls", "glob", "search", "grep"]
    assert "/资源/第一份.md" in result.system_message.content
    assert "search" in result.system_message.content and "grep" in result.system_message.content
    assert result.messages == original.messages
    assert result.response_format is original.response_format
    assert original.system_message.content == "原有规则"
    assert result.system_message.id == "original-system"
    assert result.system_message.additional_kwargs == {"keep": True}
    assert LocateFirstMiddleware().wrap_tool_call(
        tool_request(messages, "glob", {"target": "第二份", "target_type": "filename"}), lambda item: "executed"
    ) == "executed"


def test_successful_same_directory_query_is_not_executed_twice():
    messages = discovered() + completed(
        "ls", {"path": "/资源"}, {"path": "/资源", "entries": [{"path": "/资源/第一份.md", "type": "file"}]}, "ls1",
    )
    executed = []
    result = LocateFirstMiddleware().wrap_tool_call(
        tool_request(messages, "ls", {"path": "/资源/"}), lambda item: executed.append(item)
    )
    assert executed == []
    assert isinstance(result, ToolMessage)
    assert result.status == "error"
    assert json.loads(result.content)["code"] == "duplicate_discovery"


def test_duplicate_glob_recognizes_default_root_but_allows_new_target():
    messages = discovered()
    policy = LocateFirstMiddleware()
    duplicate = tool_request(messages, "glob", {"target": "第一份", "target_type": "filename", "path": "/"})
    result = policy.wrap_tool_call(duplicate, lambda item: "executed")
    assert isinstance(result, ToolMessage) and result.status == "error"
    assert policy.wrap_tool_call(tool_request(messages, "glob", {"target": "第二份", "target_type": "filename"}), lambda item: "executed") == "executed"


def duplicate_history():
    messages = discovered() + completed("ls", {"path": "/资源"}, {"entries": [{"path": "/资源/第一份.md", "type": "file"}]}, "ls1")
    messages += [
        AIMessage(content="", tool_calls=[{"name": "ls", "args": {"path": "/资源"}, "id": "repeat", "type": "tool_call"}]),
        ToolMessage(name="ls", tool_call_id="repeat", status="error", content=json.dumps({"code": "duplicate_discovery"})),
    ]
    return messages


def test_only_repeated_tool_is_paused_then_restored_after_empty_locator():
    policy = LocateFirstMiddleware()
    messages = duplicate_history()
    paused = policy.wrap_model_call(model_request(messages), lambda item: item)
    assert [item.name for item in paused.tools] == ["glob", "search", "grep"]
    messages += completed("grep", {"path": "/资源/第一份.md", "patterns": ["词"]}, {"matches": []}, "loc")
    restored = model_request(messages)
    assert policy.wrap_model_call(restored, lambda item: item) is restored
    assert policy.wrap_tool_call(tool_request(messages, "ls", {"path": "/资源"}), lambda item: "executed") == "executed"


def test_new_human_turn_resets_notice_and_discovery_history():
    messages = duplicate_history() + [HumanMessage(content="新的目录问题")]
    policy = LocateFirstMiddleware()
    result = policy.wrap_model_call(model_request(messages), lambda item: item)
    assert [item.name for item in result.tools] == ["ls", "glob", "search", "grep"]
    assert result.system_message.content == "原有规则"
    assert policy.wrap_tool_call(tool_request(messages, "ls", {"path": "/资源"}), lambda item: "executed") == "executed"


def test_rejected_glob_pauses_that_name_until_locator_not_all_new_discovery():
    """重复glob后整个glob暂退；这是违规后的限制，不是首份命中即锁。"""
    messages = discovered() + [
        AIMessage(content="", tool_calls=[{"name": "glob", "args": {"target": "第一份", "target_type": "filename"}, "id": "repeat", "type": "tool_call"}]),
        ToolMessage(name="glob", tool_call_id="repeat", content=json.dumps({"code": "duplicate_discovery"}), status="error"),
    ]
    policy = LocateFirstMiddleware()
    assert [item.name for item in policy.wrap_model_call(model_request(messages), lambda item: item).tools] == ["ls", "search", "grep"]
    result = policy.wrap_tool_call(tool_request(messages, "glob", {"target": "第二份", "target_type": "filename"}), lambda item: "executed")
    assert json.loads(result.content)["code"] == "discovery_paused"
    messages += completed("grep", {}, {"matches": []}, "loc")
    assert policy.wrap_tool_call(tool_request(messages, "glob", {"target": "第二份", "target_type": "filename"}), lambda item: "executed") == "executed"


def test_empty_second_glob_can_retry_and_directory_only_is_not_file_discovery():
    messages = discovered() + completed("glob", {"target": "第二份", "target_type": "filename"}, {"matches": []}, "g2")
    policy = LocateFirstMiddleware()
    assert policy.wrap_tool_call(tool_request(messages, "glob", {"target": "第二份", "target_type": "filename"}), lambda item: "executed") == "executed"
    directories = [HumanMessage(content="目录浏览")] + completed("ls", {"path": "/"}, {"entries": [{"path": "/资源", "type": "directory"}]}, "ls1")
    assert policy.wrap_model_call(model_request(directories), lambda item: item).system_message.content == "原有规则"


def test_discovery_correlates_missing_result_name_but_ignores_unpaired_results():
    messages = discovered()
    messages[-1].name = None
    policy = LocateFirstMiddleware()
    assert "/资源/第一份.md" in policy.wrap_model_call(model_request(messages), lambda item: item).system_message.content
    unpaired = [HumanMessage(content="内容问题"), ToolMessage(name="glob", tool_call_id="unknown", content=json.dumps({"matches": [{"path": "/not-called.md"}]}))]
    assert policy.wrap_model_call(model_request(unpaired), lambda item: item).system_message.content == "原有规则"


@pytest.mark.parametrize("parallel", [False, True])
def test_two_different_globs_in_real_graph_still_execute(parallel):
    """找两份资料仍然可用，串行与并行都不因第一份命中被拦截。"""
    from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
    from langchain_core.tools import StructuredTool
    from rag_core.agentic.agent import build_knowledge_agent
    executed = []
    class FakeModel(FakeMessagesListChatModel):
        def bind_tools(self, tools, **kwargs):
            return self
    def glob(target: str) -> dict:
        """不同目标分别返回对应文件。"""
        executed.append(target)
        return {"matches": [{"path": f"/{target}.md"}]}
    calls = [{"name": "glob", "args": {"target": target}, "id": target, "type": "tool_call"} for target in ("A", "B")]
    responses = [AIMessage(content="", tool_calls=calls)] if parallel else [AIMessage(content="", tool_calls=[call]) for call in calls]
    responses.append(AIMessage(content="", tool_calls=[{"name": "GroundedAnswer", "args": {"answer_type": "insufficient", "answer": "停止。", "evidence_ids": []}, "id": "final", "type": "tool_call"}]))
    result = build_knowledge_agent(FakeModel(responses=responses), [StructuredTool.from_function(glob)], retrieval_policy="locate_first").invoke({"messages": [HumanMessage(content="任意两资源任务")]}, config={"configurable": {"thread_id": f"discovery-{parallel}"}})
    assert sorted(executed) == ["A", "B"]
    assert not any(isinstance(item, ToolMessage) and item.status == "error" for item in result["messages"])


@pytest.mark.parametrize("content,status", [("null", "success"), ("broken", "success"), ('{"matches":[]}', "success"), ('{"matches":null}', "success"), ('[]', "success"), ('{"matches":[{"path":"/a.md"}]}', "error")])
def test_empty_failed_or_invalid_discovery_does_not_add_notice(content, status):
    messages = [HumanMessage(content="内容问题"), AIMessage(content="", tool_calls=[{"name": "glob", "args": {"target": "任意", "target_type": "filename"}, "id": "g", "type": "tool_call"}]), ToolMessage(name="glob", tool_call_id="g", content=content, status=status)]
    result = LocateFirstMiddleware().wrap_model_call(model_request(messages), lambda item: item)
    assert result.system_message.content == "原有规则"
    assert [item.name for item in result.tools] == ["ls", "glob", "search", "grep"]


def test_budget_exhaustion_does_not_reintroduce_tools_or_require_locator():
    original = model_request(duplicate_history(), tools=[])
    result = LocateFirstMiddleware().wrap_model_call(original, lambda item: item)
    assert result.tools == []
    assert result.system_message.content == "原有规则"
    assert result.response_format is GroundedAnswer


def test_dictionary_tools_and_block_system_content_are_preserved():
    tools = [{"name": "ls", "function": None}, {"type": "function", "function": {"name": "glob"}}, {"name": "search"}, {"name": "grep"}, {"name": "read"}, {"type": "unknown"}]
    system = SystemMessage(content=[{"type": "text", "text": "原规则"}], additional_kwargs={"keep": True})
    original = model_request(discovered(), tools=tools, system=system)
    result = LocateFirstMiddleware().wrap_model_call(original, lambda item: item)
    assert len(result.tools) == 5
    assert result.tools[-1] == {"type": "unknown"}
    assert result.system_message.content[0] == system.content[0]
    assert result.system_message.content[-1]["type"] == "text"
    assert system.content == [{"type": "text", "text": "原规则"}]


def test_async_duplicate_guard_matches_sync():
    messages = discovered() + completed("ls", {"path": "/资源"}, {"entries": [{"path": "/资源/第一份.md", "type": "file"}]}, "ls1")
    executed = []
    async def handler(item):
        executed.append(item)
        return "executed"
    result = asyncio.run(LocateFirstMiddleware().awrap_tool_call(tool_request(messages, "ls", {"path": "/资源"}), handler))
    assert result.status == "error" and executed == []


@pytest.mark.parametrize("policy", ["locate_first", "free"])
def test_real_agent_rejects_repeat_and_keeps_locator_then_read_available(policy):
    """重放重复目录调用：执行端拒绝后，真实图仍可完成定位和读取。"""
    from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
    from langchain_core.tools import StructuredTool
    from rag_core.agentic.agent import build_knowledge_agent, extract_tool_traces
    bindings, executed = [], []
    class FakeModel(FakeMessagesListChatModel):
        def bind_tools(self, tools, **kwargs):
            bindings.append([item.name if hasattr(item, "name") else item.get("name", item.get("function", {}).get("name")) for item in tools])
            return self
    def glob() -> dict:
        """返回一个任意候选文件。"""
        executed.append("glob")
        return {"matches": [{"path": "/资源/任意.md"}]}
    def ls(path: str) -> dict:
        """第一次目录列表可执行，重复不应再次执行。"""
        executed.append("ls")
        return {"entries": [{"path": "/资源/任意.md", "type": "file"}]}
    def grep() -> dict:
        """空结果也完成定位尝试。"""
        executed.append("grep")
        return {"matches": []}
    def read() -> dict:
        """只验证读取阶段可执行。"""
        executed.append("read")
        return {"lines": []}
    responses = [AIMessage(content="", tool_calls=[{"name": name, "args": {"path": "/资源"} if name == "ls" else {}, "id": f"call{index}", "type": "tool_call"}]) for index, name in enumerate(("glob", "ls", "ls", "grep", "read"))]
    responses.append(AIMessage(content="", tool_calls=[{"name": "GroundedAnswer", "args": {"answer_type": "insufficient", "answer": "结束。", "evidence_ids": []}, "id": "final", "type": "tool_call"}]))
    tools = [StructuredTool.from_function(func, name=func.__name__) for func in (glob, ls, grep, read)]
    result = build_knowledge_agent(FakeModel(responses=responses), tools, retrieval_policy=policy).invoke({"messages": [HumanMessage(content="任意内容任务")]}, config={"configurable": {"thread_id": "progress-test"}})
    assert executed == (["glob", "ls", "grep", "read"] if policy == "locate_first" else ["glob", "ls", "ls", "grep", "read"])
    assert ("ls" not in bindings[3]) == (policy == "locate_first")
    assert "grep" in bindings[3]
    assert "ls" in bindings[4] and "read" in bindings[4]
    assert [item["status"] for item in extract_tool_traces(result["messages"])] == (["success", "success", "error", "success", "success"] if policy == "locate_first" else ["success"] * 5)


def test_sixth_rejected_call_still_counts_and_final_answer_has_no_locator_notice(monkeypatch):
    """拦截不退款；第六次结束后仍遵守已有预算收口，不重新加工具。"""
    from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
    from langchain_core.tools import StructuredTool
    from rag_core.agentic.agent import KnowledgeToolCallLimitMiddleware, build_knowledge_agent
    bindings, notices, executed, counts = [], [], [], []
    original_request = KnowledgeToolCallLimitMiddleware._budget_model_request
    def observe_count(self, request):
        # run计数是UntrackedValue私有运行状态，不从invoke结果或checkpoint读取。
        counts.append(request.state.get("run_tool_call_count", {}).get("__all__", 0))
        return original_request(self, request)
    monkeypatch.setattr(KnowledgeToolCallLimitMiddleware, "_budget_model_request", observe_count)
    class FakeModel(FakeMessagesListChatModel):
        def bind_tools(self, tools, **kwargs):
            bindings.append([item.name if hasattr(item, "name") else item.get("name", item.get("function", {}).get("name")) for item in tools])
            return self
        def _generate(self, messages, *args, **kwargs):
            notices.append(any(isinstance(item, SystemMessage) and "【定位进展】" in item.content for item in messages))
            return super()._generate(messages, *args, **kwargs)
    def glob() -> dict:
        """返回候选文件。"""
        return {"matches": [{"path": "/资源/任意.md"}]}
    def ls(path: str) -> dict:
        """不同路径正常执行，最后一次同路径应拦截。"""
        executed.append(path)
        return {"entries": [{"path": "/资源/任意.md", "type": "file"}]}
    def grep() -> dict:
        """仅提供可用定位工具，此测试不执行它。"""
        return {"matches": []}
    responses = [AIMessage(content="", tool_calls=[{"name": "glob", "args": {}, "id": "g", "type": "tool_call"}])]
    responses.extend(AIMessage(content="", tool_calls=[{"name": "ls", "args": {"path": path}, "id": f"ls{index}", "type": "tool_call"}]) for index, path in enumerate(("/a", "/b", "/c", "/d", "/d")))
    responses.append(AIMessage(content="", tool_calls=[{"name": "GroundedAnswer", "args": {"answer_type": "insufficient", "answer": "预算内没有正文证据。", "evidence_ids": []}, "id": "final", "type": "tool_call"}]))
    knowledge_agent = build_knowledge_agent(FakeModel(responses=responses), [StructuredTool.from_function(glob), StructuredTool.from_function(ls), StructuredTool.from_function(grep)], retrieval_policy="locate_first")
    config = {"configurable": {"thread_id": "sixth-rejected"}}
    knowledge_agent.invoke({"messages": [HumanMessage(content="任意内容问题")]}, config=config)
    assert len(executed) == 4
    assert counts[-1] == 6
    assert bindings[-1] == ["GroundedAnswer"]
    assert any(notices[:-1])
    assert notices[-1] is False
