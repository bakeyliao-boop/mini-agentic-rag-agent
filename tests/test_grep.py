"""单文件关键词定位及其 Agent 接线测试；不调用真实模型。"""

import json
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, ToolMessage

from rag_core.knowledge import store


def test_grep_matches_literals_and_read_reproduces_lines(tmp_path):
    """目录和正文命中都应保留，同一行多个关键词只返回一次。"""
    (tmp_path / "指南.md").write_bytes(
        "目录：优先审查\r\n\r\n## 优先审查\r\n材料：预审 A.*B\r\n预审 优先审查\r\nAXXB".encode("utf-8")
    )
    result = store.grep_knowledge_file("/指南.md", tmp_path, ["优先审查", "预审", "A.*B"])
    assert [hit["line"] for hit in result["matches"]] == [1, 3, 4, 5]
    assert result["truncated"] is False
    for hit in result["matches"]:
        page = store.read_knowledge_page(hit["path"], tmp_path, start_line=hit["line"], limit=1)
        assert page["lines"][0]["text"] == hit["text"]
        assert "evidence_id" not in hit
    literal = store.grep_knowledge_file("/指南.md", tmp_path, ["A.*B"])
    assert [hit["line"] for hit in literal["matches"]] == [4]


def test_grep_bounds_results_and_centres_long_hit(tmp_path):
    """数量截断与长行截断均须显式标记，长行保留真实命中词。"""
    long_line = "甲" * 400 + "目标" + "乙" * 400
    (tmp_path / "指南.md").write_text(long_line + "\n目标\n目标", encoding="utf-8")
    result = store.grep_knowledge_file("/指南.md", tmp_path, ["目标"], limit=1)
    assert len(result["matches"]) == 1
    hit = result["matches"][0]
    assert len(hit["text"]) <= 300
    assert "目标" in hit["text"]
    assert hit["text"] in long_line
    assert hit["text_truncated"] is True
    assert result["truncated"] is True
    exact = store.grep_knowledge_file("/指南.md", tmp_path, ["不存在"])
    assert exact == {"matches": [], "truncated": False}


def test_grep_limit_exactly_reached_is_not_truncated(tmp_path):
    (tmp_path / "指南.md").write_text("目标\n无关", encoding="utf-8")
    result = store.grep_knowledge_file("/指南.md", tmp_path, ["目标", " 目标 "], limit=1)
    assert len(result["matches"]) == 1
    assert result["truncated"] is False


@pytest.mark.parametrize("patterns,limit", [([], 10), ([" "], 10), (["x" * 201], 10), (["x"] * 11, 10), (["x"], 0), (["x"], 21)])
def test_grep_rejects_invalid_arguments(tmp_path, patterns, limit):
    with pytest.raises(ValueError):
        store.grep_knowledge_file("/指南.md", tmp_path, patterns, limit)


def test_grep_rejects_directory_missing_non_markdown_and_escape(tmp_path):
    (tmp_path / "附件.txt").write_text("正文", encoding="utf-8")
    for path, error in [("/", IsADirectoryError), ("/missing.md", FileNotFoundError), ("/附件.txt", ValueError), ("/../outside.md", ValueError)]:
        with pytest.raises(error):
            store.grep_knowledge_file(path, tmp_path, ["正文"])


def test_grep_rejects_symlink_outside_root(tmp_path):
    root = tmp_path / "knowledge"
    root.mkdir()
    outside = tmp_path / "outside.md"
    outside.write_text("不能读取", encoding="utf-8")
    try:
        (root / "escape.md").symlink_to(outside)
    except OSError:
        pytest.skip("当前环境不支持符号链接")
    with pytest.raises(ValueError):
        store.grep_knowledge_file("/escape.md", root, ["不能"])


def test_grep_tool_does_not_register_evidence_and_errors_are_recoverable(tmp_path, monkeypatch):
    from rag_core.agentic.tools import build_knowledge_tools
    from rag_core.agentic.evidence import EvidenceRegistry

    registry = EvidenceRegistry("grep-test")
    def reject_registration(*args):
        raise AssertionError("grep 不得登记证据")
    monkeypatch.setattr(registry, "register_read_page", reject_registration)
    (tmp_path / "指南.md").write_text("## 办理条件", encoding="utf-8")
    tool = next(t for t in build_knowledge_tools(tmp_path, object(), registry) if t.name == "grep")
    result = tool.invoke({"path": "/指南.md", "patterns": ["办理条件"]})
    assert result["matches"][0]["line"] == 1
    for args in ({"path": "/missing.md", "patterns": ["条件"]}, {"path": "/指南.md", "patterns": []}):
        response = tool.invoke({"name": "grep", "args": args, "id": "g1", "type": "tool_call"})
        assert response.status == "error"


def test_grep_is_counted_traced_and_empty_hits_do_not_stop():
    from rag_core.agentic.agent import KnowledgeToolCallLimitMiddleware, KnowledgeRetrievalStopMiddleware, extract_tool_traces
    limiter = KnowledgeToolCallLimitMiddleware(run_limit=6, exit_behavior="end")
    assert limiter._matches_tool_filter({"name": "grep"}) is True
    assert limiter._matches_tool_filter({"name": "GroundedAnswer"}) is False
    call = AIMessage(content="", tool_calls=[{"name": "grep", "args": {"path": "/指南.md", "patterns": ["条件"]}, "id": "g1", "type": "tool_call"}])
    result = ToolMessage(name="grep", tool_call_id="g1", content=json.dumps({"matches": [], "truncated": False}))
    assert extract_tool_traces([call, result])[0]["name"] == "grep"
    assert KnowledgeRetrievalStopMiddleware().before_model({"messages": [call, result]}, object()) is None


def test_grep_events_are_streamed_before_final_answer(tmp_path, monkeypatch):
    from rag_core.agentic import runner
    call = AIMessage(content="", tool_calls=[{"name": "grep", "args": {"path": "/指南.md", "patterns": ["条件"]}, "id": "g1", "type": "tool_call"}])
    result = ToolMessage(tool_call_id="g1", content=json.dumps({"matches": [{"path": "/指南.md", "line": 3, "text": "条件"}], "truncated": False}))
    class FakeAgent:
        def stream(self, *args, **kwargs):
            for message in (call, result):
                yield {"type": "updates", "data": {"node": {"messages": [message]}}}
            yield {"type": "values", "data": {"messages": [call, result]}}
    monkeypatch.setattr(runner, "build_knowledge_agent", lambda *args: FakeAgent())
    events = runner.stream_agentic_question(runner.AgenticRuntime(tmp_path, object(), object(), []), "问题", "stream-grep")
    assert next(events)["event"] == "run_started"
    assert next(events)["data"]["name"] == "grep"
    completed = next(events)
    assert completed["event"] == "tool_completed"
    assert completed["data"]["summary"] == {"match_count": 1, "truncated": False}
    assert next(events)["event"] == "completed"


def test_seventh_grep_is_blocked_before_execution():
    """真实 Agent 循环最多执行六次 grep，第七次只记录拒绝结果。"""
    from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
    from langchain_core.tools import StructuredTool
    from rag_core.agentic.agent import build_knowledge_agent, extract_tool_traces

    class ScriptedModel(FakeMessagesListChatModel):
        def bind_tools(self, tools, **kwargs):
            return self

    executed = []
    def fake_grep(path: str, patterns: list[str]) -> dict:
        """离线模拟关键词定位。"""
        executed.append(patterns)
        return {"matches": [], "truncated": False}

    responses = [AIMessage(content="", tool_calls=[{
        "name": "grep", "args": {"path": "/指南.md", "patterns": [f"条件{n}"]},
        "id": f"grep-{n}", "type": "tool_call",
    }]) for n in range(7)]
    model = ScriptedModel(responses=responses)
    tool = StructuredTool.from_function(fake_grep, name="grep")
    result = build_knowledge_agent(model, [tool]).invoke(
        {"messages": [("user", "找办理条件")]},
        config={"configurable": {"thread_id": "grep-budget"}},
    )
    assert len(executed) == 6
    traces = extract_tool_traces(result["messages"])
    assert len(traces) == 7
    assert traces[-1]["status"] == "error"


def test_grep_schema_is_bounded_and_requires_file_and_patterns(tmp_path):
    """模型收到的 Schema 与工具实际接受的最小接口一致。"""
    from rag_core.agentic.tools import build_knowledge_tools
    from rag_core.agentic.evidence import EvidenceRegistry
    tool = next(t for t in build_knowledge_tools(tmp_path, object(), EvidenceRegistry("schema")) if t.name == "grep")
    schema = tool.args_schema.model_json_schema()
    assert set(schema["required"]) == {"path", "patterns"}
    assert schema["properties"]["patterns"]["minItems"] == 1
    assert schema["properties"]["patterns"]["maxItems"] == 10
    assert schema["properties"]["limit"]["default"] == 10
    assert schema["properties"]["limit"]["maximum"] == 20
