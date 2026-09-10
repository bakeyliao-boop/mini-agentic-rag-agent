"""Mini-Agent 真实工具事件流测试。"""

import json
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from rag_core.agentic.runner import AgenticRuntime, stream_agentic_question
from rag_core.models import GroundedAnswer


def test_stream_agentic_question_emits_live_tools_and_final_result(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """事件流应按真实节点顺序产生工具开始、完成和最终回答。"""

    from rag_core.agentic import runner

    tool_call = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "search",
                "args": {"query": "优先审查", "path": "/"},
                "id": "search-call-1",
                "type": "tool_call",
            }
        ],
        usage_metadata={
            "input_tokens": 10,
            "output_tokens": 2,
            "total_tokens": 12,
        },
    )
    tool_result = ToolMessage(
        content=json.dumps(
            {
                "hits": [{"path": "/办事指南.md"}],
                "retrieval_status": "relevant",
            },
            ensure_ascii=False,
        ),
        name="search",
        tool_call_id="search-call-1",
        status="success",
    )
    final_message = AIMessage(
        content="回答完成",
        usage_metadata={
            "input_tokens": 5,
            "output_tokens": 2,
            "total_tokens": 7,
        },
    )
    structured_response = GroundedAnswer(
        answer_type="conversation",
        answer="回答完成",
        evidence_ids=[],
    )
    final_messages = [
        HumanMessage(content="问题"),
        tool_call,
        tool_result,
        final_message,
    ]
    stream_options: list[dict[str, object]] = []

    class FakeAgent:
        def stream(self, input_data, **options):
            stream_options.append(options)
            yield {
                "type": "values",
                "data": {"messages": [HumanMessage(content="问题")]},
            }
            yield {
                "type": "updates",
                "data": {"model": {"messages": [tool_call]}},
            }
            yield {
                "type": "updates",
                "data": {"tools": {"messages": [tool_result]}},
            }
            yield {
                "type": "values",
                "data": {
                    "messages": final_messages,
                    "structured_response": structured_response,
                },
            }

    monkeypatch.setattr(runner, "build_knowledge_tools", lambda *a, **k: [])
    monkeypatch.setattr(
        runner,
        "build_knowledge_agent",
        lambda chat_model, tools: FakeAgent(),
    )
    runtime = AgenticRuntime(
        knowledge_root=tmp_path,
        vector_store=object(),
        chat_model=object(),
        path_snapshot=[],
    )

    events = list(
        stream_agentic_question(
            runtime=runtime,
            question="问题",
            thread_id="stream-test",
        )
    )

    assert [event["event"] for event in events] == [
        "run_started",
        "tool_started",
        "tool_completed",
        "completed",
    ]
    assert events[1]["data"] == {
        "step": 1,
        "tool_call_id": "search-call-1",
        "name": "search",
        "args": {"query": "优先审查", "path": "/"},
        "status": "running",
    }
    assert events[2]["data"]["summary"] == {
        "hit_count": 1,
        "retrieval_status": "relevant",
    }
    assert events[-1]["data"]["answer"] == "回答完成"
    assert events[-1]["data"]["token_usage"]["total_tokens"] == 19
    assert events[-1]["data"]["tool_traces"][0]["status"] == "success"
    assert stream_options[0]["stream_mode"] == ["updates", "values"]
    assert stream_options[0]["version"] == "v2"


def test_stream_pairs_nameless_results_and_ignores_duplicate_events(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """重复节点更新不能增加步骤，缺少工具名仍应按调用 ID 配对。"""

    from rag_core.agentic import runner

    call = AIMessage(content="", tool_calls=[
        {"name": "read", "args": {"path": "/guide.md"}, "id": "r1", "type": "tool_call"},
    ])
    result = ToolMessage(
        content=json.dumps({"path": "/guide.md", "lines": [{"line": 2, "text": "正文"}], "next_line": 3}),
        tool_call_id="r1",
        status="success",
    )
    internal = AIMessage(content="", tool_calls=[
        {"name": "GroundedAnswer", "args": {}, "id": "final-1", "type": "tool_call"},
    ])

    class FakeAgent:
        def stream(self, *args, **kwargs):
            for message in (call, call, result, result, internal):
                yield {"type": "updates", "data": {"node": {"messages": [message]}}}
            yield {"type": "values", "data": {"messages": [call, result, internal]}}

        def invoke(self, *args, **kwargs):
            raise AssertionError("事件流结束后不得再次调用模型")

    monkeypatch.setattr(runner, "build_knowledge_tools", lambda *a, **k: [])
    monkeypatch.setattr(runner, "build_knowledge_agent", lambda *a, **k: FakeAgent())
    runtime = AgenticRuntime(tmp_path, object(), object(), [])
    events = list(stream_agentic_question(runtime, "问题", "stream-pair"))
    assert [event["event"] for event in events] == [
        "run_started", "tool_started", "tool_completed", "completed",
    ]
    assert events[2]["data"]["name"] == "read"
    assert events[2]["data"]["summary"] == {
        "path": "/guide.md", "start_line": 2, "end_line": 2, "next_line": 3,
    }
    assert events[-1]["data"]["answer_type"] == "insufficient"
