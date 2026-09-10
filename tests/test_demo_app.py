"""长文档渐进式检索演示应用测试。"""

import importlib
import json
from pathlib import Path

from fastapi.testclient import TestClient


def _write_demo_question(project_root: Path) -> str:
    """准备不同的演示题和压力题，确认演示入口读取独立配置。"""

    question = "请比较优先审查和快速预审的办理差异。"
    spec_path = (
        project_root
        / "evaluation"
        / "demos"
        / "patent_review_comparison_001.json"
    )
    spec_path.parent.mkdir(parents=True)
    spec_path.write_text(
        json.dumps(
            {
                "id": "patent-review-comparison-001",
                "question": question,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    pilot_path = (
        project_root / "evaluation" / "pilots" / "multi_chunk_guide_001.json"
    )
    pilot_path.parent.mkdir(parents=True)
    pilot_path.write_text(
        json.dumps(
            {"id": "multi-chunk-guide-001", "question": "旧压力题：三项业务台账。"},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return question


def test_demo_config_exposes_only_traditional_and_agentic_modes(
    tmp_path: Path,
) -> None:
    """前端配置只应提供传统 RAG 与 Mini-Agent 两个选项。"""

    demo_module = importlib.import_module("app.demo")
    question = _write_demo_question(tmp_path)
    runtime = object()
    runtime_calls = 0

    def runtime_factory() -> object:
        nonlocal runtime_calls
        runtime_calls += 1
        return runtime

    application = demo_module.create_demo_app(
        project_root=tmp_path,
        runtime_factory=runtime_factory,
    )

    with TestClient(application) as client:
        response = client.get("/demo/pilot/config")
        second_response = client.get("/demo/pilot/config")

    assert response.status_code == 200
    assert response.json() == {
        "question": question,
        "modes": ["traditional", "agentic"],
    }
    assert second_response.status_code == 200
    assert runtime_calls == 1


def test_demo_rejects_unexposed_mode_and_question_override(
    tmp_path: Path,
) -> None:
    """接口不接受固定多查询模式，也不允许前端覆盖冻结问题。"""

    demo_module = importlib.import_module("app.demo")
    _write_demo_question(tmp_path)
    application = demo_module.create_demo_app(
        project_root=tmp_path,
        runtime_factory=lambda: object(),
    )

    with TestClient(application) as client:
        unsupported_mode = client.post(
            "/demo/pilot/run",
            json={"mode": "multi_query"},
        )
        overridden_question = client.post(
            "/demo/pilot/run",
            json={
                "mode": "traditional",
                "question": "覆盖冻结问题",
            },
        )

    assert unsupported_mode.status_code == 422
    assert overridden_question.status_code == 422


def test_demo_run_returns_traditional_candidates(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """传统模式应返回单次检索候选，并且不伪装成已验证引用。"""

    demo_module = importlib.import_module("app.demo")
    question = _write_demo_question(tmp_path)
    runtime = type(
        "FakeRuntime",
        (),
        {"vector_store": object(), "chat_model": object()},
    )()
    calls: list[tuple[object, ...]] = []
    hits = [
        {
            "path": "/办事指南.md",
            "start_line": 10,
            "end_line": 20,
            "score": 0.8,
            "preview": "传统检索候选内容",
        }
    ]

    def fake_traditional_answer(**options: object) -> dict[str, object]:
        calls.append(("traditional", options))
        return {
            "answer": "传统 RAG 回答",
            "hits": hits,
            "token_usage": {"total_tokens": 120},
        }

    monkeypatch.setattr(
        demo_module,
        "answer_with_traditional_rag",
        fake_traditional_answer,
    )
    application = demo_module.create_demo_app(
        project_root=tmp_path,
        runtime_factory=lambda: runtime,
    )

    with TestClient(application) as client:
        response = client.post(
            "/demo/pilot/run",
            json={"mode": "traditional"},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["mode"] == "traditional"
    assert payload["question"] == question
    assert payload["answer"] == "传统 RAG 回答"
    assert payload["retrieval_hits"] == hits
    assert payload["citations"] == []
    assert payload["tool_traces"] == []
    assert payload["token_usage"] == {"total_tokens": 120}
    assert payload["finish_reason"] == "completed"
    assert payload["latency_ms"] >= 0
    assert calls[0][1]["question"] == question


def test_demo_run_returns_agentic_trace_and_verified_citations(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Mini-Agent 模式应返回渐进式工具轨迹与已验证引用。"""

    demo_module = importlib.import_module("app.demo")
    question = _write_demo_question(tmp_path)
    runtime = object()
    calls: list[tuple[object, ...]] = []
    citations = [
        {
            "path": "/办事指南.md",
            "start_line": 10,
            "end_line": 10,
            "quote": "已验证原文",
        }
    ]
    traces = [
        {
            "step": 1,
            "tool_call_id": "call-1",
            "name": "read",
            "args": {"path": "/办事指南.md"},
            "status": "success",
        }
    ]

    def fake_agentic_answer(**options: object) -> dict[str, object]:
        calls.append(("agentic", options))
        return {
            "answer_type": "knowledge",
            "answer": "Mini-Agent 回答",
            "citations": citations,
            "tool_traces": traces,
            "token_usage": {"total_tokens": 360},
            "thread_id": options["thread_id"],
        }

    monkeypatch.setattr(
        demo_module,
        "run_agentic_question",
        fake_agentic_answer,
    )
    application = demo_module.create_demo_app(
        project_root=tmp_path,
        runtime_factory=lambda: runtime,
    )

    with TestClient(application) as client:
        response = client.post(
            "/demo/pilot/run",
            json={"mode": "agentic"},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["mode"] == "agentic"
    assert payload["question"] == question
    assert payload["answer_type"] == "knowledge"
    assert payload["answer"] == "Mini-Agent 回答"
    assert payload["retrieval_hits"] == []
    assert payload["citations"] == citations
    assert payload["tool_traces"] == traces
    assert payload["token_usage"] == {"total_tokens": 360}
    assert payload["finish_reason"] == "completed"
    assert payload["latency_ms"] >= 0
    assert calls[0][1]["question"] == question


def test_demo_traditional_no_hit_returns_complete_insufficient_payload(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """传统模式无候选时仍应返回前端可直接展示的完整结构。"""

    demo_module = importlib.import_module("app.demo")
    question = _write_demo_question(tmp_path)
    runtime = type(
        "FakeRuntime",
        (),
        {"vector_store": object(), "chat_model": object()},
    )()
    monkeypatch.setattr(
        demo_module,
        "answer_with_traditional_rag",
        lambda **options: {"answer": "", "hits": []},
    )
    application = demo_module.create_demo_app(
        project_root=tmp_path,
        runtime_factory=lambda: runtime,
    )

    with TestClient(application) as client:
        response = client.post(
            "/demo/pilot/run",
            json={"mode": "traditional"},
        )

    assert response.status_code == 200
    assert response.json() == {
        "mode": "traditional",
        "question": question,
        "answer_type": "insufficient",
        "answer": "当前检索没有返回可用答案。",
        "retrieval_hits": [],
        "tool_traces": [],
        "citations": [],
        "token_usage": {},
        "latency_ms": response.json()["latency_ms"],
        "finish_reason": "insufficient",
    }


def test_demo_agentic_stream_preserves_event_order(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Mini-Agent 流接口应逐行返回真实工具事件和统一最终结果。"""

    demo_module = importlib.import_module("app.demo")
    question = _write_demo_question(tmp_path)
    runtime = object()

    def fake_stream(**options: object):
        assert options["question"] == question
        assert options["runtime"] is runtime
        yield {
            "event": "run_started",
            "data": {"thread_id": options["thread_id"]},
        }
        yield {
            "event": "tool_started",
            "data": {
                "step": 1,
                "tool_call_id": "read-1",
                "name": "read",
                "args": {"path": "/办事指南.md"},
                "status": "running",
            },
        }
        yield {
            "event": "tool_completed",
            "data": {
                "step": 1,
                "tool_call_id": "read-1",
                "name": "read",
                "status": "success",
                "summary": {"start_line": 10, "end_line": 20},
            },
        }
        yield {
            "event": "completed",
            "data": {
                "answer_type": "knowledge",
                "answer": "流式回答",
                "tool_traces": [],
                "citations": [],
                "token_usage": {"total_tokens": 42},
            },
        }

    monkeypatch.setattr(demo_module, "stream_agentic_question", fake_stream)
    application = demo_module.create_demo_app(
        project_root=tmp_path,
        runtime_factory=lambda: runtime,
    )

    with TestClient(application) as client:
        with client.stream(
            "POST",
            "/demo/pilot/stream",
            json={"mode": "agentic"},
        ) as response:
            events = [json.loads(line) for line in response.iter_lines()]

    assert response.status_code == 200
    assert [event["event"] for event in events] == [
        "run_started",
        "tool_started",
        "tool_completed",
        "completed",
    ]
    assert events[-1]["data"]["mode"] == "agentic"
    assert events[-1]["data"]["question"] == question
    assert events[-1]["data"]["token_usage"] == {"total_tokens": 42}
    assert events[-1]["data"]["latency_ms"] >= 0


def test_demo_page_contains_two_mode_switches_only(
    tmp_path: Path,
) -> None:
    """演示页面应显示两个模式，并且不暴露固定多查询选项。"""

    demo_module = importlib.import_module("app.demo")
    _write_demo_question(tmp_path)
    application = demo_module.create_demo_app(
        project_root=tmp_path,
        runtime_factory=lambda: object(),
    )

    with TestClient(application) as client:
        response = client.get("/")

    assert response.status_code == 200
    assert "传统 RAG" in response.text
    assert "Mini-Agent" in response.text
    assert "固定多查询" not in response.text
