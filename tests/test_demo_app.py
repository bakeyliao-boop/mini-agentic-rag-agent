"""长文档渐进式检索演示应用测试。"""

import importlib
import json
from pathlib import Path

from fastapi.testclient import TestClient


DEMO_SPEC_RELATIVE_PATH = Path(
    "evaluation/demos/teacher_water_cycle_consistency_001.json"
)
DEMO_QUESTION = (
    "我准备使用这套水循环资料备课。学校规定课堂和课前准备都不能使用本生灯、"
    "电热板、火柴或制造烟雾，但可以提前两天准备其他常规材料。我希望学生学习单"
    "上的回答确实来自他们看到的现象。请核对《教师演示说明》和《学生学习单》："
    "逐项说明哪些演示可以按原方案保留、哪些只能保留部分步骤、哪些不能按原方案"
    "实施；再把学习单相关题目分为三类：能由保留活动的现场观察支持、因对应演示"
    "取消而失去现场观察依据、本来就不对应这些演示或仅凭保留活动不足以回答。"
    "资料没有给出等效替代办法时请写“未明确”，不要自行设计新实验。"
)
DEMO_CORPUS_ROOT = Path(
    "evaluation/fixtures/teacher_water_cycle_consistency_001/corpus"
)
DEMO_PERSIST_DIRECTORY = Path(
    "data/pilots/teacher-water-cycle-consistency-001"
)
DEMO_SOURCE_PATHS = [
    "/教师备课/科学/水循环/教师指南.md",
    "/教师备课/科学/水循环/教师演示说明.md",
    "/教师备课/科学/水循环/学生学习单.md",
    "/教师备课/科学/水循环/前后测.md",
]


def _write_demo_question(project_root: Path) -> str:
    """准备水循环 Demo 的唯一规格，避免题面和运行语料分别配置。"""

    spec_path = project_root / DEMO_SPEC_RELATIVE_PATH
    spec_path.parent.mkdir(parents=True)
    spec_path.write_text(
        json.dumps(
            {
                "version": 1,
                "id": "teacher-water-cycle-consistency-001",
                "title": "水循环备课资料一致性审查",
                "chat_model": "qwen3.7-flash",
                "retrieval_policy": "locate_first",
                "question": DEMO_QUESTION,
                "corpus": {
                    "root": DEMO_CORPUS_ROOT.as_posix(),
                    "isolation": "declared_documents",
                    "source_paths": DEMO_SOURCE_PATHS,
                },
                "persist_directory": DEMO_PERSIST_DIRECTORY.as_posix(),
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return DEMO_QUESTION


def _fake_runtime(demo_module, knowledge_root: Path):
    """构造能承载定位优先策略的最小 AgenticRuntime 测试桩。"""

    return demo_module.AgenticRuntime(
        knowledge_root=knowledge_root,
        vector_store=object(),
        chat_model=object(),
        path_snapshot=[],
    )


def test_demo_config_exposes_only_traditional_and_agentic_modes(
    tmp_path: Path,
) -> None:
    """前端配置只应提供传统 RAG 与 Mini-Agent 两个选项。"""

    demo_module = importlib.import_module("app.demo")
    question = _write_demo_question(tmp_path)
    runtime = _fake_runtime(demo_module, tmp_path)
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


def test_demo_default_configuration_uses_water_cycle_question_and_policy(
    tmp_path: Path,
) -> None:
    """默认 Demo 应读取水循环题面，并把定位优先策略应用到运行时。"""

    demo_module = importlib.import_module("app.demo")
    question = _write_demo_question(tmp_path)
    runtime = demo_module.AgenticRuntime(
        knowledge_root=tmp_path,
        vector_store=object(),
        chat_model=object(),
        path_snapshot=[],
    )
    application = demo_module.create_demo_app(
        project_root=tmp_path,
        runtime_factory=lambda: runtime,
    )

    with TestClient(application) as client:
        response = client.get("/demo/pilot/config")
        active_runtime = client.app.state.pilot_runtime

    assert response.status_code == 200
    assert response.json()["question"] == question
    assert active_runtime.retrieval_policy == "locate_first"


def test_demo_default_runtime_receives_same_spec_index_and_policy(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """默认构建器应使用题面同一规格中的语料、索引与策略。"""

    demo_module = importlib.import_module("app.demo")
    _write_demo_question(tmp_path)
    settings = {"DASHSCOPE_API_KEY": "test-key"}
    received: dict[str, object] = {}
    runtime = demo_module.AgenticRuntime(
        knowledge_root=tmp_path,
        vector_store=object(),
        chat_model=object(),
        path_snapshot=[],
    )

    monkeypatch.setattr(
        demo_module,
        "load_settings_from_env",
        lambda project_root: settings,
    )

    def fake_build_runtime(**options: object):
        received.update(options)
        return runtime

    monkeypatch.setattr(
        demo_module,
        "build_pilot_agentic_runtime_from_project",
        fake_build_runtime,
    )
    application = demo_module.create_demo_app(project_root=tmp_path)

    with TestClient(application) as client:
        active_runtime = client.app.state.pilot_runtime

    assert received == {
        "project_root": tmp_path.resolve(strict=False),
        "settings": settings,
        "spec_path": (tmp_path / DEMO_SPEC_RELATIVE_PATH).resolve(
            strict=False
        ),
        "persist_directory": (tmp_path / DEMO_PERSIST_DIRECTORY).resolve(
            strict=False
        ),
    }
    assert active_runtime.retrieval_policy == "locate_first"


def test_demo_rejects_unexposed_mode_and_question_override(
    tmp_path: Path,
) -> None:
    """接口不接受固定多查询模式，也不允许前端覆盖冻结问题。"""

    demo_module = importlib.import_module("app.demo")
    _write_demo_question(tmp_path)
    application = demo_module.create_demo_app(
        project_root=tmp_path,
        runtime_factory=lambda: _fake_runtime(demo_module, tmp_path),
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
    runtime = _fake_runtime(demo_module, tmp_path)
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
    runtime = _fake_runtime(demo_module, tmp_path)
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
    runtime = _fake_runtime(demo_module, tmp_path)
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
    runtime = _fake_runtime(demo_module, tmp_path)

    def fake_stream(**options: object):
        assert options["question"] == question
        active_runtime = options["runtime"]
        assert isinstance(active_runtime, demo_module.AgenticRuntime)
        assert active_runtime.knowledge_root == runtime.knowledge_root
        assert active_runtime.vector_store is runtime.vector_store
        assert active_runtime.chat_model is runtime.chat_model
        assert active_runtime.retrieval_policy == "locate_first"
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
        runtime_factory=lambda: _fake_runtime(demo_module, tmp_path),
    )

    with TestClient(application) as client:
        response = client.get("/")

    assert response.status_code == 200
    assert "传统 RAG" in response.text
    assert "Mini-Agent" in response.text
    assert "固定多查询" not in response.text
    assert "课前资源一致性审查" in response.text
    assert "课前资源一致性问题" in response.text
    assert "水循环教学资料" in response.text
    assert "办事指南" not in response.text
    assert "固定长文档" not in response.text
