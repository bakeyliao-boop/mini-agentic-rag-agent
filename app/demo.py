"""长文档渐进式检索面试演示应用。"""

import json
from collections.abc import Callable
from contextlib import asynccontextmanager
from dataclasses import replace
from pathlib import Path
from time import perf_counter
from typing import Literal
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from evaluation.cli.pilot_agentic import (
    build_pilot_agentic_runtime_from_project,
)
from rag_core.agentic.runner import (
    AgenticRuntime,
    run_agentic_question,
    stream_agentic_question,
)
from rag_core.settings import load_settings_from_env
from rag_core.traditional.service import answer_with_traditional_rag

PROJECT_ROOT = Path(__file__).resolve().parent.parent
STATIC_DIRECTORY = Path(__file__).resolve().parent / "static" / "pilot_demo"
DEMO_QUESTION_RELATIVE_PATH = Path(
    "evaluation/demos/patent_review_comparison_001.json"
)

DemoMode = Literal["traditional", "agentic"]
DemoRuntimeFactory = Callable[[], AgenticRuntime]


class PilotDemoConfigResponse(BaseModel):
    """前端初始化时需要的固定问题和可选模式。"""

    question: str
    modes: list[DemoMode]


class PilotDemoRunRequest(BaseModel):
    """演示运行请求；问题由服务端独立的 Demo 配置提供。"""

    model_config = ConfigDict(extra="forbid")

    mode: DemoMode


class PilotDemoRunResponse(BaseModel):
    """传统 RAG 与 Mini-Agent 共用的演示响应结构。"""

    mode: DemoMode
    question: str
    answer_type: Literal[
        "knowledge",
        "directory",
        "conversation",
        "insufficient",
    ]
    answer: str
    retrieval_hits: list[dict[str, object]] = Field(default_factory=list)
    tool_traces: list[dict[str, object]] = Field(default_factory=list)
    citations: list[dict[str, object]] = Field(default_factory=list)
    token_usage: dict[str, object] = Field(default_factory=dict)
    latency_ms: float = Field(ge=0)
    finish_reason: Literal["completed", "insufficient"]


def _load_demo_configuration(project_root: Path) -> tuple[str, str]:
    """读取题面和显式策略配置，不从问题措辞推断检索流程。"""

    spec_path = project_root / DEMO_QUESTION_RELATIVE_PATH
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    if not isinstance(spec, dict):
        raise ValueError("demo question configuration must be an object")
    question = spec.get("question")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("demo question must be a non-empty string")
    policy = spec.get("retrieval_policy", "free")
    if policy not in ("free", "locate_first"):
        raise ValueError("unknown retrieval_policy")
    return question.strip(), policy


def _dictionary_list(value: object) -> list[dict[str, object]]:
    """只保留可安全返回给前端的字典列表。"""

    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _token_usage(value: object) -> dict[str, object]:
    """将模型 Token 信息整理为稳定字典。"""

    return value if isinstance(value, dict) else {}


def create_demo_app(
    project_root: Path = PROJECT_ROOT,
    runtime_factory: DemoRuntimeFactory | None = None,
) -> FastAPI:
    """创建只服务冻结长文档 Pilot 的双模式演示应用。"""

    resolved_project_root = project_root.resolve(strict=False)

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        question, policy = _load_demo_configuration(resolved_project_root)
        if runtime_factory is None:
            settings = load_settings_from_env(resolved_project_root)
            runtime = build_pilot_agentic_runtime_from_project(
                project_root=resolved_project_root,
                settings=settings,
            )
        else:
            runtime = runtime_factory()
        if isinstance(runtime, AgenticRuntime):
            runtime = replace(runtime, retrieval_policy=policy)
        elif policy != "free":
            raise TypeError("retrieval policy requires an AgenticRuntime")
        application.state.pilot_question = question
        application.state.pilot_runtime = runtime
        yield

    application = FastAPI(
        title="Mini-Agent 渐进式检索演示",
        lifespan=lifespan,
    )
    application.mount(
        "/demo-assets",
        StaticFiles(directory=STATIC_DIRECTORY),
        name="demo-assets",
    )

    @application.get("/", response_class=FileResponse)
    def demo_page() -> FileResponse:
        return FileResponse(STATIC_DIRECTORY / "index.html")

    @application.get(
        "/demo/pilot/config",
        response_model=PilotDemoConfigResponse,
    )
    def demo_config(request: Request) -> dict[str, object]:
        return {
            "question": request.app.state.pilot_question,
            "modes": ["traditional", "agentic"],
        }

    @application.post(
        "/demo/pilot/run",
        response_model=PilotDemoRunResponse,
    )
    def run_demo(
        payload: PilotDemoRunRequest,
        request: Request,
    ) -> dict[str, object]:
        runtime = request.app.state.pilot_runtime
        question = request.app.state.pilot_question
        started_at = perf_counter()

        if payload.mode == "traditional":
            result = answer_with_traditional_rag(
                question=question,
                vector_store=runtime.vector_store,
                chat_model=runtime.chat_model,
                path="/",
            )
            answer = result.get("answer")
            resolved_answer = answer if isinstance(answer, str) else ""
            answer_type = (
                "knowledge" if resolved_answer.strip() else "insufficient"
            )
            return {
                "mode": payload.mode,
                "question": question,
                "answer_type": answer_type,
                "answer": resolved_answer or "当前检索没有返回可用答案。",
                "retrieval_hits": _dictionary_list(result.get("hits")),
                "tool_traces": [],
                "citations": [],
                "token_usage": _token_usage(result.get("token_usage")),
                "latency_ms": (perf_counter() - started_at) * 1_000,
                "finish_reason": (
                    "completed"
                    if answer_type == "knowledge"
                    else "insufficient"
                ),
            }

        result = run_agentic_question(
            runtime=runtime,
            question=question,
            thread_id=f"pilot-demo-{uuid4().hex}",
        )
        answer_type = result.get("answer_type")
        if answer_type not in {
            "knowledge",
            "directory",
            "conversation",
            "insufficient",
        }:
            answer_type = "insufficient"
        answer = result.get("answer")
        return {
            "mode": payload.mode,
            "question": question,
            "answer_type": answer_type,
            "answer": (
                answer
                if isinstance(answer, str) and answer
                else "当前证据不足，无法从知识库确定答案。"
            ),
            "retrieval_hits": [],
            "tool_traces": _dictionary_list(result.get("tool_traces")),
            "citations": _dictionary_list(result.get("citations")),
            "token_usage": _token_usage(result.get("token_usage")),
            "latency_ms": (perf_counter() - started_at) * 1_000,
            "finish_reason": (
                "insufficient"
                if answer_type == "insufficient"
                else "completed"
            ),
        }

    @application.post("/demo/pilot/stream")
    def stream_demo(
        payload: PilotDemoRunRequest,
        request: Request,
    ) -> StreamingResponse:
        """以 NDJSON 实时返回 Mini-Agent 的真实工具调用事件。"""

        if payload.mode != "agentic":
            return StreamingResponse(
                iter(
                    [
                        json.dumps(
                            {
                                "event": "error",
                                "data": {
                                    "message": "stream endpoint only supports agentic mode"
                                },
                            }
                        )
                        + "\n"
                    ]
                ),
                status_code=400,
                media_type="application/x-ndjson",
            )

        runtime = request.app.state.pilot_runtime
        question = request.app.state.pilot_question
        thread_id = f"pilot-demo-{uuid4().hex}"

        def event_stream():
            started_at = perf_counter()
            try:
                for event in stream_agentic_question(
                    runtime=runtime,
                    question=question,
                    thread_id=thread_id,
                ):
                    if event.get("event") == "completed":
                        data = event.get("data")
                        result = data if isinstance(data, dict) else {}
                        answer_type = result.get("answer_type")
                        event = {
                            "event": "completed",
                            "data": {
                                "mode": "agentic",
                                "question": question,
                                "answer_type": answer_type,
                                "answer": result.get("answer"),
                                "retrieval_hits": [],
                                "tool_traces": _dictionary_list(
                                    result.get("tool_traces")
                                ),
                                "citations": _dictionary_list(
                                    result.get("citations")
                                ),
                                "token_usage": _token_usage(
                                    result.get("token_usage")
                                ),
                                "latency_ms": (
                                    perf_counter() - started_at
                                )
                                * 1_000,
                                "finish_reason": (
                                    "insufficient"
                                    if answer_type == "insufficient"
                                    else "completed"
                                ),
                            },
                        }
                    yield json.dumps(event, ensure_ascii=False) + "\n"
            except Exception:
                yield json.dumps(
                    {
                        "event": "error",
                        "data": {
                            "message": "本次运行未完成，请检查服务日志。"
                        },
                    },
                    ensure_ascii=False,
                ) + "\n"

        return StreamingResponse(
            event_stream(),
            media_type="application/x-ndjson",
            headers={"Cache-Control": "no-store"},
        )

    return application


app = create_demo_app()
