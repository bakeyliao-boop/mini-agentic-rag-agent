"""Agentic RAG 的运行环境、单题执行与结果收口。"""

import json
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage

from rag_core.agentic.agent import (
    KNOWLEDGE_TOOL_NAMES,
    RetrievalPolicy,
    build_knowledge_agent,
    extract_token_usage,
    extract_tool_traces,
)
from rag_core.settings import _required_setting
from rag_core.agentic.evidence import (
    EvidenceRegistry,
    build_citations,
    validate_answer_evidence,
    validate_evidence_sources,
)
from rag_core.knowledge.indexer import (
    build_dashscope_embeddings,
    build_knowledge_index,
)
from rag_core.knowledge.store import build_knowledge_path_snapshot
from rag_core.models import GroundedAnswer
from rag_core.agentic.tools import build_knowledge_tools
from rag_core.traditional.service import (
    TraditionalRagConfig,
    build_traditional_chat_model,
    resolve_traditional_corpus_root,
)


@dataclass(frozen=True)
class AgenticRuntime:
    """可在同一批问题之间复用的知识库、索引和对话模型。"""

    knowledge_root: Path
    vector_store: object
    chat_model: object
    path_snapshot: list[dict[str, str]]
    retrieval_policy: RetrievalPolicy = "free"


def finalize_grounded_answer(
    structured_response: GroundedAnswer,
    evidence_registry: EvidenceRegistry,
    knowledge_root: Path,
) -> dict[str, object]:
    """校验证据并生成最终回答；校验失败时降级为证据不足。"""

    try:
        evidences = validate_answer_evidence(
            structured_response,
            evidence_registry,
        )
        validated_evidences = validate_evidence_sources(
            evidences,
            knowledge_root,
        )
    except ValueError:
        return {
            "answer_type": "insufficient",
            "answer": "当前证据不足，无法从知识库确定答案。",
            "citations": [],
        }

    citations = build_citations(validated_evidences)
    return {
        "answer_type": structured_response.answer_type,
        "answer": structured_response.answer,
        "citations": [citation.model_dump() for citation in citations],
    }


def finalize_agent_result(
    agent_result: Mapping[str, object],
    evidence_registry: EvidenceRegistry,
    knowledge_root: Path,
) -> dict[str, object]:
    """处理 Agent 结果；缺少结构化回答时降级为证据不足。"""

    structured_response = agent_result.get("structured_response")
    if not isinstance(structured_response, GroundedAnswer):
        return {
            "answer_type": "insufficient",
            "answer": "当前证据不足，无法从知识库确定答案。",
            "citations": [],
        }

    return finalize_grounded_answer(
        structured_response,
        evidence_registry,
        knowledge_root,
    )


def build_agentic_runtime_from_project(
    project_root: Path,
    settings: Mapping[str, str],
) -> AgenticRuntime:
    """构建可供多道问题共享的知识库索引和对话模型。"""

    config = TraditionalRagConfig()
    api_key = _required_setting(settings, "DASHSCOPE_API_KEY")
    base_url = _required_setting(settings, "DASHSCOPE_BASE_URL")
    embedding_model = _required_setting(settings, "EMBEDDING_MODEL")
    embedding_dimensions = int(
        _required_setting(settings, "EMBEDDING_DIMENSIONS")
    )
    embedding_batch_size = int(
        _required_setting(settings, "EMBEDDING_BATCH_SIZE")
    )
    persist_setting = _required_setting(settings, "CHROMA_PERSIST_DIR")

    knowledge_root = resolve_traditional_corpus_root(project_root, config)
    path_snapshot = build_knowledge_path_snapshot(knowledge_root)
    persist_directory = (
        project_root / Path(persist_setting)
    ).resolve(strict=False)
    embeddings = build_dashscope_embeddings(
        model=embedding_model,
        dimensions=embedding_dimensions,
        batch_size=embedding_batch_size,
        api_key=api_key,
        base_url=base_url,
    )
    vector_store = build_knowledge_index(
        knowledge_root,
        persist_directory,
        embeddings,
    )
    chat_model = build_traditional_chat_model(
        config,
        api_key,
        base_url,
    )

    return AgenticRuntime(
        knowledge_root=knowledge_root,
        vector_store=vector_store,
        chat_model=chat_model,
        path_snapshot=path_snapshot,
    )


def run_agentic_question(
    runtime: AgenticRuntime,
    question: str,
    thread_id: str,
) -> dict[str, object]:
    """使用共享运行环境执行一道问题，并创建本题独立证据注册表。"""

    evidence_registry = EvidenceRegistry(
        run_id=f"{thread_id}:{uuid4().hex}",
    )
    tools = build_knowledge_tools(
        runtime.knowledge_root,
        runtime.vector_store,
        evidence_registry,
        path_snapshot=runtime.path_snapshot,
    )
    if runtime.retrieval_policy == "free":
        knowledge_agent = build_knowledge_agent(runtime.chat_model, tools)
    else:
        knowledge_agent = build_knowledge_agent(
            runtime.chat_model, tools, retrieval_policy=runtime.retrieval_policy,
        )

    agent_result = knowledge_agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": question,
                }
            ]
        },
        config={"configurable": {"thread_id": thread_id}},
    )

    messages = agent_result["messages"]
    finalized_answer = finalize_agent_result(
        agent_result,
        evidence_registry,
        runtime.knowledge_root,
    )

    return {
        **finalized_answer,
        "tool_traces": extract_tool_traces(messages),
        "token_usage": extract_token_usage(messages),
        "thread_id": thread_id,
    }


def _stream_update_messages(part: object) -> list[BaseMessage]:
    """从 LangGraph v2 updates 事件中提取本轮新增消息。"""

    if not isinstance(part, dict) or part.get("type") != "updates":
        return []
    data = part.get("data")
    if not isinstance(data, dict):
        return []

    messages: list[BaseMessage] = []
    for node_update in data.values():
        if not isinstance(node_update, dict):
            continue
        value = node_update.get("messages")
        candidates = value if isinstance(value, list) else [value]
        messages.extend(
            message
            for message in candidates
            if isinstance(message, BaseMessage)
        )
    return messages


def _tool_result_summary(message: ToolMessage) -> dict[str, object]:
    """将工具返回压缩成适合实时展示的摘要，避免重复发送正文。"""

    content = message.content
    if not isinstance(content, str):
        return {}
    try:
        payload = json.loads(content)
    except (TypeError, ValueError):
        return {"message": content[:200]} if message.status == "error" else {}
    if not isinstance(payload, dict):
        return {}

    if message.name == "search":
        hits = payload.get("hits")
        return {
            "hit_count": len(hits) if isinstance(hits, list) else 0,
            "retrieval_status": payload.get("retrieval_status"),
        }
    if message.name == "read":
        lines = payload.get("lines")
        line_entries = lines if isinstance(lines, list) else []
        line_numbers = [
            line.get("line")
            for line in line_entries
            if isinstance(line, dict) and isinstance(line.get("line"), int)
        ]
        return {
            "path": payload.get("path"),
            "start_line": line_numbers[0] if line_numbers else None,
            "end_line": line_numbers[-1] if line_numbers else None,
            "next_line": payload.get("next_line"),
        }
    if message.name == "grep":
        matches = payload.get("matches")
        return {
            "match_count": len(matches) if isinstance(matches, list) else 0,
            "truncated": payload.get("truncated", False),
        }
    if message.name == "glob":
        matches = payload.get("matches")
        return {
            "match_count": len(matches) if isinstance(matches, list) else 0
        }
    if message.name == "ls":
        entries = payload.get("entries")
        return {
            "entry_count": len(entries) if isinstance(entries, list) else 0
        }
    return {}


def stream_agentic_question(
    runtime: AgenticRuntime,
    question: str,
    thread_id: str,
) -> Iterator[dict[str, object]]:
    """执行 Agent，并逐步产生真实工具调用、完成状态与最终结果。"""

    evidence_registry = EvidenceRegistry(
        run_id=f"{thread_id}:{uuid4().hex}",
    )
    tools = build_knowledge_tools(
        runtime.knowledge_root,
        runtime.vector_store,
        evidence_registry,
        path_snapshot=runtime.path_snapshot,
    )
    if runtime.retrieval_policy == "free":
        knowledge_agent = build_knowledge_agent(runtime.chat_model, tools)
    else:
        knowledge_agent = build_knowledge_agent(
            runtime.chat_model, tools, retrieval_policy=runtime.retrieval_policy,
        )
    config = {"configurable": {"thread_id": thread_id}}
    final_state: Mapping[str, object] | None = None
    steps_by_call_id: dict[str, int] = {}
    names_by_call_id: dict[str, str] = {}
    completed_call_ids: set[str] = set()

    yield {
        "event": "run_started",
        "data": {"thread_id": thread_id},
    }

    for part in knowledge_agent.stream(
        {
            "messages": [
                {
                    "role": "user",
                    "content": question,
                }
            ]
        },
        config=config,
        stream_mode=["updates", "values"],
        version="v2",
    ):
        if isinstance(part, dict) and part.get("type") == "values":
            data = part.get("data")
            if isinstance(data, Mapping):
                final_state = data
            continue

        for message in _stream_update_messages(part):
            if isinstance(message, AIMessage):
                for tool_call in message.tool_calls:
                    name = tool_call.get("name")
                    tool_call_id = tool_call.get("id")
                    if (
                        name not in KNOWLEDGE_TOOL_NAMES
                        or not isinstance(tool_call_id, str)
                        or tool_call_id in steps_by_call_id
                    ):
                        continue
                    step = len(steps_by_call_id) + 1
                    steps_by_call_id[tool_call_id] = step
                    names_by_call_id[tool_call_id] = name
                    yield {
                        "event": "tool_started",
                        "data": {
                            "step": step,
                            "tool_call_id": tool_call_id,
                            "name": name,
                            "args": tool_call.get("args") or {},
                            "status": "running",
                        },
                    }
            elif isinstance(message, ToolMessage):
                step = steps_by_call_id.get(message.tool_call_id)
                name = names_by_call_id.get(message.tool_call_id)
                if step is None or message.tool_call_id in completed_call_ids:
                    continue
                completed_call_ids.add(message.tool_call_id)
                # tool_call_id 是配对依据，部分工具结果没有重复携带工具名。
                named_message = message.model_copy(update={"name": name})
                yield {
                    "event": "tool_completed",
                    "data": {
                        "step": step,
                        "tool_call_id": message.tool_call_id,
                        "name": name,
                        "status": message.status or "success",
                        "summary": _tool_result_summary(named_message),
                    },
                }

    if final_state is None:
        raise RuntimeError("Agent stream ended without a final state")
    messages = final_state.get("messages")
    if not isinstance(messages, list):
        raise RuntimeError("Agent final state does not contain messages")
    finalized_answer = finalize_agent_result(
        final_state,
        evidence_registry,
        runtime.knowledge_root,
    )
    yield {
        "event": "completed",
        "data": {
            **finalized_answer,
            "tool_traces": extract_tool_traces(messages),
            "token_usage": extract_token_usage(messages),
            "thread_id": thread_id,
        },
    }


def run_agentic_question_from_project(
    project_root: Path,
    question: str,
    thread_id: str,
    settings: Mapping[str, str],
) -> dict[str, object]:
    """构建项目运行环境，并执行一个知识库问题。"""

    runtime = build_agentic_runtime_from_project(
        project_root=project_root,
        settings=settings,
    )
    return run_agentic_question(
        runtime=runtime,
        question=question,
        thread_id=thread_id,
    )
