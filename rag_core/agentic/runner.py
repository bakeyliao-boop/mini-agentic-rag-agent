"""Agentic RAG 的运行环境、单题执行与结果收口。"""

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from rag_core.agentic.agent import (
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
    knowledge_agent = build_knowledge_agent(runtime.chat_model, tools)

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
