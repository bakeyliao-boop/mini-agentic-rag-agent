"""多 Chunk Pilot 的 A 组运行流程。"""

from langchain_chroma import Chroma

from rag_core.knowledge.indexer import search_chroma_index
from rag_core.traditional.service import (
    TraditionalRagConfig,
    answer_with_traditional_rag,
)


def _load_arm_a(spec: dict[str, object]) -> dict[str, object]:
    """从 Pilot 规范中读取固定的 A 组配置。"""

    arms = spec.get("comparison_arms")
    if not isinstance(arms, list):
        raise ValueError("pilot comparison_arms must be a list")

    for arm in arms:
        if isinstance(arm, dict) and arm.get("id") == "A":
            return arm
    raise ValueError("pilot comparison arm A is missing")


def _load_arm_a_config(spec: dict[str, object]) -> tuple[str, int]:
    """读取并校验 A 组固定配置：单次 Top-5、完整问题检索。"""

    arm = _load_arm_a(spec)
    arm_name = arm.get("name")
    retrieval = arm.get("retrieval")
    if not isinstance(arm_name, str) or not arm_name.strip():
        raise ValueError("pilot arm A name must be a non-empty string")
    if not isinstance(retrieval, dict):
        raise ValueError("pilot arm A retrieval must be an object")

    top_k = retrieval.get("top_k_per_query")
    maximum_calls = retrieval.get("maximum_retrieval_calls")
    queries = retrieval.get("queries")
    if top_k != 5 or maximum_calls != 1:
        raise ValueError("pilot arm A must use one Top-5 retrieval call")
    if queries != ["use_full_question"]:
        raise ValueError("pilot arm A must search the full question")
    return arm_name, top_k


def run_pilot_arm_a(
    spec: dict[str, object],
    vector_store: Chroma,
) -> dict[str, object]:
    """对完整 Pilot 问题执行一次固定 Top-5 检索并记录候选。"""

    pilot_id = spec.get("id")
    question = spec.get("question")
    if not isinstance(pilot_id, str) or not pilot_id.strip():
        raise ValueError("pilot id must be a non-empty string")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("pilot question must be a non-empty string")

    arm_name, top_k = _load_arm_a_config(spec)

    search_result = search_chroma_index(
        vector_store=vector_store,
        query=question,
        path="/",
        limit=top_k,
    )
    raw_hits = search_result.get("hits")
    if not isinstance(raw_hits, list):
        raise ValueError("pilot arm A search hits must be a list")

    hits: list[dict[str, object]] = []
    for rank, hit in enumerate(raw_hits, start=1):
        if not isinstance(hit, dict):
            raise ValueError("each pilot search hit must be an object")
        hits.append(
            {
                "rank": rank,
                "path": hit["path"],
                "start_line": hit["start_line"],
                "end_line": hit["end_line"],
                "score": hit["score"],
                "content": hit["preview"],
            }
        )

    return {
        "pilot_id": pilot_id,
        "arm_id": "A",
        "arm_name": arm_name,
        "question": question,
        "retrieval_call_count": 1,
        "retrieved_chunk_count": len(hits),
        "retrieval_status": search_result.get("retrieval_status"),
        "hits": hits,
    }


def run_pilot_arm_a_answer(
    spec: dict[str, object],
    vector_store: Chroma,
    chat_model: object,
    config: TraditionalRagConfig | None = None,
) -> dict[str, object]:
    """对完整 Pilot 问题执行单次 Top-5 检索并调用传统回答链路。

    与 A 组检索实验共用同一份配置校验；回答链路复用
    answer_with_traditional_rag，保证与 education-v1 基线使用相同的
    提示词、检索与模型配置。
    """

    pilot_id = spec.get("id")
    question = spec.get("question")
    if not isinstance(pilot_id, str) or not pilot_id.strip():
        raise ValueError("pilot id must be a non-empty string")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("pilot question must be a non-empty string")

    arm_name, _ = _load_arm_a_config(spec)
    resolved_config = config if config is not None else TraditionalRagConfig()

    rag_result = answer_with_traditional_rag(
        question=question,
        vector_store=vector_store,
        chat_model=chat_model,
        config=resolved_config,
    )
    hits = rag_result.get("hits")
    if not isinstance(hits, list):
        raise ValueError("pilot arm A answer hits must be a list")

    return {
        "pilot_id": pilot_id,
        "arm_id": "A",
        "arm_name": arm_name,
        "question": question,
        "answer": rag_result.get("answer"),
        "hits": hits,
        "latency_ms": rag_result.get("latency_ms"),
        "token_usage": rag_result.get("token_usage"),
        "retrieval_call_count": 1,
        "retrieved_chunk_count": len(hits),
        "retrieval_status": rag_result.get("retrieval_status"),
    }


