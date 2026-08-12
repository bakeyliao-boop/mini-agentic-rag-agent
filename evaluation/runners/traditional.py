"""传统 RAG 固定评测执行流程。"""

from dataclasses import asdict

from langchain_chroma import Chroma

from rag_core.traditional.service import (
    TraditionalRagConfig,
    answer_with_traditional_rag,
)


def run_traditional_baseline(
    dataset: dict[str, object],
    vector_store: Chroma,
    chat_model: object,
    config: TraditionalRagConfig,
) -> dict[str, object]:
    """按固定顺序运行所有问题，并汇总传统 RAG 基线结果。"""

    questions = dataset.get("questions")
    if not isinstance(questions, list):
        raise ValueError("evaluation questions must be a list")

    results: list[dict[str, object]] = []
    for question_data in questions:
        if not isinstance(question_data, dict):
            raise ValueError("each evaluation question must be an object")

        rag_result = answer_with_traditional_rag(
            question=question_data["question"],
            vector_store=vector_store,
            chat_model=chat_model,
            config=config,
        )
        results.append(
            {
                "id": question_data["id"],
                "category": question_data["category"],
                "question": question_data["question"],
                "expected_answer_type": question_data[
                    "expected_answer_type"
                ],
                **rag_result,
            }
        )

    return {
        "version": dataset.get("version"),
        "corpus_id": dataset.get("corpus_id"),
        "config": asdict(config),
        "results": results,
    }
