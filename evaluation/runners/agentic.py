"""Agentic RAG 固定评测执行流程。"""

from collections.abc import Callable
from time import perf_counter

from rag_core.traditional.service import TraditionalRagConfig


def run_agentic_evaluation(
    dataset: dict[str, object],
    run_question: Callable[[str, str], dict[str, object]],
    config: TraditionalRagConfig,
    prompt_version: str,
    clock: Callable[[], float] = perf_counter,
) -> dict[str, object]:
    """按固定顺序运行所有问题，并汇总 Agentic RAG 评测结果。"""

    questions = dataset.get("questions")
    if not isinstance(questions, list):
        raise ValueError("evaluation questions must be a list")

    results: list[dict[str, object]] = []
    for question_data in questions:
        if not isinstance(question_data, dict):
            raise ValueError("each evaluation question must be an object")

        question_id = question_data["id"]
        question = question_data["question"]
        thread_id = f"evaluation-{question_id}"
        start_time = clock()
        agent_result = run_question(question, thread_id)
        latency_ms = (clock() - start_time) * 1000.0

        tool_traces = agent_result.get("tool_traces", [])
        if not isinstance(tool_traces, list):
            tool_traces = []

        results.append(
            {
                "id": question_id,
                "category": question_data["category"],
                "question": question,
                "expected_answer_type": question_data[
                    "expected_answer_type"
                ],
                **agent_result,
                "thread_id": thread_id,
                "latency_ms": latency_ms,
                "tool_call_count": len(tool_traces),
            }
        )

    return {
        "version": dataset.get("version"),
        "corpus_id": dataset.get("corpus_id"),
        "config": {
            "model": config.model,
            "temperature": config.temperature,
            "enable_thinking": config.enable_thinking,
            "corpus_version": config.corpus_version,
            "prompt_version": prompt_version,
        },
        "results": results,
    }
