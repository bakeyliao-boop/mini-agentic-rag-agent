import json
from pathlib import Path

import pytest

from app import evaluation
from app.evaluation import load_evaluation_questions
from app.traditional_rag import TraditionalRagConfig


def test_load_evaluation_questions_reads_fixed_dataset() -> None:
    """应读取 education-v1 对应的 10 道固定评测问题。"""

    project_root = Path(__file__).parent.parent
    source_path = project_root / "evaluation" / "questions.json"

    result = load_evaluation_questions(source_path)

    assert result["version"] == 2
    assert result["corpus_id"] == "education-v1"
    assert len(result["questions"]) == 10


def test_load_evaluation_questions_rejects_wrong_version(
    tmp_path: Path,
) -> None:
    """评测版本不是 2 时应抛出 ValueError。"""

    source_path = tmp_path / "questions.json"
    source_path.write_text(
        json.dumps(
            {
                "version": 1,
                "corpus_id": "education-v1",
                "questions": [{} for _ in range(10)],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
        newline="\n",
    )

    with pytest.raises(ValueError, match="version"):
        load_evaluation_questions(source_path)


def test_load_evaluation_questions_rejects_wrong_corpus(
    tmp_path: Path,
) -> None:
    """评测集语料 ID 不是 education-v1 时应抛出 ValueError。"""

    source_path = tmp_path / "questions.json"
    source_path.write_text(
        json.dumps(
            {
                "version": 2,
                "corpus_id": "education-v2",
                "questions": [{} for _ in range(10)],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
        newline="\n",
    )

    with pytest.raises(ValueError, match="corpus_id"):
        load_evaluation_questions(source_path)


@pytest.mark.parametrize("question_count", [9, 11])
def test_load_evaluation_questions_rejects_wrong_question_count(
    tmp_path: Path,
    question_count: int,
) -> None:
    """固定评测集不是正好 10 道问题时应抛出 ValueError。"""

    source_path = tmp_path / "questions.json"
    source_path.write_text(
        json.dumps(
            {
                "version": 2,
                "corpus_id": "education-v1",
                "questions": [{} for _ in range(question_count)],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
        newline="\n",
    )

    with pytest.raises(ValueError, match="exactly 10"):
        load_evaluation_questions(source_path)


def test_load_evaluation_questions_rejects_missing_question_field(
    tmp_path: Path,
) -> None:
    """任一道评测题缺少 question 字段时应抛出 ValueError。"""

    questions = [
        {
            "id": f"exact-{index:03d}",
            "category": "exact_fact",
            "question": f"测试问题 {index}",
            "expected_answer_type": "knowledge",
        }
        for index in range(10)
    ]
    questions[0].pop("question")
    source_path = tmp_path / "questions.json"
    source_path.write_text(
        json.dumps(
            {
                "version": 2,
                "corpus_id": "education-v1",
                "questions": questions,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
        newline="\n",
    )

    with pytest.raises(ValueError, match="question"):
        load_evaluation_questions(source_path)


def test_load_evaluation_questions_rejects_duplicate_ids(
    tmp_path: Path,
) -> None:
    """固定评测集中出现重复问题 ID 时应抛出 ValueError。"""

    questions = [
        {
            "id": f"exact-{index:03d}",
            "category": "exact_fact",
            "question": f"测试问题 {index}",
            "expected_answer_type": "knowledge",
        }
        for index in range(10)
    ]
    questions[1]["id"] = questions[0]["id"]
    source_path = tmp_path / "questions.json"
    source_path.write_text(
        json.dumps(
            {
                "version": 2,
                "corpus_id": "education-v1",
                "questions": questions,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
        newline="\n",
    )

    with pytest.raises(ValueError, match="duplicate"):
        load_evaluation_questions(source_path)


def test_run_traditional_baseline_runs_each_question_and_collects_results(
    monkeypatch,
) -> None:
    """baseline runner 应逐题调用传统 RAG 并汇总评测结果。"""

    dataset = {
        "version": 2,
        "corpus_id": "education-v1",
        "questions": [
            {
                "id": "exact-001",
                "category": "exact_fact",
                "question": "智慧农场如何灌溉？",
                "expected_answer_type": "knowledge",
            },
            {
                "id": "exact-002",
                "category": "exact_fact",
                "question": "气象站有什么作用？",
                "expected_answer_type": "knowledge",
            },
        ],
    }
    vector_store = object()
    chat_model = object()
    config = TraditionalRagConfig()
    received_questions: list[str] = []

    def fake_answer_with_traditional_rag(
        question: str,
        vector_store: object,
        chat_model: object,
        config: TraditionalRagConfig,
    ) -> dict[str, object]:
        received_questions.append(question)
        return {
            "answer": f"回答：{question}",
            "hits": [],
            "latency_ms": 10.0,
            "token_usage": {"total_tokens": 20},
        }

    monkeypatch.setattr(
        evaluation,
        "answer_with_traditional_rag",
        fake_answer_with_traditional_rag,
        raising=False,
    )

    result = evaluation.run_traditional_baseline(
        dataset=dataset,
        vector_store=vector_store,
        chat_model=chat_model,
        config=config,
    )

    assert received_questions == [
        "智慧农场如何灌溉？",
        "气象站有什么作用？",
    ]
    assert result["config"] == {
        "model": "qwen3.6-flash",
        "temperature": 0,
        "enable_thinking": False,
        "top_k": 5,
        "corpus_version": "education-v1",
    }
    assert [item["id"] for item in result["results"]] == [
        "exact-001",
        "exact-002",
    ]
    assert [item["answer"] for item in result["results"]] == [
        "回答：智慧农场如何灌溉？",
        "回答：气象站有什么作用？",
    ]


def test_run_agentic_evaluation_runs_each_question_and_collects_results() -> None:
    """Agentic 评测应逐题执行，并汇总引用、工具轨迹和耗时。"""

    dataset = {
        "version": 2,
        "corpus_id": "education-v1",
        "questions": [
            {
                "id": "exact-001",
                "category": "exact_fact",
                "question": "智慧农场如何灌溉？",
                "expected_answer_type": "knowledge",
            },
            {
                "id": "outside-001",
                "category": "out_of_scope",
                "question": "知识库介绍量子计算机了吗？",
                "expected_answer_type": "insufficient",
            },
        ],
    }
    received_calls: list[tuple[str, str]] = []
    timestamps = iter([1.0, 1.125, 2.0, 2.25])

    def fake_run_question(question: str, thread_id: str) -> dict[str, object]:
        """
        模拟实现 run_question()，返回固定答案和工具调用轨迹。
        LLM + ls + search + read + EvidenceRegistry
        """
        received_calls.append((question, thread_id))
        if thread_id == "evaluation-exact-001":
            return {
                "answer_type": "knowledge",
                "answer": "系统会根据不同农作物分类灌溉。",
                "citations": [
                    {
                        "path": "/智慧农场.md",
                        "start_line": 1,
                        "end_line": 1,
                        "quote": "系统会根据不同农作物分类灌溉。",
                    }
                ],
                "tool_traces": [
                    {"step": 1, "name": "search", "status": "success"},
                    {"step": 2, "name": "read", "status": "success"},
                ],
            }
        return {
            "answer_type": "insufficient",
            "answer": "当前证据不足。",
            "citations": [],
            "tool_traces": [
                {"step": 1, "name": "search", "status": "success"}
            ],
        }

    result = evaluation.run_agentic_evaluation(
        dataset=dataset,
        run_question=fake_run_question,
        config=TraditionalRagConfig(),
        prompt_version="Prompt-V1.2",
        clock=lambda: next(timestamps),
    )

    assert received_calls == [
        ("智慧农场如何灌溉？", "evaluation-exact-001"),
        ("知识库介绍量子计算机了吗？", "evaluation-outside-001"),
    ]
    assert result["config"] == {
        "model": "qwen3.6-flash",
        "temperature": 0,
        "enable_thinking": False,
        "corpus_version": "education-v1",
        "prompt_version": "Prompt-V1.2",
    }
    assert [item["id"] for item in result["results"]] == [
        "exact-001",
        "outside-001",
    ]
    assert [item["answer_type"] for item in result["results"]] == [
        "knowledge",
        "insufficient",
    ]
    assert [item["latency_ms"] for item in result["results"]] == [
        125.0,
        250.0,
    ]
    assert [item["tool_call_count"] for item in result["results"]] == [2, 1]
    assert result["results"][0]["citations"][0]["path"] == (
        "/智慧农场.md"
    )


def test_save_evaluation_result_writes_readable_utf8_json(
    tmp_path: Path,
) -> None:
    """评测结果应保存为可重新读取的 UTF-8 JSON。"""

    result = {
        "version": 2,
        "corpus_id": "education-v1",
        "results": [
            {
                "id": "exact-001",
                "question": "智慧农场如何灌溉？",
                "answer": "使用自动灌溉系统。",
            }
        ],
    }
    output_path = tmp_path / "evaluation" / "results" / "baseline.json"

    evaluation.save_evaluation_result(result, output_path)

    assert output_path.is_file()
    saved_result = json.loads(output_path.read_text(encoding="utf-8"))
    assert saved_result == result
