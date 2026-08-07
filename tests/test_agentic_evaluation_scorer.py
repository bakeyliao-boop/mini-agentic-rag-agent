import json
from importlib import import_module
from pathlib import Path


def test_score_agentic_evaluation_calculates_grounded_agent_metrics(
    tmp_path: Path,
) -> None:
    """Agentic 评分器应汇总回答类型、引用、工具和成本指标。"""

    scorer = import_module("app.agentic_evaluation_scorer")
    knowledge_root = tmp_path / "knowledge"
    source_path = knowledge_root / "课程资源" / "智慧农场.md"
    source_path.parent.mkdir(parents=True)
    source_path.write_text(
        "系统根据不同农作物进行分类灌溉。\n",
        encoding="utf-8",
    )
    dataset = {
        "version": 2,
        "corpus_id": "education-v1",
        "questions": [
            {
                "id": "exact-001",
                "expected_answer_type": "knowledge",
                "expected_paths": ["/课程资源/智慧农场.md"],
                "answer_points": ["分类灌溉"],
            },
            {
                "id": "directory-001",
                "expected_answer_type": "directory",
                "expected_paths": ["/课程资源/初中", "/课程资源/小学"],
                "answer_points": ["初中", "小学"],
            },
            {
                "id": "outside-001",
                "expected_answer_type": "insufficient",
                "expected_paths": [],
                "answer_points": [],
            },
        ],
    }
    run_result = {
        "version": 2,
        "corpus_id": "education-v1",
        "results": [
            {
                "id": "exact-001",
                "answer_type": "knowledge",
                "answer": "系统会根据不同农作物进行分类灌溉。",
                "citations": [
                    {
                        "path": "/课程资源/智慧农场.md",
                        "start_line": 1,
                        "end_line": 1,
                        "quote": "系统根据不同农作物进行分类灌溉。",
                    }
                ],
                "tool_traces": [
                    {"name": "search", "status": "success"},
                    {"name": "read", "status": "success"},
                ],
                "tool_call_count": 2,
                "latency_ms": 100.0,
                "token_usage": {
                    "input_tokens": 10,
                    "output_tokens": 5,
                    "total_tokens": 15,
                    "output_token_details": {"reasoning": 0},
                },
            },
            {
                "id": "directory-001",
                "answer_type": "directory",
                "answer": "课程资源的直接子目录是初中和小学。",
                "citations": [],
                "tool_traces": [
                    {"name": "ls", "status": "success"},
                ],
                "tool_call_count": 1,
                "latency_ms": 50.0,
                "token_usage": {
                    "input_tokens": 6,
                    "output_tokens": 4,
                    "total_tokens": 10,
                    "output_token_details": {"reasoning": 0},
                },
            },
            {
                "id": "outside-001",
                "answer_type": "insufficient",
                "answer": "当前证据不足，无法从知识库确定答案。",
                "citations": [],
                "tool_traces": [
                    {"name": "search", "status": "success"},
                    {"name": "ls", "status": "error"},
                ],
                "tool_call_count": 2,
                "latency_ms": 150.0,
                "token_usage": {
                    "input_tokens": 8,
                    "output_tokens": 6,
                    "total_tokens": 14,
                    "output_token_details": {"reasoning": 0},
                },
            },
        ],
    }

    score = scorer.score_agentic_evaluation(
        dataset=dataset,
        run_result=run_result,
        knowledge_root=knowledge_root,
    )

    assert score["completeness"] == {
        "question_count": 3,
        "result_count": 3,
        "complete": True,
        "missing_ids": [],
        "unexpected_ids": [],
        "duplicate_result_ids": [],
    }
    assert score["answer_types"] == {
        "eligible_questions": 3,
        "correct_answers": 3,
        "accuracy": 1.0,
    }
    assert score["answer_points"] == {
        "matched": 3,
        "total": 3,
        "coverage": 1.0,
    }
    assert score["citations"] == {
        "knowledge_questions": 1,
        "answers_with_citations": 1,
        "answer_coverage": 1.0,
        "citations_total": 1,
        "expected_path_citations": 1,
        "expected_path_accuracy": 1.0,
        "source_valid_citations": 1,
        "source_validity": 1.0,
    }
    assert score["directory"] == {
        "eligible_questions": 1,
        "correct_answers": 1,
        "accuracy": 1.0,
    }
    assert score["refusal"] == {
        "eligible_questions": 1,
        "correct_refusals": 1,
        "accuracy": 1.0,
    }
    assert score["tools"] == {
        "total_calls": 5,
        "successful_calls": 4,
        "error_calls": 1,
        "average_calls_per_question": 5 / 3,
        "knowledge_questions": 1,
        "knowledge_questions_with_successful_read": 1,
        "knowledge_read_compliance": 1.0,
    }
    assert score["performance"]["latency_ms"]["average"] == 100.0
    assert score["performance"]["tokens"] == {
        "reported_questions": 3,
        "input": 24,
        "output": 15,
        "total": 39,
        "reasoning": 0,
        "average_total": 13.0,
    }


def test_score_agentic_files_writes_independent_score_file(
    tmp_path: Path,
) -> None:
    """文件入口应读取评测数据，并把评分写入单独的 JSON 文件。"""

    scorer = import_module("app.agentic_evaluation_scorer")
    dataset_path = tmp_path / "questions.json"
    result_path = tmp_path / "agentic-result.json"
    output_path = tmp_path / "agentic-result-score.json"
    knowledge_root = tmp_path / "knowledge"
    knowledge_root.mkdir()

    dataset_path.write_text(
        json.dumps(
            {
                "questions": [
                    {
                        "id": "outside-001",
                        "expected_answer_type": "insufficient",
                        "expected_paths": [],
                        "answer_points": [],
                    }
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    result_path.write_text(
        json.dumps(
            {
                "version": 2,
                "corpus_id": "education-v1",
                "results": [
                    {
                        "id": "outside-001",
                        "answer_type": "insufficient",
                        "answer": "当前证据不足。",
                        "citations": [],
                        "tool_traces": [],
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    saved_path = scorer.score_agentic_files(
        dataset_path=dataset_path,
        result_path=result_path,
        knowledge_root=knowledge_root,
        output_path=output_path,
    )

    assert saved_path == output_path
    saved_score = json.loads(output_path.read_text(encoding="utf-8"))
    assert saved_score["completeness"]["complete"] is True
    assert saved_score["refusal"]["accuracy"] == 1.0


def test_main_scores_current_agentic_result_to_independent_file(
    tmp_path: Path,
    capsys,
) -> None:
    """命令行入口应为当前 Agentic 结果生成同版本的独立评分文件。"""

    scorer = import_module("app.agentic_evaluation_scorer")
    evaluation_directory = tmp_path / "evaluation"
    results_directory = evaluation_directory / "results"
    knowledge_root = tmp_path / "knowledge" / "education-v1"
    results_directory.mkdir(parents=True)
    knowledge_root.mkdir(parents=True)

    (evaluation_directory / "questions.json").write_text(
        json.dumps(
            {
                "version": 2,
                "corpus_id": "education-v1",
                "questions": [],
            }
        ),
        encoding="utf-8",
    )
    result_filename = (
        "agentic-baseline-qwen3.6-flash-thinking-off-prompt-v1.3.json"
    )
    (results_directory / result_filename).write_text(
        json.dumps(
            {
                "version": 2,
                "corpus_id": "education-v1",
                "results": [],
            }
        ),
        encoding="utf-8",
    )

    scorer.main(project_root=tmp_path)

    expected_path = results_directory / (
        "agentic-baseline-qwen3.6-flash-thinking-off-"
        "prompt-v1.3-score.json"
    )
    assert expected_path.is_file()
    assert capsys.readouterr().out == (
        f"Agentic evaluation score saved to: {expected_path}\n"
    )
