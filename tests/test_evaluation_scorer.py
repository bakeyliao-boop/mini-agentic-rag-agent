import json
from pathlib import Path

from app.evaluation_scorer import score_evaluation, score_files


def test_score_evaluation_calculates_quality_and_cost_metrics() -> None:
    """评分器应汇总路径、目录、拒答、要点、延迟和 token 指标。"""

    dataset = {
        "version": 2,
        "corpus_id": "education-v1",
        "questions": [
            {
                "id": "directory-001",
                "expected_answer_type": "directory",
                "expected_paths": ["/课程资源/初中", "/课程资源/小学"],
                "answer_points": ["初中", "小学"],
            },
            {
                "id": "exact-001",
                "expected_answer_type": "knowledge",
                "expected_paths": ["/智慧农场.md"],
                "answer_points": ["分类灌溉"],
            },
            {
                "id": "multi-001",
                "expected_answer_type": "knowledge",
                "expected_paths": ["/课程介绍.md", "/智慧农场.md"],
                "answer_points": ["综合两个文件"],
            },
            {
                "id": "outside-001",
                "expected_answer_type": "insufficient",
                "expected_paths": [],
                "answer_points": ["当前知识库未说明量子计算原理"],
            },
        ],
    }
    run_result = {
        "version": 2,
        "corpus_id": "education-v1",
        "results": [
            {
                "id": "directory-001",
                "answer": "课程资源的直接子目录是初中和小学。",
                "hits": [],
                "latency_ms": 50.0,
                "token_usage": {
                    "input_tokens": 6,
                    "output_tokens": 4,
                    "total_tokens": 10,
                    "output_token_details": {"reasoning": 2},
                },
            },
            {
                "id": "exact-001",
                "answer": "系统根据不同农作物进行分类灌溉。",
                "hits": [{"path": "/智慧农场.md"}],
                "latency_ms": 100.0,
                "token_usage": {
                    "input_tokens": 10,
                    "output_tokens": 5,
                    "total_tokens": 15,
                    "output_token_details": {"reasoning": 3},
                },
            },
            {
                "id": "multi-001",
                "answer": "综合两个文件可知，它们是理论和案例的关系。",
                "hits": [
                    {"path": "/智慧农场.md"},
                    {"path": "/课程介绍.md"},
                ],
                "latency_ms": 200.0,
                "token_usage": {
                    "input_tokens": 12,
                    "output_tokens": 8,
                    "total_tokens": 20,
                    "output_token_details": {"reasoning": 4},
                },
            },
            {
                "id": "outside-001",
                "answer": "当前知识库未说明量子计算原理。",
                "hits": [],
                "latency_ms": 300.0,
                "token_usage": {
                    "input_tokens": 7,
                    "output_tokens": 5,
                    "total_tokens": 12,
                    "output_token_details": {"reasoning": 2},
                },
            },
        ],
    }

    score = score_evaluation(dataset, run_result)

    assert score["completeness"] == {
        "question_count": 4,
        "result_count": 4,
        "complete": True,
        "missing_ids": [],
        "unexpected_ids": [],
        "duplicate_result_ids": [],
    }
    assert score["retrieval"] == {
        "eligible_questions": 2,
        "top1_path_hits": 2,
        "top1_path_hit_rate": 1.0,
        "topk_full_coverage_hits": 2,
        "topk_full_coverage_rate": 1.0,
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
    assert score["answer_points"]["coverage"] == 1.0
    assert score["performance"]["latency_ms"]["average"] == 162.5
    assert score["performance"]["latency_ms"]["median"] == 150.0
    assert score["performance"]["tokens"]["total"] == 57
    assert score["performance"]["tokens"]["reasoning"] == 11


def test_score_files_writes_score_json(tmp_path: Path) -> None:
    """评分脚本应读取输入文件并保存可重新读取的 UTF-8 JSON。"""

    dataset_path = tmp_path / "questions.json"
    result_path = tmp_path / "baseline.json"
    output_path = tmp_path / "baseline-score.json"
    dataset = {
        "version": 2,
        "corpus_id": "education-v1",
        "questions": [],
    }
    run_result = {
        "version": 2,
        "corpus_id": "education-v1",
        "results": [],
    }
    dataset_path.write_text(
        json.dumps(dataset, ensure_ascii=False),
        encoding="utf-8",
        newline="\n",
    )
    result_path.write_text(
        json.dumps(run_result, ensure_ascii=False),
        encoding="utf-8",
        newline="\n",
    )

    returned_path = score_files(dataset_path, result_path, output_path)

    assert returned_path == output_path
    saved_score = json.loads(output_path.read_text(encoding="utf-8"))
    assert saved_score["completeness"]["complete"] is True
