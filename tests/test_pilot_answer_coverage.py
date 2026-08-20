"""Pilot A 组最终答案的语义覆盖评分（复用 LLM-as-judge）离线测试。

测试目标：score_pilot_answer_coverage 应：
1. 从规范提取 12 个 expectation 作为答案点，交给现有语义 Judge。
2. 把 judge 的 verdict 映射回 {id, service, field, covered}，covered 仅当 matched。
3. judge 漏判时复用底层校验并报错。
4. uncertain 不算覆盖，且不中断评分。
"""

import importlib
import json
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SPEC_PATH = (
    PROJECT_ROOT
    / "evaluation"
    / "pilots"
    / "multi_chunk_guide_001.json"
)
ARM_A_ANSWER_PATH = (
    PROJECT_ROOT
    / "evaluation"
    / "results"
    / "pilot-multi-chunk-guide-001-arm-a-answer.json"
)


def _load_json(path: Path) -> dict[str, object]:
    """从项目文件读取 JSON 对象。"""

    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return data


def test_score_pilot_answer_coverage_maps_verdicts_on_real_files() -> None:
    """真实规范与回答上：judge 判 3 个 matched 时映射为 3/12 覆盖。"""

    pilot_scorer = importlib.import_module("evaluation.scorers.pilot")
    spec = _load_json(SPEC_PATH)
    answer_result = _load_json(ARM_A_ANSWER_PATH)
    captured: dict[str, object] = {}

    def fake_judge(
        question: str,
        answer_points: list[str],
        answer: str,
    ) -> list[dict[str, object]]:
        captured["question"] = question
        captured["answer_points"] = answer_points
        captured["answer"] = answer
        return [
            {
                "point_index": index,
                "verdict": (
                    "matched" if index < 3 else "not_matched"
                ),
                "reason": f"reason-{index}",
            }
            for index in range(len(answer_points))
        ]

    result = pilot_scorer.score_pilot_answer_coverage(
        spec=spec,
        answer_result=answer_result,
        judge=fake_judge,
    )

    expected_expectations = [
        point["expectation"]
        for point in spec["answer_points"]
    ]
    assert captured["question"] == answer_result["question"]
    assert captured["answer"] == answer_result["answer"]
    assert captured["answer_points"] == expected_expectations

    assert result["pilot_id"] == "multi-chunk-guide-001"
    assert result["arm_id"] == "A"
    assert result["total_answer_point_count"] == 12
    assert result["covered_answer_point_count"] == 3
    assert result["answer_point_coverage"] == 0.25

    assert result["answer_points"][0] == {
        "id": "priority-review-condition",
        "service": "专利优先审查受理",
        "field": "受理条件",
        "verdict": "matched",
        "reason": "reason-0",
        "covered": True,
    }
    assert result["answer_points"][3]["id"] == "priority-review-result"
    assert result["answer_points"][3]["verdict"] == "not_matched"
    assert result["answer_points"][3]["covered"] is False


def test_score_pilot_answer_coverage_rejects_partial_judgments() -> None:
    """judge 漏判某个答案点时，应复用底层校验并报错。"""

    pilot_scorer = importlib.import_module("evaluation.scorers.pilot")
    spec = _load_json(SPEC_PATH)
    answer_result = _load_json(ARM_A_ANSWER_PATH)

    def fake_judge(
        question: str,
        answer_points: list[str],
        answer: str,
    ) -> list[dict[str, object]]:
        return [
            {
                "point_index": index,
                "verdict": "not_matched",
                "reason": "missing one",
            }
            for index in range(len(answer_points) - 1)
        ]

    with pytest.raises(ValueError):
        pilot_scorer.score_pilot_answer_coverage(
            spec=spec,
            answer_result=answer_result,
            judge=fake_judge,
        )


def test_score_pilot_answer_coverage_uncertain_is_not_covered() -> None:
    """uncertain 判断不应算覆盖，且不中断整体评分。"""

    pilot_scorer = importlib.import_module("evaluation.scorers.pilot")
    spec = _load_json(SPEC_PATH)
    answer_result = _load_json(ARM_A_ANSWER_PATH)

    def fake_judge(
        question: str,
        answer_points: list[str],
        answer: str,
    ) -> list[dict[str, object]]:
        return [
            {
                "point_index": index,
                "verdict": "uncertain" if index == 0 else "matched",
                "reason": "cannot tell",
            }
            for index in range(len(answer_points))
        ]

    result = pilot_scorer.score_pilot_answer_coverage(
        spec=spec,
        answer_result=answer_result,
        judge=fake_judge,
    )

    assert result["answer_points"][0]["verdict"] == "uncertain"
    assert result["answer_points"][0]["covered"] is False
    assert result["covered_answer_point_count"] == 11
    assert result["answer_point_coverage"] == 11 / 12


def test_score_pilot_arm_answer_from_project_wires_judge_and_saves(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """评分项目入口（A 组默认）应构建 Judge、读取真实文件、评分并保存。"""

    pilot_cli = importlib.import_module("evaluation.cli.pilot")
    spec_path = (
        tmp_path
        / "evaluation"
        / "pilots"
        / "multi_chunk_guide_001.json"
    )
    spec_path.parent.mkdir(parents=True)
    spec = {
        "id": "multi-chunk-guide-001",
        "question": "请整理三项业务的办理清单。",
        "comparison_arms": [],
        "answer_points": [],
    }
    spec_path.write_text(
        json.dumps(spec, ensure_ascii=False),
        encoding="utf-8",
    )
    answer_path = (
        tmp_path
        / "evaluation"
        / "results"
        / "pilot-multi-chunk-guide-001-arm-a-answer.json"
    )
    answer_path.parent.mkdir(parents=True)
    answer_result = {
        "pilot_id": "multi-chunk-guide-001",
        "arm_id": "A",
        "question": "请整理三项业务的办理清单。",
        "answer": "台账内容。",
        "hits": [],
    }
    answer_path.write_text(
        json.dumps(answer_result, ensure_ascii=False),
        encoding="utf-8",
    )
    settings = {
        "SEMANTIC_JUDGE_MODEL": "deepseek-v4-flash",
        "SEMANTIC_JUDGE_API_KEY": "test-judge-key",
        "SEMANTIC_JUDGE_BASE_URL": "https://api.deepseek.com",
    }
    fake_chat_model = object()
    fake_judge = object()
    fake_score_result = {"pilot_id": "multi-chunk-guide-001", "arm_id": "A"}
    calls: list[tuple[object, ...]] = []

    def fake_build_judge_chat_model(
        loaded_settings: object,
    ) -> object:
        calls.append(("judge_model", loaded_settings))
        return fake_chat_model

    def fake_build_judge(chat_model: object) -> object:
        calls.append(("judge", chat_model))
        return fake_judge

    def fake_score(
        loaded_spec: dict[str, object],
        loaded_answer: dict[str, object],
        judge: object,
    ) -> dict[str, object]:
        calls.append(("score", loaded_spec, loaded_answer, judge))
        return fake_score_result

    def fake_save_result(
        result: dict[str, object],
        output_path: Path,
    ) -> None:
        calls.append(("save", result, output_path))

    monkeypatch.setattr(
        pilot_cli,
        "build_semantic_judge_chat_model",
        fake_build_judge_chat_model,
    )
    monkeypatch.setattr(pilot_cli, "build_semantic_judge", fake_build_judge)
    monkeypatch.setattr(
        pilot_cli,
        "score_pilot_answer_coverage",
        fake_score,
    )
    monkeypatch.setattr(
        pilot_cli,
        "save_evaluation_result",
        fake_save_result,
    )

    output_path = pilot_cli.score_pilot_arm_answer_from_project(
        project_root=tmp_path,
        settings=settings,
    )

    expected_output_path = (
        tmp_path
        / "evaluation"
        / "results"
        / "pilot-multi-chunk-guide-001-arm-a-answer-coverage.json"
    )
    assert output_path == expected_output_path
    assert calls == [
        ("judge_model", settings),
        ("judge", fake_chat_model),
        ("score", spec, answer_result, fake_judge),
        ("save", fake_score_result, expected_output_path),
    ]
