import json
from importlib import import_module
from pathlib import Path


def test_run_semantic_judge_calibration_scores_fixed_cases() -> None:
    """校准器应比较三类固定样例的预期判断与实际判断。"""

    calibration = import_module("evaluation.calibration")
    cases = [
        {
            "id": "paraphrase-001",
            "question": "气象站能做什么？",
            "answer_point": "实时监测天气",
            "answer": "气象站能够持续观察天气状况。",
            "expected_verdict": "matched",
        },
        {
            "id": "partial-001",
            "question": "气象站如何降低农业风险？",
            "answer_point": "预测农业灾害并提前采取防治措施",
            "answer": "气象站能够预测农业灾害。",
            "expected_verdict": "not_matched",
        },
        {
            "id": "contradiction-001",
            "question": "气象站能否预测农业灾害？",
            "answer_point": "气象站可以预测农业灾害",
            "answer": "气象站不能预测农业灾害。",
            "expected_verdict": "not_matched",
        },
    ]
    received_answers: list[str] = []

    def fake_judge(
        question: str,
        answer_points: list[str],
        answer: str,
    ) -> list[dict[str, object]]:
        received_answers.append(answer)
        verdict_by_answer = {
            "气象站能够持续观察天气状况。": "matched",
            "气象站能够预测农业灾害。": "not_matched",
            "气象站不能预测农业灾害。": "not_matched",
        }
        return [
            {
                "point_index": 0,
                "verdict": verdict_by_answer[answer],
                "reason": "离线假裁判结果。",
            }
        ]

    result = calibration.run_semantic_judge_calibration(
        cases=cases,
        judge=fake_judge,
    )

    assert received_answers == [
        "气象站能够持续观察天气状况。",
        "气象站能够预测农业灾害。",
        "气象站不能预测农业灾害。",
    ]
    assert result["summary"] == {
        "case_count": 3,
        "correct": 3,
        "accuracy": 1.0,
        "passed": True,
    }
    assert [case["actual_verdict"] for case in result["per_case"]] == [
        "matched",
        "not_matched",
        "not_matched",
    ]
    assert all(case["correct"] for case in result["per_case"])


def test_main_runs_deepseek_calibration_and_saves_result(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    """命令行入口应运行固定校准样例并保存可复现结果。"""

    calibration = import_module("evaluation.calibration")
    settings = {
        "SEMANTIC_JUDGE_MODEL": "deepseek-v4-flash",
        "SEMANTIC_JUDGE_API_KEY": "test-key",
        "SEMANTIC_JUDGE_BASE_URL": "https://api.deepseek.com",
    }
    fake_model = object()
    fake_judge = object()
    calibration_result = {
        "summary": {
            "case_count": 3,
            "correct": 3,
            "accuracy": 1.0,
            "passed": True,
        },
        "per_case": [],
    }
    events: list[tuple[object, ...]] = []

    def fake_load_settings(project_root: Path) -> dict[str, str]:
        events.append(("load_settings", project_root))
        return settings

    def fake_build_chat_model(
        received_settings: dict[str, str],
    ) -> object:
        events.append(("build_chat_model", received_settings))
        return fake_model

    def fake_build_judge(chat_model: object) -> object:
        events.append(("build_judge", chat_model))
        return fake_judge

    def fake_run_calibration(
        cases: list[dict[str, object]],
        judge: object,
    ) -> dict[str, object]:
        events.append(
            (
                "run_calibration",
                [case["id"] for case in cases],
                judge,
            )
        )
        return calibration_result

    monkeypatch.setattr(
        calibration,
        "load_settings_from_env",
        fake_load_settings,
        raising=False,
    )
    monkeypatch.setattr(
        calibration,
        "build_semantic_judge_chat_model",
        fake_build_chat_model,
        raising=False,
    )
    monkeypatch.setattr(
        calibration,
        "build_semantic_judge",
        fake_build_judge,
        raising=False,
    )
    monkeypatch.setattr(
        calibration,
        "run_semantic_judge_calibration",
        fake_run_calibration,
    )

    calibration.main(project_root=tmp_path)

    output_path = (
        tmp_path
        / "evaluation"
        / "results"
        / "semantic-judge-deepseek-v4-flash-calibration.json"
    )
    assert events == [
        ("load_settings", tmp_path),
        ("build_chat_model", settings),
        ("build_judge", fake_model),
        (
            "run_calibration",
            ["paraphrase-001", "partial-001", "contradiction-001"],
            fake_judge,
        ),
    ]
    saved_result = json.loads(output_path.read_text(encoding="utf-8"))
    assert saved_result == {
        "version": 1,
        "judge": {
            "model": "deepseek-v4-flash",
            "temperature": 0,
            "thinking": "disabled",
            "prompt_version": "SemanticJudge-V1",
        },
        **calibration_result,
    }
    assert capsys.readouterr().out == (
        f"Semantic judge calibration saved to: {output_path}\n"
    )
