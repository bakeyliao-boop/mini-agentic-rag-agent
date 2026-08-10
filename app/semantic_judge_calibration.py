"""使用固定正反样例校准语义裁判的判断能力。"""

import json
from pathlib import Path

from app.baseline_runner import _required_setting, load_settings_from_env
from app.semantic_evaluation_scorer import (
    SEMANTIC_JUDGE_PROMPT_VERSION,
    SemanticJudge,
    build_semantic_judge,
    build_semantic_judge_chat_model,
    score_semantic_answer_points,
)


SEMANTIC_JUDGE_CALIBRATION_CASES: list[dict[str, object]] = [
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


def run_semantic_judge_calibration(
    cases: list[dict[str, object]],
    judge: SemanticJudge,
) -> dict[str, object]:
    """逐条运行固定样例，并比较实际判断与预期判断。"""

    per_case: list[dict[str, object]] = []
    correct_count = 0

    for case in cases:
        case_id = case.get("id")
        question = case.get("question")
        answer_point = case.get("answer_point")
        answer = case.get("answer")
        expected_verdict = case.get("expected_verdict")
        if not all(
            isinstance(value, str)
            for value in (
                case_id,
                question,
                answer_point,
                answer,
                expected_verdict,
            )
        ):
            raise ValueError("each calibration case must contain string fields")

        score = score_semantic_answer_points(
            question=question,
            answer_points=[answer_point],
            answer=answer,
            judge=judge,
        )
        judgment = score["judgments"][0]
        actual_verdict = judgment["verdict"]
        is_correct = actual_verdict == expected_verdict
        correct_count += int(is_correct)
        per_case.append(
            {
                "id": case_id,
                "expected_verdict": expected_verdict,
                "actual_verdict": actual_verdict,
                "correct": is_correct,
                "reason": judgment["reason"],
            }
        )

    case_count = len(cases)
    return {
        "summary": {
            "case_count": case_count,
            "correct": correct_count,
            "accuracy": correct_count / case_count if case_count else None,
            "passed": bool(case_count) and correct_count == case_count,
        },
        "per_case": per_case,
    }


def main(project_root: Path | None = None) -> None:
    """使用真实 DeepSeek Judge 运行固定校准样例并保存结果。"""

    resolved_project_root = (
        project_root
        if project_root is not None
        else Path(__file__).resolve().parent.parent
    )
    settings = load_settings_from_env(resolved_project_root)
    chat_model = build_semantic_judge_chat_model(settings)
    judge = build_semantic_judge(chat_model)
    result = run_semantic_judge_calibration(
        cases=SEMANTIC_JUDGE_CALIBRATION_CASES,
        judge=judge,
    )

    model_name = _required_setting(settings, "SEMANTIC_JUDGE_MODEL")
    output_path = (
        resolved_project_root
        / "evaluation"
        / "results"
        / f"semantic-judge-{model_name}-calibration.json"
    )
    saved_result = {
        "version": 1,
        "judge": {
            "model": model_name,
            "temperature": 0,
            "thinking": "disabled",
            "prompt_version": SEMANTIC_JUDGE_PROMPT_VERSION,
        },
        **result,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(saved_result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Semantic judge calibration saved to: {output_path}")


if __name__ == "__main__":
    main()
