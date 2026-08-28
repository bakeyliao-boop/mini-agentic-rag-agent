"""固定评测问题集的读取、校验与结果保存。"""

import json
from pathlib import Path

EVALUATION_VERSION = 2
EVALUATION_CORPUS_ID = "education-v1"
EVALUATION_QUESTION_COUNT = 10


def load_evaluation_questions(source_path: Path) -> dict[str, object]:
    """读取固定问题集，并校验版本、语料 ID 和问题数量。"""

    data = json.loads(source_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("evaluation data must be a JSON object")
    if data.get("version") != EVALUATION_VERSION:
        raise ValueError("evaluation version must be 2")
    if data.get("corpus_id") != EVALUATION_CORPUS_ID:
        raise ValueError("evaluation corpus_id must be education-v1")

    questions = data.get("questions")
    if not isinstance(questions, list):
        raise ValueError("evaluation questions must be a list")
    if len(questions) != EVALUATION_QUESTION_COUNT:
        raise ValueError("evaluation questions must contain exactly 10 items")

    required_fields = (
        "id",
        "category",
        "question",
        "expected_answer_type",
    )
    question_ids: set[str] = set()
    for index, question in enumerate(questions):
        if not isinstance(question, dict):
            raise ValueError(f"evaluation questions[{index}] must be an object")

        for field_name in required_fields:
            field_value = question.get(field_name)
            if not isinstance(field_value, str) or not field_value.strip():
                raise ValueError(
                    f"evaluation questions[{index}].{field_name} "
                    "must be a non-empty string"
                )

        question_id = question["id"]
        if question_id in question_ids:
            raise ValueError(f"duplicate evaluation question id: {question_id}")
        question_ids.add(question_id)

    return data


def save_evaluation_result(
    result: dict[str, object],
    output_path: Path,
) -> None:
    """将评测结果保存为便于阅读和比较的 UTF-8 JSON。"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
