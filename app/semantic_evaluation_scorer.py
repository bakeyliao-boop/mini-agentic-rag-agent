"""对最终回答中的标准答案点进行独立语义评分。"""

import argparse
import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from app.baseline_runner import _required_setting, load_settings_from_env


SEMANTIC_JUDGE_PROMPT_VERSION = "SemanticJudge-V1"


class SemanticPointJudgment(BaseModel):
    """语义裁判对一个标准答案点的判断。"""

    point_index: int = Field(ge=0)
    verdict: Literal["matched", "not_matched", "uncertain"]
    reason: str = Field(min_length=1)


class SemanticJudgeResponse(BaseModel):
    """语义裁判一次返回的全部逐点判断。"""

    judgments: list[SemanticPointJudgment]


SemanticJudge = Callable[
    [str, list[str], str],
    list[dict[str, object]],
]


def build_semantic_judge_chat_model(
    settings: Mapping[str, str],
) -> ChatOpenAI:
    """根据独立配置创建 DeepSeek 官方非思考 Judge 模型。"""

    return ChatOpenAI(
        model=_required_setting(settings, "SEMANTIC_JUDGE_MODEL"),
        temperature=0,
        api_key=_required_setting(settings, "SEMANTIC_JUDGE_API_KEY"),
        base_url=_required_setting(settings, "SEMANTIC_JUDGE_BASE_URL"),
        extra_body={
            "thinking": {"type": "disabled"},
        },
    )


def build_semantic_judge(chat_model: BaseChatModel) -> SemanticJudge:
    """把聊天模型封装为返回结构化逐点判断的语义裁判。"""

    structured_model = chat_model.with_structured_output(
        SemanticJudgeResponse,
        method="json_mode",
    )

    def judge(
        question: str,
        answer_points: list[str],
        answer: str,
    ) -> list[dict[str, object]]:
        """只根据问题、标准答案点和实际回答判断语义覆盖。"""

        response = structured_model.invoke(
            [
                SystemMessage(
                    content=(
                        "你是答案点语义裁判。只根据用户问题、标准答案点和实际回答判断，"
                        "不得使用外部知识。逐个答案点返回判断：完整表达相同含义时使用 "
                        "matched；缺失、只表达部分含义或含义相反时使用 not_matched；"
                        "无法可靠判断时使用 uncertain。含义相反必须判为 not_matched。"
                        "只输出 JSON，格式为 "
                        '{"judgments":[{"point_index":0,'
                        '"verdict":"matched","reason":"判断理由"}]}。'
                    )
                ),
                HumanMessage(
                    content=json.dumps(
                        {
                            "question": question,
                            "answer_points": answer_points,
                            "answer": answer,
                        },
                        ensure_ascii=False,
                    )
                ),
            ]
        )
        validated_response = SemanticJudgeResponse.model_validate(
            response,
        )
        return [
            judgment.model_dump()
            for judgment in validated_response.judgments
        ]

    return judge


def score_semantic_answer_points(
    question: str,
    answer_points: list[str],
    answer: str,
    judge: SemanticJudge,
) -> dict[str, object]:
    """调用语义裁判逐点评分，并汇总命中情况。"""

    raw_judgments = judge(question, answer_points, answer)
    expected_indices = set(range(len(answer_points)))
    received_indices = [
        judgment.get("point_index")
        for judgment in raw_judgments
        if isinstance(judgment, dict)
    ]
    if (
        len(raw_judgments) != len(answer_points)
        or len(received_indices) != len(raw_judgments)
        or set(received_indices) != expected_indices
        or len(set(received_indices)) != len(received_indices)
    ):
        raise ValueError(
            "semantic judge must return exactly one judgment "
            "for each answer point"
        )

    judgments: list[dict[str, object]] = []
    verdict_counts = {
        "matched": 0,
        "not_matched": 0,
        "uncertain": 0,
    }

    for raw_judgment in raw_judgments:
        point_index = raw_judgment["point_index"]
        verdict = raw_judgment["verdict"]
        reason = raw_judgment["reason"]

        verdict_counts[str(verdict)] += 1
        judgments.append(
            {
                "point_index": point_index,
                "answer_point": answer_points[int(point_index)],
                "verdict": verdict,
                "reason": reason,
            }
        )

    total = len(answer_points)
    matched = verdict_counts["matched"]
    return {
        "matched": matched,
        "not_matched": verdict_counts["not_matched"],
        "uncertain": verdict_counts["uncertain"],
        "total": total,
        "coverage": matched / total if total else None,
        "judgments": judgments,
    }


def score_semantic_evaluation(
    dataset: dict[str, object],
    run_result: dict[str, object],
    judge: SemanticJudge,
) -> dict[str, object]:
    """只对评测集中的知识题执行语义答案点评分。"""

    questions = dataset.get("questions")
    results = run_result.get("results")
    if not isinstance(questions, list):
        raise ValueError("evaluation questions must be a list")
    if not isinstance(results, list):
        raise ValueError("evaluation results must be a list")

    result_by_id = {
        result["id"]: result
        for result in results
        if isinstance(result, dict) and isinstance(result.get("id"), str)
    }
    per_question: list[dict[str, object]] = []
    matched = 0
    not_matched = 0
    uncertain = 0
    total = 0

    for question_data in questions:
        if not isinstance(question_data, dict):
            raise ValueError("each evaluation question must be an object")
        if question_data.get("expected_answer_type") != "knowledge":
            continue

        question_id = question_data.get("id")
        question = question_data.get("question")
        raw_answer_points = question_data.get("answer_points")
        if not isinstance(question_id, str) or not isinstance(question, str):
            raise ValueError("knowledge question must have string id and question")
        if not isinstance(raw_answer_points, list) or not all(
            isinstance(point, str)
            for point in raw_answer_points
        ):
            raise ValueError("knowledge answer_points must be a list of strings")

        result = result_by_id.get(question_id, {})
        answer = result.get("answer", "")
        if not isinstance(answer, str):
            raise ValueError("evaluation answer must be a string")

        question_score = score_semantic_answer_points(
            question=question,
            answer_points=raw_answer_points,
            answer=answer,
            judge=judge,
        )
        matched += int(question_score["matched"])
        not_matched += int(question_score["not_matched"])
        uncertain += int(question_score["uncertain"])
        total += int(question_score["total"])
        per_question.append(
            {
                "id": question_id,
                **question_score,
            }
        )

    return {
        "version": run_result.get("version"),
        "corpus_id": run_result.get("corpus_id"),
        "scoring_method": {
            "answer_points": "llm_semantic_judge",
            "semantic_judge_used": True,
        },
        "answer_points": {
            "eligible_questions": len(per_question),
            "matched": matched,
            "not_matched": not_matched,
            "uncertain": uncertain,
            "total": total,
            "coverage": matched / total if total else None,
        },
        "per_question": per_question,
    }


def _load_json_object(source_path: Path) -> dict[str, object]:
    """读取 UTF-8 JSON 文件，并要求顶层是对象。"""

    data = json.loads(source_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("semantic scoring input must be a JSON object")
    return data


def score_semantic_files(
    dataset_path: Path,
    result_path: Path,
    output_path: Path,
    judge: SemanticJudge,
) -> Path:
    """读取评测文件并把语义评分写入独立 JSON 文件。"""

    dataset = _load_json_object(dataset_path)
    run_result = _load_json_object(result_path)
    score = score_semantic_evaluation(
        dataset=dataset,
        run_result=run_result,
        judge=judge,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(score, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return output_path


def build_semantic_score_filename(result_filename: str) -> str:
    """根据 Agentic 原始结果文件名生成独立语义评分文件名。"""

    result_path = Path(result_filename)
    return f"{result_path.stem}-semantic-score.json"


def main(
    project_root: Path | None = None,
    argv: list[str] | None = None,
) -> None:
    """使用 DeepSeek Judge 为指定评测结果生成语义评分。"""

    from app.agentic_evaluation_runner import (
        build_agentic_evaluation_result_filename,
    )
    from app.prompts import KNOWLEDGE_AGENT_PROMPT_VERSION
    from app.traditional_rag import TraditionalRagConfig

    resolved_project_root = (
        project_root
        if project_root is not None
        else Path(__file__).resolve().parent.parent
    )
    parser = argparse.ArgumentParser(
        description="使用 DeepSeek Judge 生成语义评分",
    )
    parser.add_argument(
        "--result-filename",
        help="evaluation/results 中需要评分的原始结果文件名",
    )
    parsed_args = parser.parse_args(
        argv if argv is not None else ([] if project_root is not None else None)
    )

    settings = load_settings_from_env(resolved_project_root)
    chat_model = build_semantic_judge_chat_model(settings)
    judge = build_semantic_judge(chat_model)
    default_result_filename = build_agentic_evaluation_result_filename(
        TraditionalRagConfig(),
        KNOWLEDGE_AGENT_PROMPT_VERSION,
    )
    result_filename = (
        parsed_args.result_filename or default_result_filename
    )
    output_filename = build_semantic_score_filename(result_filename)
    output_path = score_semantic_files(
        dataset_path=(
            resolved_project_root / "evaluation" / "questions.json"
        ),
        result_path=(
            resolved_project_root
            / "evaluation"
            / "results"
            / result_filename
        ),
        output_path=(
            resolved_project_root
            / "evaluation"
            / "results"
            / output_filename
        ),
        judge=judge,
    )
    output_label = (
        "Agentic semantic evaluation score"
        if parsed_args.result_filename is None
        else "Semantic evaluation score"
    )
    print(f"{output_label} saved to: {output_path}")


if __name__ == "__main__":
    main()
