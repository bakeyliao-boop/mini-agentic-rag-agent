"""使用固定规则为 RAG 评测结果生成可复现评分。"""

import json
import math
import statistics
import unicodedata
from pathlib import Path

REFUSAL_MARKERS = (
    "未说明",
    "未包含",
    "没有足够",
    "没有相关",
    "无相关",
    "无法回答",
    "不能回答",
    "不足以",
)


def _normalize_text(value: object) -> str:
    """统一字符形式并移除空白、标点和 Markdown 符号。"""

    if not isinstance(value, str):
        return ""
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return "".join(character for character in normalized if character.isalnum())


def _rate(numerator: int, denominator: int) -> float | None:
    """计算比例；没有可评分样本时返回 None。"""

    return numerator / denominator if denominator else None


def _number(value: object) -> float | None:
    """读取数值并排除会被当作整数的布尔值。"""

    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)


def _load_json_object(source_path: Path) -> dict[str, object]:
    """读取一个顶层必须为对象的 UTF-8 JSON 文件。"""

    data = json.loads(source_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"JSON root must be an object: {source_path}")
    return data


def score_evaluation(
    dataset: dict[str, object],
    run_result: dict[str, object],
) -> dict[str, object]:
    """对问题集和一次 RAG 运行结果进行确定性评分。"""

    questions = dataset.get("questions")
    results = run_result.get("results")
    if not isinstance(questions, list):
        raise ValueError("evaluation questions must be a list")
    if not isinstance(results, list):
        raise ValueError("evaluation results must be a list")

    expected_ids: list[str] = []
    question_by_id: dict[str, dict[str, object]] = {}
    for question in questions:
        if not isinstance(question, dict) or not isinstance(question.get("id"), str):
            raise ValueError("each evaluation question must have a string id")
        question_id = question["id"]
        expected_ids.append(question_id)
        question_by_id[question_id] = question

    result_by_id: dict[str, dict[str, object]] = {}
    duplicate_result_ids: set[str] = set()
    result_ids: list[str] = []
    for result in results:
        if not isinstance(result, dict) or not isinstance(result.get("id"), str):
            raise ValueError("each evaluation result must have a string id")
        result_id = result["id"]
        result_ids.append(result_id)
        if result_id in result_by_id:
            duplicate_result_ids.add(result_id)
            continue
        result_by_id[result_id] = result

    expected_id_set = set(expected_ids)
    result_id_set = set(result_ids)
    missing_ids = [question_id for question_id in expected_ids if question_id not in result_id_set]
    unexpected_ids = sorted(result_id_set - expected_id_set)

    retrieval_eligible = 0
    top1_path_hits = 0
    topk_full_coverage_hits = 0
    directory_eligible = 0
    directory_correct = 0
    refusal_eligible = 0
    refusal_correct = 0
    answer_points_total = 0
    answer_points_matched = 0
    latencies: list[float] = []
    input_tokens = 0
    output_tokens = 0
    total_tokens = 0
    reasoning_tokens = 0
    token_reports = 0
    per_question: list[dict[str, object]] = []

    normalized_refusal_markers = tuple(
        _normalize_text(marker) for marker in REFUSAL_MARKERS
    )

    for question_id in expected_ids:
        question = question_by_id[question_id]
        result = result_by_id.get(question_id)
        expected_answer_type = question.get("expected_answer_type")
        expected_paths = [
            path
            for path in question.get("expected_paths", [])
            if isinstance(path, str)
        ] if isinstance(question.get("expected_paths", []), list) else []
        answer_points = [
            point
            for point in question.get("answer_points", [])
            if isinstance(point, str)
        ] if isinstance(question.get("answer_points", []), list) else []

        answer = result.get("answer", "") if result is not None else ""
        normalized_answer = _normalize_text(answer)
        raw_hits = result.get("hits", []) if result is not None else []
        hits = raw_hits if isinstance(raw_hits, list) else []
        actual_paths = [
            hit["path"]
            for hit in hits
            if isinstance(hit, dict) and isinstance(hit.get("path"), str)
        ]

        top1_path_hit: bool | None = None
        topk_full_coverage: bool | None = None
        if expected_answer_type == "knowledge" and expected_paths:
            retrieval_eligible += 1
            top1_path_hit = bool(actual_paths) and actual_paths[0] in expected_paths
            topk_full_coverage = set(expected_paths).issubset(actual_paths)
            top1_path_hits += int(top1_path_hit)
            topk_full_coverage_hits += int(topk_full_coverage)

        directory_answer_correct: bool | None = None
        if expected_answer_type == "directory":
            directory_eligible += 1
            expected_names = [path.rsplit("/", 1)[-1] for path in expected_paths]
            directory_answer_correct = bool(expected_names) and all(
                _normalize_text(name) in normalized_answer for name in expected_names
            )
            directory_correct += int(directory_answer_correct)

        refusal_answer_correct: bool | None = None
        if expected_answer_type == "insufficient":
            refusal_eligible += 1
            refusal_answer_correct = any(
                marker in normalized_answer for marker in normalized_refusal_markers
            )
            refusal_correct += int(refusal_answer_correct)

        matched_points = [
            point
            for point in answer_points
            if _normalize_text(point) in normalized_answer
        ]
        answer_points_total += len(answer_points)
        answer_points_matched += len(matched_points)

        latency = _number(result.get("latency_ms")) if result is not None else None
        if latency is not None:
            latencies.append(latency)

        token_usage = result.get("token_usage") if result is not None else None
        if isinstance(token_usage, dict):
            token_reports += 1
            input_tokens += int(_number(token_usage.get("input_tokens")) or 0)
            output_tokens += int(_number(token_usage.get("output_tokens")) or 0)
            total_tokens += int(_number(token_usage.get("total_tokens")) or 0)
            output_details = token_usage.get("output_token_details")
            if isinstance(output_details, dict):
                reasoning_tokens += int(
                    _number(output_details.get("reasoning")) or 0
                )

        per_question.append(
            {
                "id": question_id,
                "result_present": result is not None,
                "top1_path_hit": top1_path_hit,
                "topk_full_coverage": topk_full_coverage,
                "directory_answer_correct": directory_answer_correct,
                "refusal_answer_correct": refusal_answer_correct,
                "answer_points_matched": matched_points,
                "answer_points_total": len(answer_points),
                "answer_point_coverage": _rate(
                    len(matched_points),
                    len(answer_points),
                ),
                "latency_ms": latency,
                "total_tokens": (
                    int(_number(token_usage.get("total_tokens")) or 0)
                    if isinstance(token_usage, dict)
                    else None
                ),
            }
        )

    sorted_latencies = sorted(latencies)
    latency_count = len(sorted_latencies)
    latency_summary = {
        "count": latency_count,
        "minimum": sorted_latencies[0] if sorted_latencies else None,
        "average": (
            sum(sorted_latencies) / latency_count if latency_count else None
        ),
        "median": statistics.median(sorted_latencies) if sorted_latencies else None,
        "p95_nearest_rank": (
            sorted_latencies[math.ceil(0.95 * latency_count) - 1]
            if latency_count
            else None
        ),
        "maximum": sorted_latencies[-1] if sorted_latencies else None,
    }

    return {
        "version": run_result.get("version"),
        "corpus_id": run_result.get("corpus_id"),
        "scoring_method": {
            "answer_points": "normalized_exact_substring",
            "refusal": "fixed_phrase_match",
            "semantic_judge_used": False,
        },
        "completeness": {
            "question_count": len(questions),
            "result_count": len(results),
            "complete": (
                len(questions) == len(results)
                and not missing_ids
                and not unexpected_ids
                and not duplicate_result_ids
            ),
            "missing_ids": missing_ids,
            "unexpected_ids": unexpected_ids,
            "duplicate_result_ids": sorted(duplicate_result_ids),
        },
        "retrieval": {
            "eligible_questions": retrieval_eligible,
            "top1_path_hits": top1_path_hits,
            "top1_path_hit_rate": _rate(top1_path_hits, retrieval_eligible),
            "topk_full_coverage_hits": topk_full_coverage_hits,
            "topk_full_coverage_rate": _rate(
                topk_full_coverage_hits,
                retrieval_eligible,
            ),
        },
        "directory": {
            "eligible_questions": directory_eligible,
            "correct_answers": directory_correct,
            "accuracy": _rate(directory_correct, directory_eligible),
        },
        "refusal": {
            "eligible_questions": refusal_eligible,
            "correct_refusals": refusal_correct,
            "accuracy": _rate(refusal_correct, refusal_eligible),
        },
        "answer_points": {
            "matched": answer_points_matched,
            "total": answer_points_total,
            "coverage": _rate(answer_points_matched, answer_points_total),
        },
        "performance": {
            "latency_ms": latency_summary,
            "tokens": {
                "reported_questions": token_reports,
                "input": input_tokens,
                "output": output_tokens,
                "total": total_tokens,
                "reasoning": reasoning_tokens,
                "average_total": _rate(total_tokens, token_reports),
            },
        },
        "per_question": per_question,
    }


def score_files(
    dataset_path: Path,
    result_path: Path,
    output_path: Path,
) -> Path:
    """读取问题与运行结果，生成评分 JSON 并返回输出路径。"""

    dataset = _load_json_object(dataset_path)
    run_result = _load_json_object(result_path)
    score = score_evaluation(dataset, run_result)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(score, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return output_path


def main(project_root: Path | None = None) -> None:
    """评分项目中的传统 RAG baseline，并打印评分文件路径。"""

    resolved_project_root = (
        project_root
        if project_root is not None
        else Path(__file__).resolve().parent.parent
    )
    output_path = score_files(
        resolved_project_root / "evaluation" / "questions.json",
        resolved_project_root
        / "evaluation"
        / "results"
        / "traditional-baseline.json",
        resolved_project_root
        / "evaluation"
        / "results"
        / "traditional-baseline-score.json",
    )
    print(f"Evaluation score saved to: {output_path}")


if __name__ == "__main__":
    main()
