"""为 Agentic RAG 结果生成可复现的确定性评分。"""

import json
import math
import statistics
import unicodedata
from pathlib import Path

from app.knowledge_store import read_markdown_lines, resolve_knowledge_path

ANSWER_POINT_FILLER_TERMS = (
    "可以",
    "能够",
    "进行",
    "并且",
    "以及",
    "的",
)


def _rate(numerator: int, denominator: int) -> float | None:
    """计算比例；没有可评分样本时返回 None。"""

    return numerator / denominator if denominator else None


def _number(value: object) -> float | None:
    """读取数值，并排除会被当作整数的布尔值。"""

    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)


def _normalize_text(value: object) -> str:
    """统一字符形式，并移除空白、标点和 Markdown 符号。"""

    if not isinstance(value, str):
        return ""
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return "".join(
        character
        for character in normalized
        if character.isalnum()
    )


def _normalize_answer_point_text(value: object) -> str:
    """在基础规范化后忽略不改变核心含义的少量连接词。"""

    normalized = _normalize_text(value)
    for filler_term in ANSWER_POINT_FILLER_TERMS:
        normalized = normalized.replace(filler_term, "")
    return normalized


def _answer_point_matches(point: str, answer: object) -> bool:
    """判断回答是否包含答案点的核心连续内容。"""

    normalized_point = _normalize_answer_point_text(point)
    normalized_answer = _normalize_answer_point_text(answer)
    return bool(normalized_point) and normalized_point in normalized_answer


def _citation_source_is_valid(
    citation: dict[str, object],
    knowledge_root: Path,
) -> bool:
    """重新读取 Source Markdown，验证 Citation 的行号和原文。"""

    virtual_path = citation.get("path")
    start_line = citation.get("start_line")
    end_line = citation.get("end_line")
    quote = citation.get("quote")
    if (
        not isinstance(virtual_path, str)
        or not isinstance(start_line, int)
        or isinstance(start_line, bool)
        or not isinstance(end_line, int)
        or isinstance(end_line, bool)
        or start_line < 1
        or end_line < start_line
        or not isinstance(quote, str)
    ):
        return False

    try:
        source_path = resolve_knowledge_path(
            virtual_path,
            knowledge_root,
        )
        source_lines = read_markdown_lines(source_path)
    except (FileNotFoundError, IsADirectoryError, ValueError):
        return False

    selected_lines = [
        text
        for line_number, text in source_lines
        if start_line <= line_number <= end_line
    ]
    expected_line_count = end_line - start_line + 1
    return (
        len(selected_lines) == expected_line_count
        and "\n".join(selected_lines) == quote
    )


def score_agentic_evaluation(
    dataset: dict[str, object],
    run_result: dict[str, object],
    knowledge_root: Path,
) -> dict[str, object]:
    """对固定问题集和一次 Agentic RAG 结果进行确定性评分。"""

    questions = dataset.get("questions")
    results = run_result.get("results")
    if not isinstance(questions, list):
        raise ValueError("evaluation questions must be a list")
    if not isinstance(results, list):
        raise ValueError("evaluation results must be a list")

    expected_ids: list[str] = []
    question_by_id: dict[str, dict[str, object]] = {}
    for question in questions:
        if not isinstance(question, dict) or not isinstance(
            question.get("id"),
            str,
        ):
            raise ValueError("each evaluation question must have a string id")
        question_id = question["id"]
        expected_ids.append(question_id)
        question_by_id[question_id] = question #根据ID索引问题数据

    result_by_id: dict[str, dict[str, object]] = {}
    duplicate_result_ids: set[str] = set()
    result_ids: list[str] = []
    for result in results:
        if not isinstance(result, dict) or not isinstance(
            result.get("id"),
            str,
        ):
            raise ValueError("each evaluation result must have a string id")
        result_id = result["id"]
        result_ids.append(result_id)
        if result_id in result_by_id:
            duplicate_result_ids.add(result_id)
            continue
        result_by_id[result_id] = result  # 根据 ID 索引结果数据
        #得到exact-001
        #        ├─ 标准问题 question
        #        └─ 实际结果 result

    expected_id_set = set(expected_ids)
    result_id_set = set(result_ids)
    missing_ids = [
        question_id
        for question_id in expected_ids
        if question_id not in result_id_set
    ]
    unexpected_ids = sorted(result_id_set - expected_id_set)

    answer_type_correct = 0
    answer_points_matched = 0
    answer_points_total = 0
    knowledge_questions = 0
    answers_with_citations = 0
    citations_total = 0
    expected_path_citations = 0
    source_valid_citations = 0
    directory_questions = 0
    directory_correct = 0
    refusal_questions = 0
    refusal_correct = 0
    total_tool_calls = 0
    successful_tool_calls = 0
    error_tool_calls = 0
    knowledge_with_successful_read = 0
    latencies: list[float] = []
    token_reports = 0
    input_tokens = 0
    output_tokens = 0
    total_tokens = 0
    reasoning_tokens = 0
    per_question: list[dict[str, object]] = []

    for question_id in expected_ids:
        question = question_by_id[question_id]
        result = result_by_id.get(question_id)
        expected_answer_type = question.get("expected_answer_type")
        actual_answer_type = (
            result.get("answer_type") if result is not None else None
        )
        answer_type_matches = actual_answer_type == expected_answer_type
        answer_type_correct += int(answer_type_matches) #int(true) = 1, int(false) = 0

        answer = result.get("answer", "") if result is not None else ""
        answer_points = [
            point
            for point in question.get("answer_points", [])
            if isinstance(point, str)
        ] if isinstance(question.get("answer_points", []), list) else []
        matched_points = [
            point
            for point in answer_points
            if _answer_point_matches(point, answer)
        ]
        answer_points_total += len(answer_points)
        answer_points_matched += len(matched_points)

        expected_paths = [
            path
            for path in question.get("expected_paths", [])
            if isinstance(path, str)
        ] if isinstance(question.get("expected_paths", []), list) else []
        raw_citations = result.get("citations", []) if result else []
        citations = [
            citation
            for citation in raw_citations
            if isinstance(citation, dict)
        ] if isinstance(raw_citations, list) else []

        if expected_answer_type == "knowledge":
            knowledge_questions += 1
            answers_with_citations += int(bool(citations))

        citations_total += len(citations)
        expected_path_citations += sum(
            1
            for citation in citations
            if citation.get("path") in expected_paths
        )
        source_valid_citations += sum(
            1
            for citation in citations
            if _citation_source_is_valid(citation, knowledge_root)
        )

        directory_answer_correct: bool | None = None
        if expected_answer_type == "directory":
            directory_questions += 1
            expected_names = [
                path.rsplit("/", 1)[-1]
                for path in expected_paths
            ]
            normalized_answer = _normalize_text(answer)
            directory_answer_correct = (
                actual_answer_type == "directory"
                and bool(expected_names)
                and all(
                    _normalize_text(name) in normalized_answer
                    for name in expected_names
                )
            )
            directory_correct += int(directory_answer_correct)

        refusal_answer_correct: bool | None = None
        if expected_answer_type == "insufficient":
            refusal_questions += 1
            refusal_answer_correct = actual_answer_type == "insufficient"
            refusal_correct += int(refusal_answer_correct)

        raw_traces = result.get("tool_traces", []) if result else []
        traces = [
            trace
            for trace in raw_traces
            if isinstance(trace, dict)
        ] if isinstance(raw_traces, list) else []
        total_tool_calls += len(traces)
        successful_tool_calls += sum(
            1 for trace in traces if trace.get("status") == "success"
        )
        error_tool_calls += sum(
            1 for trace in traces if trace.get("status") == "error"
        )
        successful_read = any(
            trace.get("name") == "read"
            and trace.get("status") == "success"
            for trace in traces
        )
        if expected_answer_type == "knowledge":
            knowledge_with_successful_read += int(successful_read)

        latency = _number(result.get("latency_ms")) if result else None
        if latency is not None:
            latencies.append(latency)

        token_usage = result.get("token_usage") if result else None
        if isinstance(token_usage, dict):
            token_reports += 1
            input_tokens += int(_number(token_usage.get("input_tokens")) or 0)
            output_tokens += int(
                _number(token_usage.get("output_tokens")) or 0
            )
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
                "answer_type_correct": answer_type_matches,
                "answer_points_matched": matched_points,
                "answer_points_total": len(answer_points),
                "citation_count": len(citations),
                "directory_answer_correct": directory_answer_correct,
                "refusal_answer_correct": refusal_answer_correct,
                "tool_call_count": len(traces),
                "successful_read": successful_read,
                "latency_ms": latency,
            }
        )

    sorted_latencies = sorted(latencies)
    latency_count = len(sorted_latencies)
    latency_summary = {
        "count": latency_count,
        "minimum": sorted_latencies[0] if sorted_latencies else None,
        "average": (
            sum(sorted_latencies) / latency_count
            if latency_count
            else None
        ),
        "median": (
            statistics.median(sorted_latencies)
            if sorted_latencies
            else None
        ),
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
            "answer_points": "normalized_content_substring",
            "citations": "expected_path_and_exact_source_quote",
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
        "answer_types": {
            "eligible_questions": len(questions),
            "correct_answers": answer_type_correct,
            "accuracy": _rate(answer_type_correct, len(questions)),
        },
        "answer_points": {
            "matched": answer_points_matched,
            "total": answer_points_total,
            "coverage": _rate(answer_points_matched, answer_points_total),
        },
        "citations": {
            "knowledge_questions": knowledge_questions,
            "answers_with_citations": answers_with_citations,
            "answer_coverage": _rate(
                answers_with_citations,
                knowledge_questions,
            ),
            "citations_total": citations_total,
            "expected_path_citations": expected_path_citations,
            "expected_path_accuracy": _rate(
                expected_path_citations,
                citations_total,
            ),
            "source_valid_citations": source_valid_citations,
            "source_validity": _rate(
                source_valid_citations,
                citations_total,
            ),
        },
        "directory": {
            "eligible_questions": directory_questions,
            "correct_answers": directory_correct,
            "accuracy": _rate(directory_correct, directory_questions),
        },
        "refusal": {
            "eligible_questions": refusal_questions,
            "correct_refusals": refusal_correct,
            "accuracy": _rate(refusal_correct, refusal_questions),
        },
        "tools": {
            "total_calls": total_tool_calls,
            "successful_calls": successful_tool_calls,
            "error_calls": error_tool_calls,
            "average_calls_per_question": _rate(
                total_tool_calls,
                len(questions),
            ),
            "knowledge_questions": knowledge_questions,
            "knowledge_questions_with_successful_read": (
                knowledge_with_successful_read
            ),
            "knowledge_read_compliance": _rate(
                knowledge_with_successful_read,
                knowledge_questions,
            ),
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


def _load_json_object(path: Path) -> dict[str, object]:
    """读取 UTF-8 JSON 文件，并确保最外层是对象。"""

    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return data


def score_agentic_files(
    dataset_path: Path,
    result_path: Path,
    knowledge_root: Path,
    output_path: Path,
) -> Path:
    """读取 Agentic 评测文件，评分后写入独立的 JSON 文件。"""

    dataset = _load_json_object(dataset_path)
    run_result = _load_json_object(result_path)
    score = score_agentic_evaluation(
        dataset=dataset,
        run_result=run_result,
        knowledge_root=knowledge_root,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(score, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return output_path


def build_agentic_score_filename(result_filename: str) -> str:
    """根据 Agentic 结果文件名生成对应的独立评分文件名。"""

    result_path = Path(result_filename)
    return f"{result_path.stem}-score.json"


def main(project_root: Path | None = None) -> None:
    """评分项目中的当前 Agentic 结果，并打印评分文件路径。"""

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
    result_filename = build_agentic_evaluation_result_filename(
        TraditionalRagConfig(),
        KNOWLEDGE_AGENT_PROMPT_VERSION,
    )
    output_filename = build_agentic_score_filename(result_filename)
    output_path = score_agentic_files(
        dataset_path=(
            resolved_project_root / "evaluation" / "questions.json"
        ),
        result_path=(
            resolved_project_root
            / "evaluation"
            / "results"
            / result_filename
        ),
        knowledge_root=(
            resolved_project_root / "knowledge" / "education-v1"
        ),
        output_path=(
            resolved_project_root
            / "evaluation"
            / "results"
            / output_filename
        ),
    )
    print(f"Agentic evaluation score saved to: {output_path}")


if __name__ == "__main__":
    main()
