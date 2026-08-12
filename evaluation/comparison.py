"""离线比较传统 RAG 与 Agentic RAG 的评测结果。"""

import json
import math
import unicodedata
from pathlib import Path

TRADITIONAL_SCORE_FILENAME = (
    "traditional-baseline-qwen3.6-flash-thinking-off-score.json"
)
TRADITIONAL_SEMANTIC_SCORE_FILENAME = (
    "traditional-baseline-qwen3.6-flash-thinking-off-"
    "semantic-score.json"
)
AGENTIC_SCORE_FILENAME = (
    "agentic-baseline-qwen3.6-flash-thinking-off-"
    "prompt-v1.5-score.json"
)
AGENTIC_SEMANTIC_SCORE_FILENAME = (
    "agentic-baseline-qwen3.6-flash-thinking-off-"
    "prompt-v1.5-semantic-score.json"
)
COMPARISON_REPORT_FILENAME = (
    "rag-comparison-qwen3.6-flash-thinking-off-prompt-v1.5.md"
)


def _nested_number(
    data: dict[str, object],
    *keys: str,
) -> float:
    """按字段路径读取一个数值。"""

    current: object = data
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            raise ValueError(f"missing score field: {'.'.join(keys)}")
        current = current[key]
    if isinstance(current, bool) or not isinstance(current, int | float):
        raise ValueError(f"score field must be numeric: {'.'.join(keys)}")
    return float(current)


def _compare_metric(
    traditional: float,
    agentic: float,
    *,
    higher_is_better: bool,
) -> dict[str, object]:
    """计算一个共同指标的差值和更优方案。"""

    delta = round(agentic - traditional, 10)
    if math.isclose(traditional, agentic):
        better = "tie"
    elif (agentic > traditional) == higher_is_better:
        better = "agentic"
    else:
        better = "traditional"
    return {
        "traditional": traditional,
        "agentic": agentic,
        "delta": delta,
        "better": better,
    }


def compare_evaluation_scores(
    traditional_score: dict[str, object],
    agentic_score: dict[str, object],
    traditional_semantic_score: dict[str, object] | None = None,
    agentic_semantic_score: dict[str, object] | None = None,
) -> dict[str, object]:
    """对齐两种 RAG 的共同指标，并保留 Agentic 专属指标。"""

    if (traditional_semantic_score is None) != (
        agentic_semantic_score is None
    ):
        raise ValueError("both semantic score files must be provided")

    uses_semantic_scores = traditional_semantic_score is not None
    traditional_answer_score = (
        traditional_semantic_score
        if traditional_semantic_score is not None
        else traditional_score
    )
    agentic_answer_score = (
        agentic_semantic_score
        if agentic_semantic_score is not None
        else agentic_score
    )

    common_metrics = {
        "answer_point_coverage": _compare_metric(
            _nested_number(
                traditional_answer_score,
                "answer_points",
                "coverage",
            ),
            _nested_number(
                agentic_answer_score,
                "answer_points",
                "coverage",
            ),
            higher_is_better=True,
        ),
        "directory_accuracy": _compare_metric(
            _nested_number(traditional_score, "directory", "accuracy"),
            _nested_number(agentic_score, "directory", "accuracy"),
            higher_is_better=True,
        ),
        "refusal_accuracy": _compare_metric(
            _nested_number(traditional_score, "refusal", "accuracy"),
            _nested_number(agentic_score, "refusal", "accuracy"),
            higher_is_better=True,
        ),
        "average_latency_ms": _compare_metric(
            _nested_number(
                traditional_score,
                "performance",
                "latency_ms",
                "average",
            ),
            _nested_number(
                agentic_score,
                "performance",
                "latency_ms",
                "average",
            ),
            higher_is_better=False,
        ),
        "average_tokens": _compare_metric(
            _nested_number(
                traditional_score,
                "performance",
                "tokens",
                "average_total",
            ),
            _nested_number(
                agentic_score,
                "performance",
                "tokens",
                "average_total",
            ),
            higher_is_better=False,
        ),
    }
    agentic_metrics = {
        "answer_type_accuracy": _nested_number(
            agentic_score,
            "answer_types",
            "accuracy",
        ),
        "citation_answer_coverage": _nested_number(
            agentic_score,
            "citations",
            "answer_coverage",
        ),
        "citation_source_validity": _nested_number(
            agentic_score,
            "citations",
            "source_validity",
        ),
        "knowledge_read_compliance": _nested_number(
            agentic_score,
            "tools",
            "knowledge_read_compliance",
        ),
        "average_tool_calls": _nested_number(
            agentic_score,
            "tools",
            "average_calls_per_question",
        ),
        "total_tool_calls": _nested_number(
            agentic_score,
            "tools",
            "total_calls",
        ),
        "error_tool_calls": _nested_number(
            agentic_score,
            "tools",
            "error_calls",
        ),
    }
    corpus_id = traditional_score.get("corpus_id", "unknown")
    if not isinstance(corpus_id, str):
        raise ValueError("corpus_id must be a string")
    evaluation_scope = {
        "corpus_id": corpus_id,
        "question_count": int(
            _nested_number(
                traditional_score,
                "completeness",
                "question_count",
            )
        ),
        "knowledge_question_count": (
            int(
                _nested_number(
                    traditional_answer_score,
                    "answer_points",
                    "eligible_questions",
                )
            )
            if uses_semantic_scores
            else 0
        ),
        "answer_point_count": (
            int(
                _nested_number(
                    traditional_answer_score,
                    "answer_points",
                    "total",
                )
            )
            if uses_semantic_scores
            else 0
        ),
        "answer_scoring": (
            "llm_semantic_judge"
            if uses_semantic_scores
            else "normalized_content_substring"
        ),
    }
    latency = common_metrics["average_latency_ms"]
    tokens = common_metrics["average_tokens"]
    cost_ratios = {
        "latency": float(latency["agentic"]) / float(latency["traditional"]),
        "tokens": float(tokens["agentic"]) / float(tokens["traditional"]),
    }
    return {
        "evaluation_scope": evaluation_scope,
        "common_metrics": common_metrics,
        "agentic_metrics": agentic_metrics,
        "cost_ratios": cost_ratios,
    }


def _winner_label(value: object) -> str:
    """把内部方案名称转成报告中的中文名称。"""

    return {
        "traditional": "传统 RAG",
        "agentic": "Agentic RAG",
        "tie": "相同",
    }.get(value, "未知")


def _percentage_row(
    label: str,
    metric: dict[str, object],
) -> list[str]:
    """生成百分比指标的单元格。"""

    traditional = float(metric["traditional"])
    agentic = float(metric["agentic"])
    delta = float(metric["delta"])
    return [
        label,
        f"{traditional:.1%}",
        f"{agentic:.1%}",
        f"{delta * 100:+.1f} 个百分点",
        _winner_label(metric["better"]),
    ]


def _number_row(
    label: str,
    metric: dict[str, object],
    unit: str,
) -> list[str]:
    """生成普通数值指标的单元格。"""

    traditional = float(metric["traditional"])
    agentic = float(metric["agentic"])
    delta = float(metric["delta"])
    return [
        label,
        f"{traditional:.1f} {unit}",
        f"{agentic:.1f} {unit}",
        f"{delta:+.1f} {unit}",
        _winner_label(metric["better"]),
    ]


def _display_width(value: str) -> int:
    """计算文本在终端中的显示宽度，中文字符按两格计算。"""

    return sum(
        2 if unicodedata.east_asian_width(character) in "WFA" else 1
        for character in value
    )


def _pad_cell(value: str, width: int, *, right: bool) -> str:
    """按照显示宽度为一个表格单元格补齐空格。"""

    padding = " " * (width - _display_width(value))
    return f"{padding}{value}" if right else f"{value}{padding}"


def _render_markdown_table(
    headers: list[str],
    rows: list[list[str]],
    *,
    right_aligned_columns: set[int],
) -> str:
    """生成在 Markdown 源文件中也能保持列对齐的表格。"""

    widths = [
        max(
            3,
            *(
                _display_width(row[column_index])
                for row in [headers, *rows]
            ),
        )
        for column_index in range(len(headers))
    ]

    def render_row(row: list[str]) -> str:
        cells = [
            _pad_cell(
                value,
                widths[index],
                right=index in right_aligned_columns,
            )
            for index, value in enumerate(row)
        ]
        return "| " + " | ".join(cells) + " |"

    separator_cells = [
        (
            "-" * (width + 1) + ":"
            if index in right_aligned_columns
            else "-" * (width + 2)
        )
        for index, width in enumerate(widths)
    ]
    separator = "|" + "|".join(separator_cells) + "|"
    return "\n".join(
        [
            render_row(headers),
            separator,
            *(render_row(row) for row in rows),
        ]
    )


def render_comparison_markdown(
    comparison: dict[str, object],
) -> str:
    """把结构化对比结果渲染为正式 Markdown 报告。"""

    scope = comparison["evaluation_scope"]
    common = comparison["common_metrics"]
    agentic = comparison["agentic_metrics"]
    cost_ratios = comparison["cost_ratios"]
    if not all(
        isinstance(section, dict)
        for section in (scope, common, agentic, cost_ratios)
    ):
        raise ValueError("invalid evaluation comparison")

    semantic_scoring = scope["answer_scoring"] == "llm_semantic_judge"
    answer_metric_label = (
        "知识答案语义覆盖率"
        if semantic_scoring
        else "答案点覆盖率"
    )
    common_rows = [
        _percentage_row(
            answer_metric_label,
            common["answer_point_coverage"],
        ),
        _percentage_row("目录题准确率", common["directory_accuracy"]),
        _percentage_row("拒答准确率", common["refusal_accuracy"]),
        _number_row("平均延迟", common["average_latency_ms"], "ms"),
        _number_row("平均 Token", common["average_tokens"], "tokens"),
    ]
    agentic_rows = [
        ["回答类型准确率", f"{float(agentic['answer_type_accuracy']):.1%}"],
        ["知识题引用覆盖率", f"{float(agentic['citation_answer_coverage']):.1%}"],
        ["引用原文有效率", f"{float(agentic['citation_source_validity']):.1%}"],
        ["read 合规率", f"{float(agentic['knowledge_read_compliance']):.1%}"],
        ["平均工具调用次数", f"{float(agentic['average_tool_calls']):.1f}"],
        ["工具调用总数", f"{float(agentic['total_tool_calls']):.0f}"],
        ["工具错误数", f"{float(agentic['error_tool_calls']):.0f}"],
    ]
    common_table = _render_markdown_table(
        [
            "指标",
            "传统 RAG",
            "Agentic RAG",
            "差值（Agentic - 传统）",
            "更优",
        ],
        common_rows,
        right_aligned_columns={1, 2, 3},
    )
    agentic_table = _render_markdown_table(
        ["指标", "数值"],
        agentic_rows,
        right_aligned_columns={1},
    )

    answer_metric = common["answer_point_coverage"]
    directory_metric = common["directory_accuracy"]
    refusal_metric = common["refusal_accuracy"]
    latency_ratio = float(cost_ratios["latency"])
    token_ratio = float(cost_ratios["tokens"])
    if answer_metric["better"] == "tie":
        answer_conclusion = (
            "当前固定题集中，两种方案的知识答案正确性持平"
        )
    else:
        answer_conclusion = (
            f"知识答案正确性由{_winner_label(answer_metric['better'])}领先"
        )
    directory_conclusion = (
        "目录题由"
        f"{_winner_label(directory_metric['better'])}表现更好"
        if directory_metric["better"] != "tie"
        else "目录题表现相同"
    )
    refusal_conclusion = (
        "两种方案都能正确拒答知识库外问题"
        if refusal_metric["better"] == "tie"
        and math.isclose(float(refusal_metric["traditional"]), 1.0)
        else f"拒答题由{_winner_label(refusal_metric['better'])}表现更好"
    )

    return "\n".join(
        [
            "# 传统 RAG 与 Agentic RAG A/B 对比报告",
            "",
            "## 评测范围",
            "",
            f"- 语料版本：`{scope['corpus_id']}`",
            f"- 固定问题数：{int(scope['question_count'])}",
            (
                "- 知识问答："
                f"{int(scope['knowledge_question_count'])} 题，"
                f"共 {int(scope['answer_point_count'])} 个答案点"
            ),
            "- 回答模型：Qwen3.6-Flash，关闭思考模式",
            "- Agent 提示词：Prompt V1.5",
            "- 知识答案评分：独立 LLM 语义裁判",
            "- 目录、拒答、引用与性能：确定性评分",
            "",
            "## 核心指标",
            "",
            common_table,
            "",
            "## Agentic 专属指标",
            "",
            agentic_table,
            "",
            "## 结论",
            "",
            f"1. {answer_conclusion}。",
            f"2. {directory_conclusion}。",
            f"3. {refusal_conclusion}。",
            (
                "4. Agentic RAG 的平均延迟约为传统 RAG 的 "
                f"{latency_ratio:.2f} 倍，平均 Token 约为 "
                f"{token_ratio:.1f} 倍。"
            ),
            (
                "5. Agentic RAG 的知识题引用覆盖率、引用原文有效率"
                "和 read 合规率均可由服务端验证。"
            ),
            "",
            "## 适用建议",
            "",
            "- 简单、单跳知识问答：优先传统 RAG，延迟和 Token 成本更低。",
            (
                "- 目录定位、多步查找、需要精确引用的问答：优先 "
                "Agentic RAG。"
            ),
            "- 当前实验不支持用 Agentic RAG 全面替代传统 RAG。",
            "",
            "## 评测边界",
            "",
            (
                f"本报告结论仅适用于 `{scope['corpus_id']}` 语料和当前 "
                f"{int(scope['question_count'])} 道固定问题。"
            ),
            "",
        ]
    )


def _load_json_object(path: Path) -> dict[str, object]:
    """读取 UTF-8 JSON，并确保最外层是对象。"""

    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return data


def compare_evaluation_files(
    traditional_score_path: Path,
    agentic_score_path: Path,
    output_path: Path,
    traditional_semantic_score_path: Path | None = None,
    agentic_semantic_score_path: Path | None = None,
) -> Path:
    """读取确定性与语义评分文件并写入 Markdown 对比报告。"""

    traditional_semantic_score = (
        _load_json_object(traditional_semantic_score_path)
        if traditional_semantic_score_path is not None
        else None
    )
    agentic_semantic_score = (
        _load_json_object(agentic_semantic_score_path)
        if agentic_semantic_score_path is not None
        else None
    )

    comparison = compare_evaluation_scores(
        traditional_score=_load_json_object(traditional_score_path),
        agentic_score=_load_json_object(agentic_score_path),
        traditional_semantic_score=traditional_semantic_score,
        agentic_semantic_score=agentic_semantic_score,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        render_comparison_markdown(comparison),
        encoding="utf-8",
        newline="\n",
    )
    return output_path


def main(project_root: Path | None = None) -> None:
    """比较项目中的当前评分文件，并打印报告路径。"""

    resolved_project_root = (
        project_root
        if project_root is not None
        else Path(__file__).resolve().parent.parent
    )
    results_directory = resolved_project_root / "evaluation" / "results"
    output_path = compare_evaluation_files(
        traditional_score_path=(
            results_directory / TRADITIONAL_SCORE_FILENAME
        ),
        agentic_score_path=results_directory / AGENTIC_SCORE_FILENAME,
        traditional_semantic_score_path=(
            results_directory / TRADITIONAL_SEMANTIC_SCORE_FILENAME
        ),
        agentic_semantic_score_path=(
            results_directory / AGENTIC_SEMANTIC_SCORE_FILENAME
        ),
        output_path=results_directory / COMPARISON_REPORT_FILENAME,
    )
    print(f"Evaluation comparison saved to: {output_path}")


if __name__ == "__main__":
    main()
