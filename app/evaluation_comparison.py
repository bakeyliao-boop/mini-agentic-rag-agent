"""离线比较传统 RAG 与 Agentic RAG 的评分结果。"""

import json
import math
import unicodedata
from pathlib import Path

TRADITIONAL_SCORE_FILENAME = (
    "traditional-baseline-qwen3.6-flash-thinking-off-score.json"
)
AGENTIC_SCORE_FILENAME = (
    "agentic-baseline-qwen3.6-flash-thinking-off-"
    "prompt-v1.2-score.json"
)
COMPARISON_REPORT_FILENAME = (
    "rag-comparison-qwen3.6-flash-thinking-off-prompt-v1.2.md"
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
) -> dict[str, object]:
    """对齐两种 RAG 的共同指标，并保留 Agentic 专属指标。"""

    common_metrics = {
        "answer_point_coverage": _compare_metric(
            _nested_number(
                traditional_score,
                "answer_points",
                "coverage",
            ),
            _nested_number(agentic_score, "answer_points", "coverage"),
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
    }
    return {
        "common_metrics": common_metrics,
        "agentic_metrics": agentic_metrics,
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
    """把结构化对比结果渲染为 Markdown 表格。"""

    common = comparison["common_metrics"]
    agentic = comparison["agentic_metrics"]
    if not isinstance(common, dict) or not isinstance(agentic, dict):
        raise ValueError("invalid evaluation comparison")

    common_rows = [
        _percentage_row("答案点覆盖率", common["answer_point_coverage"]),
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

    return "\n".join(
        [
            "# 传统 RAG 与 Agentic RAG 自动对比",
            "",
            "## 共同指标",
            "",
            common_table,
            "",
            "## Agentic 专属指标",
            "",
            agentic_table,
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
) -> Path:
    """读取两份评分文件并写入 Markdown 对比报告。"""

    comparison = compare_evaluation_scores(
        traditional_score=_load_json_object(traditional_score_path),
        agentic_score=_load_json_object(agentic_score_path),
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
        output_path=results_directory / COMPARISON_REPORT_FILENAME,
    )
    print(f"Evaluation comparison saved to: {output_path}")


if __name__ == "__main__":
    main()
