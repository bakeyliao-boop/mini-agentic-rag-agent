"""比较 Agentic RAG 优化前后的质量与成本指标。"""

import json
from pathlib import Path

BEFORE_SCORE_FILENAME = (
    "agentic-baseline-qwen3.6-flash-thinking-off-"
    "prompt-v1.6.pre-retrieval-stop-score.json"
)
AFTER_SCORE_FILENAME = (
    "agentic-baseline-qwen3.6-flash-thinking-off-"
    "prompt-v1.6-score.json"
)
BEFORE_SEMANTIC_SCORE_FILENAME = (
    "agentic-baseline-qwen3.6-flash-thinking-off-"
    "prompt-v1.6.pre-retrieval-stop-semantic-score.json"
)
AFTER_SEMANTIC_SCORE_FILENAME = (
    "agentic-baseline-qwen3.6-flash-thinking-off-"
    "prompt-v1.6-semantic-score.json"
)
OPTIMIZATION_REPORT_FILENAME = (
    "agentic-optimization-comparison-qwen3.6-flash-"
    "thinking-off-prompt-v1.6.md"
)


def _nested_number(
    data: dict[str, object],
    *keys: str,
) -> float:
    """按照字段路径读取数值，并拒绝缺失或非数值字段。"""

    current: object = data
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            raise ValueError(f"missing score field: {'.'.join(keys)}")
        current = current[key]

    if isinstance(current, bool) or not isinstance(current, int | float):
        raise ValueError(f"score field must be numeric: {'.'.join(keys)}")
    return float(current)


def _compare_metric(
    before: float,
    after: float,
) -> dict[str, float]:
    """记录一个指标优化前、优化后及两者差值。"""

    return {
        "before": before,
        "after": after,
        "delta": round(after - before, 10),
    }


def compare_agentic_optimization_scores(
    before_score: dict[str, object],
    after_score: dict[str, object],
) -> dict[str, dict[str, dict[str, float]]]:
    """对齐两份 Agentic 评分中的质量与效率指标。"""

    quality_paths = {
        "answer_type_accuracy": ("answer_types", "accuracy"),
        "answer_point_coverage": ("answer_points", "coverage"),
        "citation_source_validity": ("citations", "source_validity"),
        "directory_accuracy": ("directory", "accuracy"),
        "refusal_accuracy": ("refusal", "accuracy"),
    }
    efficiency_paths = {
        "total_tool_calls": ("tools", "total_calls"),
        "error_tool_calls": ("tools", "error_calls"),
        "total_tokens": ("performance", "tokens", "total"),
        "average_latency_ms": (
            "performance",
            "latency_ms",
            "average",
        ),
    }

    return {
        "quality": {
            name: _compare_metric(
                _nested_number(before_score, *path),
                _nested_number(after_score, *path),
            )
            for name, path in quality_paths.items()
        },
        "efficiency": {
            name: _compare_metric(
                _nested_number(before_score, *path),
                _nested_number(after_score, *path),
            )
            for name, path in efficiency_paths.items()
        },
    }


def _load_json_object(source_path: Path) -> dict[str, object]:
    """读取 UTF-8 JSON 文件，并确保最外层是对象。"""

    data = json.loads(source_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"JSON root must be an object: {source_path}")
    return data


def compare_agentic_optimization_files(
    before_score_path: Path,
    after_score_path: Path,
    before_semantic_score_path: Path | None = None,
    after_semantic_score_path: Path | None = None,
) -> dict[str, dict[str, dict[str, float]]]:
    """读取优化前后评分文件，并返回结构化比较结果。"""

    comparison = compare_agentic_optimization_scores(
        _load_json_object(before_score_path),
        _load_json_object(after_score_path),
    )
    if (before_semantic_score_path is None) != (
        after_semantic_score_path is None
    ):
        raise ValueError("semantic score paths must be provided together")

    if (
        before_semantic_score_path is not None
        and after_semantic_score_path is not None
    ):
        before_semantic = _load_json_object(before_semantic_score_path)
        after_semantic = _load_json_object(after_semantic_score_path)
        comparison["quality"]["semantic_answer_point_coverage"] = (
            _compare_metric(
                _nested_number(
                    before_semantic,
                    "answer_points",
                    "coverage",
                ),
                _nested_number(
                    after_semantic,
                    "answer_points",
                    "coverage",
                ),
            )
        )
    return comparison


def _percentage_row(
    label: str,
    metric: dict[str, float],
) -> str:
    """把百分比指标渲染为一行 Markdown 表格。"""

    return (
        f"| {label} | {metric['before']:.1%} | {metric['after']:.1%} | "
        f"{metric['delta'] * 100:+.1f} 个百分点 |"
    )


def _integer_row(
    label: str,
    metric: dict[str, float],
) -> str:
    """把整数指标渲染为一行 Markdown 表格。"""

    return (
        f"| {label} | {metric['before']:.0f} | {metric['after']:.0f} | "
        f"{metric['delta']:+.0f} |"
    )


def _decrease_percentage(metric: dict[str, float]) -> float:
    """计算一个成本指标从优化前到优化后的下降比例。"""

    before = metric["before"]
    if before == 0:
        return 0.0
    return (before - metric["after"]) / before


def render_agentic_optimization_markdown(
    comparison: dict[str, dict[str, dict[str, float]]],
) -> str:
    """把 Agentic RAG 优化前后指标渲染为 Markdown 报告。"""

    quality = comparison.get("quality")
    efficiency = comparison.get("efficiency")
    if not isinstance(quality, dict) or not isinstance(efficiency, dict):
        raise ValueError("invalid agentic optimization comparison")

    tool_calls = efficiency["total_tool_calls"]
    error_calls = efficiency["error_tool_calls"]
    total_tokens = efficiency["total_tokens"]
    average_latency = efficiency["average_latency_ms"]
    tool_call_decrease = _decrease_percentage(tool_calls)
    token_decrease = _decrease_percentage(total_tokens)

    lines = [
        "# Agentic RAG 阶段 9 优化前后对比报告",
        "",
        "## 质量指标",
        "",
        "| 指标 | 优化前 | 优化后 | 变化 |",
        "|---|---:|---:|---:|",
        _percentage_row(
            "回答类型准确率",
            quality["answer_type_accuracy"],
        ),
        _percentage_row(
            "答案点覆盖率",
            quality["answer_point_coverage"],
        ),
        *(
            [
                _percentage_row(
                    "独立语义答案点覆盖率",
                    quality["semantic_answer_point_coverage"],
                )
            ]
            if "semantic_answer_point_coverage" in quality
            else []
        ),
        _percentage_row(
            "引用来源有效率",
            quality["citation_source_validity"],
        ),
        _percentage_row("目录题准确率", quality["directory_accuracy"]),
        _percentage_row("拒答准确率", quality["refusal_accuracy"]),
        "",
        "## 效率指标",
        "",
        "| 指标 | 优化前 | 优化后 | 变化 |",
        "|---|---:|---:|---:|",
        _integer_row("工具调用总数", tool_calls),
        _integer_row("工具错误数", error_calls),
        _integer_row("总 Token", total_tokens),
        (
            "| 平均延迟 | "
            f"{average_latency['before']:.1f} ms | "
            f"{average_latency['after']:.1f} ms | "
            f"{average_latency['delta']:+.1f} ms |"
        ),
        "",
        "## 结论",
        "",
        (
            "- 本阶段为 `search` 结果增加了基于检索分数的三态判断，"
            "由系统中间件决定 Agent 继续检索还是立即停止："
        ),
        "  - `relevant`：继续调用 `read` 获取原文证据。",
        "  - `uncertain`：允许 Agent 继续探索。",
        "  - `none`：中间件立即停止，并返回证据不足。",
        "- 回答类型、引用、目录题和拒答质量没有回退。",
        *(
            ["- 独立语义裁判确认答案点覆盖率没有回退。"]
            if "semantic_answer_point_coverage" in quality
            else []
        ),
        f"- 工具调用总数下降 {tool_call_decrease:.1%}。",
        f"- 总 Token 下降 {token_decrease:.1%}。",
        (
            "- 延迟仅记录观测值，不作为本轮优化的硬性通过条件；"
            "两次评测跨时间运行，远程模型服务和网络波动会影响结果。"
        ),
        "",
    ]
    return "\n".join(lines)


def write_agentic_optimization_report(
    before_score_path: Path,
    after_score_path: Path,
    output_path: Path,
    before_semantic_score_path: Path | None = None,
    after_semantic_score_path: Path | None = None,
) -> Path:
    """比较优化前后评分文件，并保存 Markdown 报告。"""

    comparison = compare_agentic_optimization_files(
        before_score_path,
        after_score_path,
        before_semantic_score_path,
        after_semantic_score_path,
    )
    markdown = render_agentic_optimization_markdown(comparison)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        markdown,
        encoding="utf-8",
        newline="\n",
    )
    return output_path


def main(project_root: Path | None = None) -> None:
    """生成项目当前阶段 9 的 Agentic 优化前后对比报告。"""

    resolved_project_root = (
        project_root
        if project_root is not None
        else Path(__file__).resolve().parent.parent
    )
    results_directory = resolved_project_root / "evaluation" / "results"
    output_path = write_agentic_optimization_report(
        before_score_path=results_directory / BEFORE_SCORE_FILENAME,
        after_score_path=results_directory / AFTER_SCORE_FILENAME,
        output_path=results_directory / OPTIMIZATION_REPORT_FILENAME,
        before_semantic_score_path=(
            results_directory / BEFORE_SEMANTIC_SCORE_FILENAME
        ),
        after_semantic_score_path=(
            results_directory / AFTER_SEMANTIC_SCORE_FILENAME
        ),
    )
    print(f"Agentic optimization report saved to: {output_path}")


if __name__ == "__main__":
    main()
