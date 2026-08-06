import json
import unicodedata
from importlib import import_module
from pathlib import Path


def _traditional_score() -> dict[str, object]:
    """构造一份最小的传统 RAG 评分。"""

    return {
        "answer_points": {"coverage": 0.56},
        "directory": {"accuracy": 0.0},
        "refusal": {"accuracy": 1.0},
        "performance": {
            "latency_ms": {"average": 2686.0},
            "tokens": {"average_total": 980.8},
        },
    }


def _agentic_score() -> dict[str, object]:
    """构造一份最小的 Agentic RAG 评分。"""

    return {
        "answer_types": {"accuracy": 0.8},
        "answer_points": {"coverage": 0.48},
        "citations": {
            "answer_coverage": 0.875,
            "source_validity": 1.0,
        },
        "directory": {"accuracy": 0.0},
        "refusal": {"accuracy": 1.0},
        "tools": {
            "knowledge_read_compliance": 0.875,
            "average_calls_per_question": 3.2,
        },
        "performance": {
            "latency_ms": {"average": 3939.0},
            "tokens": {"average_total": 8575.5},
        },
    }


def test_compare_evaluation_scores_builds_markdown_table() -> None:
    """对比器应对齐共同指标，并生成可直接阅读的 Markdown 表格。"""

    comparison_module = import_module("app.evaluation_comparison")

    comparison = comparison_module.compare_evaluation_scores(
        traditional_score=_traditional_score(),
        agentic_score=_agentic_score(),
    )
    markdown = comparison_module.render_comparison_markdown(comparison)

    assert comparison["common_metrics"]["answer_point_coverage"] == {
        "traditional": 0.56,
        "agentic": 0.48,
        "delta": -0.08,
        "better": "traditional",
    }
    assert comparison["common_metrics"]["directory_accuracy"]["better"] == (
        "tie"
    )
    assert comparison["agentic_metrics"]["citation_source_validity"] == 1.0
    assert "答案点覆盖率" in markdown
    assert "-8.0 个百分点" in markdown
    assert "引用原文有效率" in markdown
    assert "100.0%" in markdown


def test_render_comparison_markdown_aligns_raw_table_columns() -> None:
    """Markdown 源文件中的每一列也应具有相同的显示宽度。"""

    comparison_module = import_module("app.evaluation_comparison")
    comparison = comparison_module.compare_evaluation_scores(
        traditional_score=_traditional_score(),
        agentic_score=_agentic_score(),
    )

    markdown = comparison_module.render_comparison_markdown(comparison)

    tables = [
        block.splitlines()
        for block in markdown.split("\n\n")
        if block.startswith("|")
    ]
    for table in tables:
        row_widths = [
            [
                sum(
                    2
                    if unicodedata.east_asian_width(character) in "WFA"
                    else 1
                    for character in cell
                )
                for cell in row.split("|")[1:-1]
            ]
            for row in table
        ]
        expected_widths = row_widths[0]
        assert all(widths == expected_widths for widths in row_widths)


def test_main_writes_versioned_comparison_report(
    tmp_path: Path,
    capsys,
) -> None:
    """命令行入口应读取当前两份评分并写入版本化对比报告。"""

    comparison_module = import_module("app.evaluation_comparison")
    results_directory = tmp_path / "evaluation" / "results"
    results_directory.mkdir(parents=True)
    traditional_filename = (
        "traditional-baseline-qwen3.6-flash-thinking-off-score.json"
    )
    agentic_filename = (
        "agentic-baseline-qwen3.6-flash-thinking-off-"
        "prompt-v1.2-score.json"
    )
    (results_directory / traditional_filename).write_text(
        json.dumps(_traditional_score()),
        encoding="utf-8",
    )
    (results_directory / agentic_filename).write_text(
        json.dumps(_agentic_score()),
        encoding="utf-8",
    )

    comparison_module.main(project_root=tmp_path)

    expected_path = results_directory / (
        "rag-comparison-qwen3.6-flash-thinking-off-prompt-v1.2.md"
    )
    assert expected_path.is_file()
    assert "答案点覆盖率" in expected_path.read_text(encoding="utf-8")
    assert capsys.readouterr().out == (
        f"Evaluation comparison saved to: {expected_path}\n"
    )
