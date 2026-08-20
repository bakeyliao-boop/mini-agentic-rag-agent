"""Agentic RAG 优化前后对比测试。"""

import json
from importlib import import_module
from pathlib import Path


def test_compare_agentic_optimization_scores_reports_quality_and_cost() -> None:
    """比较结果应同时保留质量、工具调用、Token 和延迟变化。"""

    comparison_module = import_module(
        "evaluation.optimization_comparison"
    )
    before_score = {
        "answer_types": {"accuracy": 1.0},
        "answer_points": {"coverage": 0.56},
        "citations": {"source_validity": 1.0},
        "directory": {"accuracy": 1.0},
        "refusal": {"accuracy": 1.0},
        "tools": {
            "total_calls": 28,
            "error_calls": 2,
        },
        "performance": {
            "tokens": {"total": 112_731},
            "latency_ms": {"average": 3_621.9},
        },
    }
    after_score = {
        "answer_types": {"accuracy": 1.0},
        "answer_points": {"coverage": 0.64},
        "citations": {"source_validity": 1.0},
        "directory": {"accuracy": 1.0},
        "refusal": {"accuracy": 1.0},
        "tools": {
            "total_calls": 22,
            "error_calls": 1,
        },
        "performance": {
            "tokens": {"total": 83_180},
            "latency_ms": {"average": 7_509.0},
        },
    }

    result = comparison_module.compare_agentic_optimization_scores(
        before_score,
        after_score,
    )

    assert result["quality"]["answer_type_accuracy"] == {
        "before": 1.0,
        "after": 1.0,
        "delta": 0.0,
    }
    assert result["quality"]["answer_point_coverage"] == {
        "before": 0.56,
        "after": 0.64,
        "delta": 0.08,
    }
    assert result["quality"]["citation_source_validity"] == {
        "before": 1.0,
        "after": 1.0,
        "delta": 0.0,
    }
    assert result["efficiency"]["total_tool_calls"] == {
        "before": 28.0,
        "after": 22.0,
        "delta": -6.0,
    }
    assert result["efficiency"]["error_tool_calls"] == {
        "before": 2.0,
        "after": 1.0,
        "delta": -1.0,
    }
    assert result["efficiency"]["total_tokens"] == {
        "before": 112_731.0,
        "after": 83_180.0,
        "delta": -29_551.0,
    }
    assert result["efficiency"]["average_latency_ms"] == {
        "before": 3_621.9,
        "after": 7_509.0,
        "delta": 3_887.1,
    }


def test_compare_agentic_optimization_files_loads_two_score_files(
    tmp_path: Path,
) -> None:
    """文件入口应读取优化前后评分文件并返回结构化比较。"""

    comparison_module = import_module(
        "evaluation.optimization_comparison"
    )

    def build_score(tool_calls: int, total_tokens: int) -> dict[str, object]:
        return {
            "answer_types": {"accuracy": 1.0},
            "answer_points": {"coverage": 0.6},
            "citations": {"source_validity": 1.0},
            "directory": {"accuracy": 1.0},
            "refusal": {"accuracy": 1.0},
            "tools": {
                "total_calls": tool_calls,
                "error_calls": 0,
            },
            "performance": {
                "tokens": {"total": total_tokens},
                "latency_ms": {"average": 1_000.0},
            },
        }

    before_path = tmp_path / "before-score.json"
    after_path = tmp_path / "after-score.json"
    before_path.write_text(
        json.dumps(build_score(28, 112_731)),
        encoding="utf-8",
    )
    after_path.write_text(
        json.dumps(build_score(22, 83_180)),
        encoding="utf-8",
    )

    result = comparison_module.compare_agentic_optimization_files(
        before_path,
        after_path,
    )

    assert result["efficiency"]["total_tool_calls"]["delta"] == -6.0
    assert result["efficiency"]["total_tokens"]["delta"] == -29_551.0


def test_compare_agentic_optimization_files_includes_semantic_coverage(
    tmp_path: Path,
) -> None:
    """文件比较应纳入同一语义裁判生成的优化前后覆盖率。"""

    comparison_module = import_module(
        "evaluation.optimization_comparison"
    )
    deterministic_score = {
        "answer_types": {"accuracy": 1.0},
        "answer_points": {"coverage": 0.6},
        "citations": {"source_validity": 1.0},
        "directory": {"accuracy": 1.0},
        "refusal": {"accuracy": 1.0},
        "tools": {"total_calls": 1, "error_calls": 0},
        "performance": {
            "tokens": {"total": 100},
            "latency_ms": {"average": 100.0},
        },
    }
    before_score_path = tmp_path / "before-score.json"
    after_score_path = tmp_path / "after-score.json"
    before_semantic_path = tmp_path / "before-semantic-score.json"
    after_semantic_path = tmp_path / "after-semantic-score.json"
    before_score_path.write_text(
        json.dumps(deterministic_score),
        encoding="utf-8",
    )
    after_score_path.write_text(
        json.dumps(deterministic_score),
        encoding="utf-8",
    )
    before_semantic_path.write_text(
        json.dumps({"answer_points": {"coverage": 1.0}}),
        encoding="utf-8",
    )
    after_semantic_path.write_text(
        json.dumps({"answer_points": {"coverage": 1.0}}),
        encoding="utf-8",
    )

    result = comparison_module.compare_agentic_optimization_files(
        before_score_path,
        after_score_path,
        before_semantic_path,
        after_semantic_path,
    )

    assert result["quality"]["semantic_answer_point_coverage"] == {
        "before": 1.0,
        "after": 1.0,
        "delta": 0.0,
    }


def test_render_agentic_optimization_markdown_reports_quality_and_cost() -> None:
    """Markdown 报告应展示质量、成本变化及延迟解释边界。"""

    comparison_module = import_module(
        "evaluation.optimization_comparison"
    )
    comparison = {
        "quality": {
            "answer_type_accuracy": {
                "before": 1.0,
                "after": 1.0,
                "delta": 0.0,
            },
            "answer_point_coverage": {
                "before": 0.56,
                "after": 0.64,
                "delta": 0.08,
            },
            "citation_source_validity": {
                "before": 1.0,
                "after": 1.0,
                "delta": 0.0,
            },
            "directory_accuracy": {
                "before": 1.0,
                "after": 1.0,
                "delta": 0.0,
            },
            "refusal_accuracy": {
                "before": 1.0,
                "after": 1.0,
                "delta": 0.0,
            },
            "semantic_answer_point_coverage": {
                "before": 1.0,
                "after": 1.0,
                "delta": 0.0,
            },
        },
        "efficiency": {
            "total_tool_calls": {
                "before": 28.0,
                "after": 22.0,
                "delta": -6.0,
            },
            "error_tool_calls": {
                "before": 2.0,
                "after": 1.0,
                "delta": -1.0,
            },
            "total_tokens": {
                "before": 112_731.0,
                "after": 83_180.0,
                "delta": -29_551.0,
            },
            "average_latency_ms": {
                "before": 3_621.9,
                "after": 7_509.0,
                "delta": 3_887.1,
            },
        },
    }

    markdown = comparison_module.render_agentic_optimization_markdown(
        comparison
    )

    assert "# Agentic RAG 阶段 9 优化前后对比报告" in markdown
    assert "回答类型准确率" in markdown
    assert "答案点覆盖率" in markdown
    assert "56.0%" in markdown
    assert "64.0%" in markdown
    assert "+8.0 个百分点" in markdown
    assert "独立语义答案点覆盖率" in markdown
    assert "100.0%" in markdown
    assert "工具调用总数" in markdown
    assert "28" in markdown
    assert "22" in markdown
    assert "总 Token" in markdown
    assert "112731" in markdown
    assert "83180" in markdown
    assert "总 Token 下降 26.2%" in markdown
    assert "平均延迟" in markdown
    assert "3621.9 ms" in markdown
    assert "7509.0 ms" in markdown
    assert "延迟仅记录观测值，不作为本轮优化的硬性通过条件" in markdown
    assert "基于检索分数的三态判断" in markdown
    assert "`relevant`：继续调用 `read` 获取原文证据" in markdown
    assert "`uncertain`：允许 Agent 继续探索" in markdown
    assert "`none`：中间件立即停止，并返回证据不足" in markdown


def test_write_agentic_optimization_report_saves_markdown(
    tmp_path: Path,
) -> None:
    """报告入口应比较两份评分文件并保存 Markdown。"""

    comparison_module = import_module(
        "evaluation.optimization_comparison"
    )

    def build_score(tool_calls: int, total_tokens: int) -> dict[str, object]:
        return {
            "answer_types": {"accuracy": 1.0},
            "answer_points": {"coverage": 0.6},
            "citations": {"source_validity": 1.0},
            "directory": {"accuracy": 1.0},
            "refusal": {"accuracy": 1.0},
            "tools": {
                "total_calls": tool_calls,
                "error_calls": 0,
            },
            "performance": {
                "tokens": {"total": total_tokens},
                "latency_ms": {"average": 1_000.0},
            },
        }

    before_path = tmp_path / "before-score.json"
    after_path = tmp_path / "after-score.json"
    output_path = tmp_path / "reports" / "stage-9-comparison.md"
    before_path.write_text(
        json.dumps(build_score(28, 112_731)),
        encoding="utf-8",
    )
    after_path.write_text(
        json.dumps(build_score(22, 83_180)),
        encoding="utf-8",
    )

    result = comparison_module.write_agentic_optimization_report(
        before_path,
        after_path,
        output_path,
    )

    assert result == output_path
    assert output_path.is_file()
    markdown = output_path.read_text(encoding="utf-8")
    assert "# Agentic RAG 阶段 9 优化前后对比报告" in markdown
    assert "总 Token 下降 26.2%" in markdown


def test_main_writes_current_stage9_optimization_report(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    """项目入口应使用阶段 9 的固定评分文件并打印报告路径。"""

    comparison_module = import_module(
        "evaluation.optimization_comparison"
    )
    calls: list[tuple[Path, Path, Path, Path, Path]] = []

    def fake_write_report(
        before_score_path: Path,
        after_score_path: Path,
        output_path: Path,
        before_semantic_score_path: Path,
        after_semantic_score_path: Path,
    ) -> Path:
        calls.append(
            (
                before_score_path,
                after_score_path,
                output_path,
                before_semantic_score_path,
                after_semantic_score_path,
            )
        )
        return output_path

    monkeypatch.setattr(
        comparison_module,
        "write_agentic_optimization_report",
        fake_write_report,
    )

    comparison_module.main(project_root=tmp_path)

    results_directory = tmp_path / "evaluation" / "results"
    expected_output_path = results_directory / (
        "agentic-optimization-comparison-qwen3.6-flash-"
        "thinking-off-prompt-v1.6.md"
    )
    assert calls == [
        (
            results_directory
            / (
                "agentic-baseline-qwen3.6-flash-thinking-off-"
                "prompt-v1.6.pre-retrieval-stop-score.json"
            ),
            results_directory
            / (
                "agentic-baseline-qwen3.6-flash-thinking-off-"
                "prompt-v1.6-score.json"
            ),
            expected_output_path,
            results_directory
            / (
                "agentic-baseline-qwen3.6-flash-thinking-off-"
                "prompt-v1.6.pre-retrieval-stop-semantic-score.json"
            ),
            results_directory
            / (
                "agentic-baseline-qwen3.6-flash-thinking-off-"
                "prompt-v1.6-semantic-score.json"
            ),
        )
    ]
    assert capsys.readouterr().out == (
        f"Agentic optimization report saved to: {expected_output_path}\n"
    )
