"""A 组金证据覆盖评分（score_pilot_retrieval_coverage）离线测试。

评分规则：
1. 读取规范中每个 answer_points 的金证据行区间。
2. 只接受规范 corpus.source_path 声明的文件。
3. 合并所有命中 Chunk 的行区间。
4. 金证据区间的每一行都被覆盖才算 covered（部分覆盖不算）。
5. 输出每个答案点明细与汇总指标。
"""

import importlib
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SPEC_PATH = (
    PROJECT_ROOT
    / "evaluation"
    / "pilots"
    / "multi_chunk_guide_001.json"
)
ARM_A_RESULT_PATH = (
    PROJECT_ROOT
    / "evaluation"
    / "results"
    / "pilot-multi-chunk-guide-001-arm-a-retrieval.json"
)


def _load_json(path: Path) -> dict[str, object]:
    """从项目文件读取 JSON 对象。"""

    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return data


def test_score_pilot_retrieval_coverage_on_real_arm_a_result() -> None:
    """真实 A 组结果应得到 3/12 答案点覆盖，覆盖率 0.25。"""

    pilot_scorer = importlib.import_module("evaluation.scorers.pilot")
    spec = _load_json(SPEC_PATH)
    retrieval_result = _load_json(ARM_A_RESULT_PATH)

    result = pilot_scorer.score_pilot_retrieval_coverage(
        spec,
        retrieval_result,
    )

    assert result["covered_answer_point_count"] == 3
    assert result["total_answer_point_count"] == 12
    assert result["retrieved_gold_region_coverage"] == 0.25

    covered_by_id = {
        point["id"]: point["covered"]
        for point in result["answer_points"]
    }
    assert covered_by_id["priority-review-condition"] is True
    assert covered_by_id["priority-review-materials"] is True
    assert covered_by_id["priority-review-process"] is True
    assert covered_by_id["priority-review-result"] is False
    assert covered_by_id["rapid-pre-review-condition"] is False
    assert covered_by_id["rapid-pre-review-materials"] is False
    assert covered_by_id["rapid-pre-review-process"] is False
    assert covered_by_id["rapid-pre-review-result"] is False
    assert covered_by_id["rights-assistance-condition"] is False
    assert covered_by_id["rights-assistance-materials"] is False
    assert covered_by_id["rights-assistance-process"] is False
    assert covered_by_id["rights-assistance-result"] is False


def test_score_pilot_retrieval_coverage_requires_full_row_coverage() -> None:
    """金证据区间只被部分覆盖时，该答案点不应标记为 covered。"""

    pilot_scorer = importlib.import_module("evaluation.scorers.pilot")
    spec = {
        "corpus": {"source_path": "/办事指南.md"},
        "answer_points": [
            {
                "id": "demo-condition",
                "service": "演示业务",
                "field": "受理条件",
                "evidence": {"start_line": 10, "end_line": 20},
            }
        ],
    }
    retrieval_result = {
        "hits": [
            {
                "rank": 1,
                "path": "/办事指南.md",
                "start_line": 1,
                "end_line": 15,
            }
        ]
    }

    result = pilot_scorer.score_pilot_retrieval_coverage(
        spec,
        retrieval_result,
    )

    assert result["answer_points"][0]["covered"] is False
    assert result["covered_answer_point_count"] == 0
    assert result["retrieved_gold_region_coverage"] == 0.0


def test_score_pilot_retrieval_coverage_merges_overlapping_hit_ranges() -> None:
    """多个命中 Chunk 合并后完整覆盖金证据区间时，应标记为 covered。"""

    pilot_scorer = importlib.import_module("evaluation.scorers.pilot")
    spec = {
        "corpus": {"source_path": "/办事指南.md"},
        "answer_points": [
            {
                "id": "demo-materials",
                "service": "演示业务",
                "field": "申请材料",
                "evidence": {"start_line": 10, "end_line": 20},
            }
        ],
    }
    retrieval_result = {
        "hits": [
            {
                "rank": 1,
                "path": "/办事指南.md",
                "start_line": 1,
                "end_line": 12,
            },
            {
                "rank": 2,
                "path": "/办事指南.md",
                "start_line": 13,
                "end_line": 25,
            },
        ]
    }

    result = pilot_scorer.score_pilot_retrieval_coverage(
        spec,
        retrieval_result,
    )

    assert result["answer_points"][0]["covered"] is True
    assert result["answer_points"][0]["matched_ranks"] == [1, 2]


def test_score_pilot_retrieval_coverage_ignores_other_source_paths() -> None:
    """非规范声明的 source path 命中不应参与覆盖计算。"""

    pilot_scorer = importlib.import_module("evaluation.scorers.pilot")
    spec = {
        "corpus": {"source_path": "/办事指南.md"},
        "answer_points": [
            {
                "id": "demo-result",
                "service": "演示业务",
                "field": "办理结果",
                "evidence": {"start_line": 30, "end_line": 40},
            }
        ],
    }
    retrieval_result = {
        "hits": [
            {
                "rank": 1,
                "path": "/其他文档.md",
                "start_line": 1,
                "end_line": 100,
            }
        ]
    }

    result = pilot_scorer.score_pilot_retrieval_coverage(
        spec,
        retrieval_result,
    )

    assert result["answer_points"][0]["covered"] is False
    assert result["answer_points"][0]["matched_ranks"] == []
