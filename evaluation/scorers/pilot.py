"""Pilot 检索证据覆盖与最终答案覆盖评分。

- 检索证据覆盖：只依赖规范文件与检索结果 JSON，不调用模型，离线可复现。
- 最终答案覆盖：复用语义 Judge（evaluation/scorers/semantic.py）逐点判定
  答案是否表达了 expectation 的完整含义。
"""

from evaluation.scorers.semantic import (
    SemanticJudge,
    score_semantic_answer_points,
)


def _valid_hits(
    retrieval_result: dict[str, object],
    source_path: str,
) -> list[dict[str, object]]:
    """筛选出属于规范声明文档的命中 Chunk。"""

    hits = retrieval_result.get("hits")
    if not isinstance(hits, list):
        raise ValueError("pilot retrieval hits must be a list")

    valid_hits: list[dict[str, object]] = []
    for hit in hits:
        if not isinstance(hit, dict):
            raise ValueError("each pilot hit must be an object")
        if hit.get("path") == source_path:
            valid_hits.append(hit)
    return valid_hits


def _merge_ranges(ranges: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """合并行号闭区间；相邻区间（无缺失行）也合并为一段。"""

    if not ranges:
        return []

    sorted_ranges = sorted(ranges, key=lambda item: item[0])
    merged: list[tuple[int, int]] = []
    for start, end in sorted_ranges:
        if not merged or start > merged[-1][1] + 1:
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    return merged


def _covers_range(
    merged: list[tuple[int, int]],
    start_line: int,
    end_line: int,
) -> bool:
    """金证据区间内的每一行都被合并区间覆盖时返回 True。"""

    for line in range(start_line, end_line + 1):
        if not any(chunk_start <= line <= chunk_end for chunk_start, chunk_end in merged):
            return False
    return True


def _intersects(
    hit_start: int,
    hit_end: int,
    evidence_start: int,
    evidence_end: int,
) -> bool:
    """命中 Chunk 与金证据区间是否有共同行。"""

    return hit_start <= evidence_end and hit_end >= evidence_start


def score_pilot_retrieval_coverage(
    spec: dict[str, object],
    retrieval_result: dict[str, object],
) -> dict[str, object]:
    """按金证据答案点计算 A 组检索结果的覆盖评分。

    1. 读取规范中全部 answer_points 的金证据行区间。
    2. 只接受规范 corpus.source_path 声明的文档命中。
    3. 合并所有命中 Chunk 的行区间。
    4. 金证据区间每一行都被覆盖时，该答案点才算 covered。
    5. 输出每个答案点明细与汇总指标。
    """

    corpus = spec.get("corpus")
    points = spec.get("answer_points")
    if not isinstance(corpus, dict):
        raise ValueError("pilot spec corpus must be an object")
    if not isinstance(points, list):
        raise ValueError("pilot spec answer_points must be a list")

    source_path = corpus.get("source_path")
    if not isinstance(source_path, str) or not source_path.strip():
        raise ValueError("pilot corpus source_path must be a non-empty string")

    valid_hits = _valid_hits(retrieval_result, source_path)
    hit_ranges = [
        (hit["start_line"], hit["end_line"])
        for hit in valid_hits
    ]
    merged_ranges = _merge_ranges(hit_ranges)

    scored_points: list[dict[str, object]] = []
    for point in points:
        if not isinstance(point, dict):
            raise ValueError("each answer point must be an object")
        evidence = point.get("evidence")
        if not isinstance(evidence, dict):
            raise ValueError("each answer point must declare an evidence object")

        evidence_start = evidence["start_line"]
        evidence_end = evidence["end_line"]
        covered = _covers_range(merged_ranges, evidence_start, evidence_end)

        matched_ranks = [
            hit["rank"]
            for hit in valid_hits
            if _intersects(hit["start_line"], hit["end_line"], evidence_start, evidence_end)
        ]

        scored_points.append(
            {
                "id": point["id"],
                "service": point.get("service"),
                "field": point.get("field"),
                "evidence": {
                    "start_line": evidence_start,
                    "end_line": evidence_end,
                },
                "covered": covered,
                "matched_ranks": matched_ranks,
            }
        )

    covered_count = sum(1 for point in scored_points if point["covered"])
    total_count = len(scored_points)
    if total_count == 0:
        coverage = 0.0
    else:
        coverage = covered_count / total_count

    return {
        "answer_points": scored_points,
        "covered_answer_point_count": covered_count,
        "total_answer_point_count": total_count,
        "retrieved_gold_region_coverage": coverage,
    }


def score_pilot_answer_coverage(
    spec: dict[str, object],
    answer_result: dict[str, object],
    judge: SemanticJudge,
) -> dict[str, object]:
    """复用语义 Judge 对最终答案做答案点覆盖评分。

    1. 把规范的 expectation 文本按顺序作为答案点，交给现有
       score_semantic_answer_points 逐点评判（它校验每个答案点
       恰好返回一个判断）。
    2. 把 judge 的 verdict 映射回答案点 id，covered 仅当 matched。
    3. 汇总 answer_point_coverage，与检索证据覆盖结果相互印证。
    """

    points = spec.get("answer_points")
    question = answer_result.get("question")
    answer = answer_result.get("answer")
    if not isinstance(points, list):
        raise ValueError("pilot spec answer_points must be a list")
    if not isinstance(question, str) or not isinstance(answer, str):
        raise ValueError(
            "pilot answer result must contain question and answer"
        )

    expectations = [
        point["expectation"]
        for point in points
        if isinstance(point, dict)
    ]
    if len(expectations) != len(points):
        raise ValueError("each pilot answer point must declare an expectation")

    score = score_semantic_answer_points(
        question=question,
        answer_points=expectations,
        answer=answer,
        judge=judge,
    )

    scored_points: list[dict[str, object]] = []
    for raw_judgment in score["judgments"]:
        point_index = int(raw_judgment["point_index"])
        point = points[point_index]
        verdict = raw_judgment["verdict"]
        scored_points.append(
            {
                "id": point["id"],
                "service": point.get("service"),
                "field": point.get("field"),
                "verdict": verdict,
                "reason": raw_judgment.get("reason"),
                "covered": verdict == "matched",
            }
        )

    return {
        "pilot_id": spec.get("id"),
        "arm_id": answer_result.get("arm_id"),
        "question": question,
        "answer_points": scored_points,
        "covered_answer_point_count": int(score["matched"]),
        "total_answer_point_count": int(score["total"]),
        "answer_point_coverage": score.get("coverage"),
    }
