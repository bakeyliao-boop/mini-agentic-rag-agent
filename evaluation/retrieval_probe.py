"""离线探针：重跑 V1.6 基线中每题全部 search 的查询，打印 Top-K 候选分数。

目的：比较超范围题与正常知识题每次检索的候选分数分布。
看 out-of-scope-001 的 4 次 search 是否都处于低分区间，
从而判断“低分信号”是否稳定，为后续相关性信号设计提供数据。

score 含义：本项目 Chroma 索引使用 cosine 距离，
langchain_core 将距离换算为 score = 1 - distance。
score 越大越相关，1.0 表示与查询完全同向，0.0 表示正交。
"""

import json
from pathlib import Path

from rag_core.agentic.runner import build_agentic_runtime_from_project
from rag_core.knowledge.indexer import search_chroma_index
from rag_core.settings import load_settings_from_env

DEFAULT_RESULT_PATH = (
    Path(__file__).resolve().parent
    / "results"
    / "agentic-baseline-qwen3.6-flash-thinking-off-prompt-v1.6.json"
)


def load_all_search_queries(result_path: Path) -> list[dict[str, object]]:
    """从基线结果中提取每题全部 search 调用的真实查询词。"""

    with result_path.open(encoding="utf-8") as handle:
        result = json.load(handle)

    entries: list[dict[str, object]] = []
    for item in result["results"]:
        search_traces = [
            trace
            for trace in item["tool_traces"]
            if trace["name"] == "search"
        ]
        if not search_traces:
            continue
        entries.append(
            {
                "question_id": item["id"],
                "category": item["category"],
                "question": item["question"],
                "searches": [
                    {
                        # step 是该调用在全部工具轨迹中的序号，
                        # search_index 是它在本题内第几次 search。
                        "step": trace["step"],
                        "search_index": index,
                        "query": trace["args"]["query"],
                        "path": trace["args"].get("path", "/"),
                    }
                    for index, trace in enumerate(search_traces, start=1)
                ],
            }
        )
    return entries


def _estimate_tokens(text: str) -> int:
    """启发式 token 估算：ASCII 约 4 字符/token，中文约 1.5 字符/token。"""

    ascii_chars = sum(1 for char in text if ord(char) < 128)
    cjk_chars = len(text) - ascii_chars
    return ascii_chars // 4 + int(cjk_chars / 1.5)


def _print_payload_stats(hits: list[dict[str, object]]) -> None:
    """打印 search 返回 payload 的字段成本统计，用于 9.4 瘦身决策。"""

    preview_chars = sum(len(hit["preview"]) for hit in hits)
    preview_tokens = sum(_estimate_tokens(hit["preview"]) for hit in hits)
    location_chars = sum(
        len(hit["path"]) + len(f"L{hit['start_line']}-L{hit['end_line']}")
        for hit in hits
    )
    print(
        f"  [payload] preview: {preview_chars} 字符 ≈ {preview_tokens} token; "
        f"path+行号: ~{location_chars} 字符; hits={len(hits)}"
    )


def probe_retrieval_scores(
    project_root: Path,
    settings: dict[str, str],
    result_path: Path,
) -> None:
    """对每题每次 search 查询重跑检索，并按题目打印 Top-K 候选。"""

    runtime = build_agentic_runtime_from_project(
        project_root=project_root,
        settings=settings,
    )
    entries = load_all_search_queries(result_path)

    for entry in entries:
        print(f"=== {entry['question_id']} [{entry['category']}] ===")
        print(f"question: {entry['question']}")
        for search in entry["searches"]:
            print(
                f"--- search #{search['search_index']} "
                f"(step {search['step']}) ---"
            )
            print(f"query: {search['query']}")
            hits = search_chroma_index(
                vector_store=runtime.vector_store,
                query=search["query"],
                path=search["path"],
                limit=5,
            )["hits"]
            if not hits:
                print("  (no hits)")
            for index, hit in enumerate(hits, start=1):
                print(
                    f"  #{index} score={hit['score']:.4f} "
                    f"{hit['path']} L{hit['start_line']}-L{hit['end_line']}"
                )
            _print_payload_stats(hits)
        print()


def main(project_root: Path | None = None) -> None:
    """加载项目配置并运行检索分数探针。"""

    resolved_project_root = (
        project_root
        if project_root is not None
        else Path(__file__).resolve().parent.parent
    )
    settings = load_settings_from_env(resolved_project_root)
    probe_retrieval_scores(
        project_root=resolved_project_root,
        settings=settings,
        result_path=DEFAULT_RESULT_PATH,
    )


if __name__ == "__main__":
    main()
