"""多 Chunk Pilot A/B/C 运行流程测试。"""

import importlib
import json
from pathlib import Path


def test_run_pilot_arm_a_executes_single_top5_and_records_chunks(
    monkeypatch,
) -> None:
    """A 组应只检索一次 Top-5，并完整记录命中的 Chunk。"""

    pilot_runner = importlib.import_module("evaluation.runners.pilot")
    fake_vector_store = object()
    spec = {
        "id": "multi-chunk-guide-001",
        "question": "请整理三项业务的办理清单。",
        "comparison_arms": [
            {
                "id": "A",
                "name": "traditional-single-top5",
                "retrieval": {
                    "queries": ["use_full_question"],
                    "top_k_per_query": 5,
                    "maximum_retrieval_calls": 1,
                },
            }
        ],
    }
    fake_hits = [
        {
            "path": "/办事指南.md",
            "start_line": 212,
            "end_line": 220,
            "score": 0.81,
            "preview": "优先审查的受理条件。",
        },
        {
            "path": "/办事指南.md",
            "start_line": 3633,
            "end_line": 3671,
            "score": 0.72,
            "preview": "维权援助的申请材料和办理步骤。",
        },
    ]
    search_calls: list[dict[str, object]] = []

    def fake_search_chroma_index(**kwargs: object) -> dict[str, object]:
        search_calls.append(kwargs)
        return {
            "hits": fake_hits,
            "usage": "candidate_only",
            "retrieval_status": "relevant",
        }

    monkeypatch.setattr(
        pilot_runner,
        "search_chroma_index",
        fake_search_chroma_index,
    )

    result = pilot_runner.run_pilot_arm_a(spec, fake_vector_store)

    assert search_calls == [
        {
            "vector_store": fake_vector_store,
            "query": "请整理三项业务的办理清单。",
            "path": "/",
            "limit": 5,
        }
    ]
    assert result == {
        "pilot_id": "multi-chunk-guide-001",
        "arm_id": "A",
        "arm_name": "traditional-single-top5",
        "question": "请整理三项业务的办理清单。",
        "retrieval_call_count": 1,
        "retrieved_chunk_count": 2,
        "retrieval_status": "relevant",
        "hits": [
            {
                "rank": 1,
                "path": "/办事指南.md",
                "start_line": 212,
                "end_line": 220,
                "score": 0.81,
                "content": "优先审查的受理条件。",
            },
            {
                "rank": 2,
                "path": "/办事指南.md",
                "start_line": 3633,
                "end_line": 3671,
                "score": 0.72,
                "content": "维权援助的申请材料和办理步骤。",
            },
        ],
    }


def test_run_pilot_arm_a_from_project_opens_index_and_saves_result(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """A 组项目入口应打开独立索引、运行检索并保存结果。"""

    pilot_cli = importlib.import_module("evaluation.cli.pilot")
    spec_path = (
        tmp_path
        / "evaluation"
        / "pilots"
        / "multi_chunk_guide_001.json"
    )
    spec_path.parent.mkdir(parents=True)
    spec = {
        "id": "multi-chunk-guide-001",
        "question": "请整理三项业务的办理清单。",
        "comparison_arms": [],
    }
    spec_path.write_text(
        json.dumps(spec, ensure_ascii=False),
        encoding="utf-8",
    )
    settings = {
        "DASHSCOPE_API_KEY": "test-key",
        "DASHSCOPE_BASE_URL": (
            "https://dashscope.aliyuncs.com/compatible-mode/v1"
        ),
        "EMBEDDING_MODEL": "text-embedding-v4",
        "EMBEDDING_DIMENSIONS": "1024",
        "EMBEDDING_BATCH_SIZE": "10",
        "PILOT_CHROMA_PERSIST_DIR": "data/pilots/multi-chunk-guide-001",
    }
    fake_embedding = object()
    fake_vector_store = object()
    fake_result = {"pilot_id": "multi-chunk-guide-001", "arm_id": "A"}
    calls: list[tuple[object, ...]] = []

    def fake_build_embeddings(**kwargs: object) -> object:
        calls.append(("embedding", kwargs))
        return fake_embedding

    def fake_chroma(**kwargs: object) -> object:
        calls.append(("open_index", kwargs))
        return fake_vector_store

    def fake_run_arm_a(
        loaded_spec: dict[str, object],
        vector_store: object,
    ) -> dict[str, object]:
        calls.append(("run", loaded_spec, vector_store))
        return fake_result

    def fake_save_result(
        result: dict[str, object],
        output_path: Path,
    ) -> None:
        calls.append(("save", result, output_path))

    monkeypatch.setattr(
        pilot_cli,
        "build_dashscope_embeddings",
        fake_build_embeddings,
    )
    monkeypatch.setattr(pilot_cli, "Chroma", fake_chroma)
    monkeypatch.setattr(pilot_cli, "run_pilot_arm_a", fake_run_arm_a)
    monkeypatch.setattr(
        pilot_cli,
        "save_evaluation_result",
        fake_save_result,
    )

    output_path = pilot_cli.run_pilot_arm_a_from_project(
        project_root=tmp_path,
        settings=settings,
    )

    expected_persist_directory = (
        tmp_path / "data" / "pilots" / "multi-chunk-guide-001"
    ).resolve(strict=False)
    expected_output_path = (
        tmp_path
        / "evaluation"
        / "results"
        / "pilot-multi-chunk-guide-001-arm-a-retrieval.json"
    )
    assert output_path == expected_output_path
    assert calls == [
        (
            "embedding",
            {
                "model": "text-embedding-v4",
                "dimensions": 1024,
                "batch_size": 10,
                "api_key": "test-key",
                "base_url": (
                    "https://dashscope.aliyuncs.com/compatible-mode/v1"
                ),
            },
        ),
        (
            "open_index",
            {
                "collection_name": "knowledge_chunks",
                "persist_directory": str(expected_persist_directory),
                "embedding_function": fake_embedding,
            },
        ),
        ("run", spec, fake_vector_store),
        ("save", fake_result, expected_output_path),
    ]
