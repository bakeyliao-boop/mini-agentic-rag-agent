"""长文档 Pilot Agent 运行时接线测试。"""

import importlib
import json
from pathlib import Path


def test_build_pilot_agentic_runtime_uses_matching_corpus_and_index(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """search 索引、知识工具和路径快照应属于同一份 Pilot 语料。"""

    pilot_cli = importlib.import_module("evaluation.cli.pilot_agentic")
    spec_path = (
        tmp_path
        / "evaluation"
        / "pilots"
        / "multi_chunk_guide_001.json"
    )
    spec_path.parent.mkdir(parents=True)
    spec_path.write_text(
        json.dumps(
            {
                "id": "multi-chunk-guide-001",
                "corpus": {
                    "root": (
                        "evaluation/fixtures/"
                        "multi_chunk_guide_001/corpus"
                    ),
                    "isolation": "single_document",
                    "source_path": "/办事指南.md",
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    corpus_root = (
        tmp_path
        / "evaluation"
        / "fixtures"
        / "multi_chunk_guide_001"
        / "corpus"
    )
    corpus_root.mkdir(parents=True)

    settings = {
        "DASHSCOPE_API_KEY": "test-key",
        "DASHSCOPE_BASE_URL": "https://example.test/v1",
        "EMBEDDING_MODEL": "test-embedding",
        "EMBEDDING_DIMENSIONS": "1024",
        "EMBEDDING_BATCH_SIZE": "10",
    }
    fake_embedding = object()
    class FakeVectorStore:
        def get(self, **options: object) -> dict[str, object]:
            events.append(("index_check", options))
            return {"ids": ["pilot-chunk-1"]}

    fake_vector_store = FakeVectorStore()
    fake_chat_model = object()
    fake_snapshot = [
        {"path": "/", "type": "directory"},
        {"path": "/办事指南.md", "type": "file"},
    ]
    events: list[tuple[object, ...]] = []

    expected_index_directory = (
        tmp_path / "data" / "pilots" / "multi-chunk-guide-001"
    ).resolve(strict=False)
    expected_index_directory.mkdir(parents=True)

    def fake_build_embeddings(**options: object) -> object:
        events.append(("embedding", options))
        return fake_embedding

    def fake_chroma(**options: object) -> object:
        events.append(("open_index", options))
        return fake_vector_store

    def fake_build_chat_model(
        config: object,
        api_key: str,
        base_url: str,
    ) -> object:
        events.append(("chat_model", config, api_key, base_url))
        return fake_chat_model

    def fake_build_snapshot(root: Path) -> list[dict[str, str]]:
        events.append(("snapshot", root))
        return fake_snapshot

    def reject_index_rebuild(*args: object, **kwargs: object) -> object:
        raise AssertionError("Pilot Agent 不应在运行时重新构建索引")

    monkeypatch.setattr(
        pilot_cli,
        "build_dashscope_embeddings",
        fake_build_embeddings,
    )
    monkeypatch.setattr(pilot_cli, "Chroma", fake_chroma)
    monkeypatch.setattr(
        pilot_cli,
        "build_traditional_chat_model",
        fake_build_chat_model,
    )
    monkeypatch.setattr(
        pilot_cli,
        "build_knowledge_path_snapshot",
        fake_build_snapshot,
        raising=False,
    )
    monkeypatch.setattr(
        pilot_cli,
        "build_knowledge_index",
        reject_index_rebuild,
        raising=False,
    )

    runtime = pilot_cli.build_pilot_agentic_runtime_from_project(
        project_root=tmp_path,
        settings=settings,
    )

    expected_corpus_root = corpus_root.resolve(strict=False)
    assert runtime.knowledge_root == expected_corpus_root
    assert runtime.vector_store is fake_vector_store
    assert runtime.chat_model is fake_chat_model
    assert runtime.path_snapshot is fake_snapshot
    assert events[1] == (
        "open_index",
        {
            "collection_name": "knowledge_chunks",
            "persist_directory": str(expected_index_directory),
            "embedding_function": fake_embedding,
        },
    )
    assert ("index_check", {"limit": 1, "include": []}) in events
    assert events[-1] == ("snapshot", expected_corpus_root)


def test_build_pilot_agentic_runtime_rejects_missing_index(
    tmp_path: Path,
) -> None:
    """未提前构建 Pilot 索引时应明确失败，不能静默创建空库。"""

    import pytest

    pilot_cli = importlib.import_module("evaluation.cli.pilot_agentic")
    spec_path = (
        tmp_path
        / "evaluation"
        / "pilots"
        / "multi_chunk_guide_001.json"
    )
    spec_path.parent.mkdir(parents=True)
    spec_path.write_text(
        json.dumps(
            {
                "id": "multi-chunk-guide-001",
                "corpus": {
                    "root": "evaluation/fixtures/pilot/corpus",
                    "isolation": "single_document",
                    "source_path": "/办事指南.md",
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (tmp_path / "evaluation" / "fixtures" / "pilot" / "corpus").mkdir(
        parents=True
    )

    with pytest.raises(FileNotFoundError, match="pilot index does not exist"):
        pilot_cli.build_pilot_agentic_runtime_from_project(
            project_root=tmp_path,
            settings={},
        )
