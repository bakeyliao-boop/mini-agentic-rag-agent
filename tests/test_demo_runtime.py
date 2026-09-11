"""水循环 Demo 语料加载与运行时接线测试。"""

import importlib
import json
from pathlib import Path

import pytest


DEMO_SPEC_RELATIVE_PATH = Path(
    "evaluation/demos/teacher_water_cycle_consistency_001.json"
)
DEMO_CORPUS_ROOT = Path(
    "evaluation/fixtures/teacher_water_cycle_consistency_001/corpus"
)
DEMO_PERSIST_DIRECTORY = Path(
    "data/pilots/teacher-water-cycle-consistency-001"
)
DEMO_SOURCE_PATHS = [
    "/教师备课/科学/水循环/教师指南.md",
    "/教师备课/科学/水循环/教师演示说明.md",
    "/教师备课/科学/水循环/学生学习单.md",
    "/教师备课/科学/水循环/前后测.md",
]


def _write_declared_demo_spec(
    project_root: Path,
    source_paths: list[str] | None = None,
) -> Path:
    """写入只加载明确声明文档的水循环 Demo 规格。"""

    spec_path = project_root / DEMO_SPEC_RELATIVE_PATH
    spec_path.parent.mkdir(parents=True, exist_ok=True)
    spec_path.write_text(
        json.dumps(
            {
                "version": 1,
                "id": "teacher-water-cycle-consistency-001",
                "chat_model": "qwen3.7-flash",
                "retrieval_policy": "locate_first",
                "question": "请核对水循环演示和学生学习单。",
                "corpus": {
                    "root": DEMO_CORPUS_ROOT.as_posix(),
                    "isolation": "declared_documents",
                    "source_paths": (
                        DEMO_SOURCE_PATHS
                        if source_paths is None
                        else source_paths
                    ),
                },
                "persist_directory": DEMO_PERSIST_DIRECTORY.as_posix(),
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return spec_path


def test_default_demo_declared_documents_load_four_paths_and_17_chunks(
) -> None:
    """冻结 Demo 应稳定加载四份资料，并保持已实测的 17 个 Chunk。"""

    loader = importlib.import_module("evaluation.pilot_loader")
    project_root = Path(__file__).resolve().parent.parent
    spec_path = project_root / DEMO_SPEC_RELATIVE_PATH
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    declared_paths = spec["corpus"]["source_paths"]

    chunks = loader.load_pilot_chunks(project_root, spec_path)

    assert len(chunks) == 17
    assert list(dict.fromkeys(chunk.path for chunk in chunks)) == (
        declared_paths
    )
    assert sorted(declared_paths) == sorted(DEMO_SOURCE_PATHS)


def test_declared_documents_reject_duplicate_source_path(
    tmp_path: Path,
) -> None:
    """同一文档不能重复声明，否则会重复建库并扭曲检索排序。"""

    loader = importlib.import_module("evaluation.pilot_loader")
    duplicate = [DEMO_SOURCE_PATHS[0], DEMO_SOURCE_PATHS[0]]
    spec_path = _write_declared_demo_spec(tmp_path, duplicate)

    with pytest.raises(ValueError, match="duplicate"):
        loader.load_pilot_chunks(tmp_path, spec_path)


def test_declared_documents_reject_path_outside_corpus(
    tmp_path: Path,
) -> None:
    """声明路径不能通过父目录穿越到冻结语料之外。"""

    loader = importlib.import_module("evaluation.pilot_loader")
    spec_path = _write_declared_demo_spec(
        tmp_path,
        ["/../不属于语料.md"],
    )

    with pytest.raises(ValueError, match="must not contain.*segments"):
        loader.load_pilot_chunks(tmp_path, spec_path)


def test_declared_documents_reject_missing_source_file(
    tmp_path: Path,
) -> None:
    """任一声明文档缺失时应停止，不能静默少建一部分索引。"""

    loader = importlib.import_module("evaluation.pilot_loader")
    spec_path = _write_declared_demo_spec(
        tmp_path,
        ["/教师备课/科学/水循环/缺失.md"],
    )

    with pytest.raises(FileNotFoundError):
        loader.load_pilot_chunks(tmp_path, spec_path)


def test_demo_runtime_uses_qwen37_independent_index_without_rebuild(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Demo 应打开独立索引并显式使用 qwen3.7，不在启动时重建。"""

    runtime_module = importlib.import_module("evaluation.cli.pilot_agentic")
    spec_path = _write_declared_demo_spec(tmp_path)
    corpus_root = tmp_path / DEMO_CORPUS_ROOT
    corpus_root.mkdir(parents=True)
    persist_directory = (tmp_path / DEMO_PERSIST_DIRECTORY).resolve(
        strict=False
    )
    persist_directory.mkdir(parents=True)
    settings = {
        "DASHSCOPE_API_KEY": "test-key",
        "DASHSCOPE_BASE_URL": "https://example.test/v1",
        "EMBEDDING_MODEL": "test-embedding",
        "EMBEDDING_DIMENSIONS": "1024",
        "EMBEDDING_BATCH_SIZE": "10",
    }
    events: list[tuple[object, ...]] = []
    fake_embedding = object()
    fake_model = object()
    fake_snapshot = [{"path": "/", "type": "directory"}]

    class FakeVectorStore:
        def get(self, **options: object) -> dict[str, object]:
            events.append(("index_check", options))
            return {"ids": ["water-cycle-chunk-1"]}

    fake_vector_store = FakeVectorStore()

    def fake_embeddings(**options: object) -> object:
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
        return fake_model

    def fake_path_snapshot(root: Path) -> list[dict[str, str]]:
        events.append(("snapshot", root))
        return fake_snapshot

    def reject_index_rebuild(*args: object, **kwargs: object) -> object:
        raise AssertionError("Demo 启动时不应重新构建索引")

    monkeypatch.setattr(
        runtime_module,
        "build_dashscope_embeddings",
        fake_embeddings,
    )
    monkeypatch.setattr(runtime_module, "Chroma", fake_chroma)
    monkeypatch.setattr(
        runtime_module,
        "build_traditional_chat_model",
        fake_build_chat_model,
    )
    monkeypatch.setattr(
        runtime_module,
        "build_knowledge_path_snapshot",
        fake_path_snapshot,
    )
    monkeypatch.setattr(
        runtime_module,
        "build_knowledge_index",
        reject_index_rebuild,
        raising=False,
    )

    runtime = runtime_module.build_pilot_agentic_runtime_from_project(
        project_root=tmp_path,
        settings=settings,
        spec_path=spec_path.resolve(strict=False),
        persist_directory=persist_directory,
    )

    chat_event = next(event for event in events if event[0] == "chat_model")
    config = chat_event[1]
    assert config.model == "qwen3.7-flash"
    assert runtime.knowledge_root == corpus_root.resolve(strict=False)
    assert runtime.vector_store is fake_vector_store
    assert runtime.chat_model is fake_model
    assert runtime.path_snapshot is fake_snapshot
    assert (
        "open_index",
        {
            "collection_name": "knowledge_chunks",
            "persist_directory": str(persist_directory),
            "embedding_function": fake_embedding,
        },
    ) in events


def test_demo_index_builder_uses_water_cycle_spec_and_independent_directory(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """索引命令应从同一 Demo 规格建库，不借用旧 Pilot 路径。"""

    index_module = importlib.import_module("evaluation.cli.demo_index")
    spec_path = _write_declared_demo_spec(tmp_path)
    persist_directory = (tmp_path / DEMO_PERSIST_DIRECTORY).resolve(
        strict=False
    )
    settings = {
        "DASHSCOPE_API_KEY": "test-key",
        "DASHSCOPE_BASE_URL": "https://example.test/v1",
        "EMBEDDING_MODEL": "test-embedding",
        "EMBEDDING_DIMENSIONS": "1024",
        "EMBEDDING_BATCH_SIZE": "10",
    }
    fake_embedding = object()
    received: dict[str, object] = {}

    def fake_embedding_factory(**options: object) -> object:
        received["embedding_options"] = options
        return fake_embedding

    def fake_build_vector_store(**options: object) -> object:
        received["build_options"] = options
        return object()

    monkeypatch.setattr(
        index_module,
        "build_dashscope_embeddings",
        fake_embedding_factory,
    )
    monkeypatch.setattr(
        index_module,
        "build_pilot_vector_store",
        fake_build_vector_store,
    )

    result = index_module.build_demo_index_from_project(
        project_root=tmp_path,
        settings=settings,
    )

    assert result == persist_directory
    assert received["build_options"] == {
        "project_root": tmp_path.resolve(strict=False),
        "spec_path": spec_path.resolve(strict=False),
        "persist_directory": persist_directory,
        "embedding": fake_embedding,
    }
    assert received["embedding_options"] == {
        "model": "test-embedding",
        "dimensions": 1024,
        "batch_size": 10,
        "api_key": "test-key",
        "base_url": "https://example.test/v1",
    }
