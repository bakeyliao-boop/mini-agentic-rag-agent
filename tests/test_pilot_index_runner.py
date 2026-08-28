"""独立 Pilot 向量索引命令行入口测试。"""

import importlib
from pathlib import Path


def test_build_pilot_index_from_project_wires_isolated_index(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """项目入口应读取 Embedding 配置并构建独立 Pilot 索引。"""

    runner = importlib.import_module("evaluation.cli.pilot_index")
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
    calls: list[tuple[object, ...]] = []

    def fake_build_embeddings(**kwargs: object) -> object:
        calls.append(("embedding", kwargs))
        return fake_embedding

    def fake_build_vector_store(
        project_root: Path,
        spec_path: Path,
        persist_directory: Path,
        embedding: object,
    ) -> object:
        calls.append(
            (
                "index",
                project_root,
                spec_path,
                persist_directory,
                embedding,
            )
        )
        return fake_vector_store

    monkeypatch.setattr(
        runner,
        "build_dashscope_embeddings",
        fake_build_embeddings,
    )
    monkeypatch.setattr(
        runner,
        "build_pilot_vector_store",
        fake_build_vector_store,
    )

    result = runner.build_pilot_index_from_project(
        project_root=tmp_path,
        settings=settings,
    )

    expected_spec_path = (
        tmp_path
        / "evaluation"
        / "pilots"
        / "multi_chunk_guide_001.json"
    )
    expected_persist_directory = (
        tmp_path / "data" / "pilots" / "multi-chunk-guide-001"
    ).resolve(strict=False)
    assert result == expected_persist_directory
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
            "index",
            tmp_path,
            expected_spec_path,
            expected_persist_directory,
            fake_embedding,
        ),
    ]


def test_build_pilot_index_uses_default_directory_when_setting_is_missing(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """旧 .env 缺少 Pilot 目录配置时，应使用稳定的默认目录。"""

    runner = importlib.import_module("evaluation.cli.pilot_index")
    settings = {
        "DASHSCOPE_API_KEY": "test-key",
        "DASHSCOPE_BASE_URL": (
            "https://dashscope.aliyuncs.com/compatible-mode/v1"
        ),
        "EMBEDDING_MODEL": "text-embedding-v4",
        "EMBEDDING_DIMENSIONS": "1024",
        "EMBEDDING_BATCH_SIZE": "10",
    }
    fake_embedding = object()
    calls: list[tuple[object, ...]] = []

    def fake_build_embeddings(**kwargs: object) -> object:
        calls.append(("embedding", kwargs))
        return fake_embedding

    def fake_build_vector_store(
        project_root: Path,
        spec_path: Path,
        persist_directory: Path,
        embedding: object,
    ) -> object:
        calls.append(
            (
                "index",
                project_root,
                spec_path,
                persist_directory,
                embedding,
            )
        )
        return object()

    monkeypatch.setattr(
        runner,
        "build_dashscope_embeddings",
        fake_build_embeddings,
    )
    monkeypatch.setattr(
        runner,
        "build_pilot_vector_store",
        fake_build_vector_store,
    )

    result = runner.build_pilot_index_from_project(
        project_root=tmp_path,
        settings=settings,
    )

    expected_directory = (
        tmp_path / "data" / "pilots" / "multi-chunk-guide-001"
    ).resolve(strict=False)
    assert result == expected_directory
    assert calls[1] == (
        "index",
        tmp_path,
        (
            tmp_path
            / "evaluation"
            / "pilots"
            / "multi_chunk_guide_001.json"
        ),
        expected_directory,
        fake_embedding,
    )


def test_main_loads_settings_builds_pilot_index_and_prints_path(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    """main 应加载环境配置、构建 Pilot 索引并打印保存目录。"""

    runner = importlib.import_module("evaluation.cli.pilot_index")
    settings = {"DASHSCOPE_API_KEY": "test-key"}
    persist_directory = tmp_path / "data" / "pilots" / "pilot-index"
    calls: list[tuple[object, ...]] = []

    def fake_load_settings(project_root: Path) -> dict[str, str]:
        calls.append(("load", project_root))
        return settings

    def fake_build_index(
        project_root: Path,
        settings: dict[str, str],
    ) -> Path:
        calls.append(("build", project_root, settings))
        return persist_directory

    monkeypatch.setattr(
        runner,
        "load_settings_from_env",
        fake_load_settings,
        raising=False,
    )
    monkeypatch.setattr(
        runner,
        "build_pilot_index_from_project",
        fake_build_index,
    )

    runner.main(project_root=tmp_path)

    assert calls == [
        ("load", tmp_path),
        ("build", tmp_path, settings),
    ]
    assert capsys.readouterr().out == (
        f"Pilot vector index saved to: {persist_directory}\n"
    )
