"""独立 Pilot 语料加载器测试。"""

import hashlib
import json
from pathlib import Path

import pytest

from evaluation import pilot_loader
from evaluation.pilot_loader import load_pilot_chunks
from rag_core.models import Chunk


PILOT_FIXTURE_RELATIVE_PATH = Path(
    "evaluation/fixtures/multi_chunk_guide_001/corpus/"
    "政务与公共服务/深圳市/知识产权公共服务/"
    "深圳市知识产权公共服务事项办事指南（第二版）.md"
)
PILOT_FIXTURE_SHA256 = (
    "c94b6b1d12e9aace29f218e793fb574d5c1b1ad74c2930892c7d309e560bf2d6"
)
PILOT_FIXTURE_LINE_COUNT = 4203


def test_frozen_pilot_fixture_has_expected_content_snapshot() -> None:
    """提交的 Pilot 语料快照必须存在且保持固定内容。"""

    project_root = Path(__file__).resolve().parent.parent
    source_path = project_root / PILOT_FIXTURE_RELATIVE_PATH

    assert source_path.is_file(), (
        "冻结 Pilot 语料必须随仓库提供，"
        "不能依赖本机 knowledge/yunzhi-eval-v1 或 Windows 转换脚本。"
    )

    content = source_path.read_bytes()
    assert hashlib.sha256(content).hexdigest() == PILOT_FIXTURE_SHA256
    assert len(content.decode("utf-8").splitlines()) == PILOT_FIXTURE_LINE_COUNT


def test_load_pilot_chunks_reads_only_declared_source(
    tmp_path: Path,
) -> None:
    """单文档 Pilot 不应扫描同一语料目录中的其他 Markdown。"""

    corpus_root = tmp_path / "knowledge" / "yunzhi-eval-v1"
    source_path = corpus_root / "政务服务" / "办事指南.md"
    source_path.parent.mkdir(parents=True)
    source_path.write_text(
        "# 办事指南\n\n这是 Pilot 指定的目标内容。\n",
        encoding="utf-8",
        newline="\n",
    )

    unrelated_path = corpus_root / "其他资源" / "超长段落.md"
    unrelated_path.parent.mkdir(parents=True)
    unrelated_path.write_text(
        "无关内容" * 300,
        encoding="utf-8",
        newline="\n",
    )

    spec_path = tmp_path / "evaluation" / "pilots" / "pilot.json"
    spec_path.parent.mkdir(parents=True)
    spec_path.write_text(
        json.dumps(
            {
                "id": "multi-chunk-guide-001",
                "corpus": {
                    "root": "knowledge/yunzhi-eval-v1",
                    "isolation": "single_document",
                    "source_path": "/政务服务/办事指南.md",
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
        newline="\n",
    )

    chunks = load_pilot_chunks(tmp_path, spec_path)

    assert [chunk.path for chunk in chunks] == [
        "/政务服务/办事指南.md"
    ]
    assert chunks[0].text == "# 办事指南\n\n这是 Pilot 指定的目标内容。"
    assert unrelated_path.is_file()


def test_load_real_multi_chunk_guide_pilot_returns_eighty_six_chunks() -> None:
    """冻结深圳办事指南应被隔离加载为固定的 86 个 Chunk。"""

    project_root = Path(__file__).resolve().parent.parent
    spec_path = (
        project_root
        / "evaluation"
        / "pilots"
        / "multi_chunk_guide_001.json"
    )
    chunks = load_pilot_chunks(project_root, spec_path)

    expected_virtual_path = (
        "/政务与公共服务/深圳市/知识产权公共服务/"
        "深圳市知识产权公共服务事项办事指南（第二版）.md"
    )
    assert len(chunks) == 86
    assert {chunk.path for chunk in chunks} == {expected_virtual_path}


def test_build_pilot_vector_store_uses_isolated_chunks_and_directory(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Pilot 索引应只写入指定的独立 Chroma 目录。"""

    spec_path = tmp_path / "evaluation" / "pilots" / "pilot.json"
    persist_directory = tmp_path / "data" / "pilots" / "pilot-chroma"
    fake_embedding = object()
    fake_vector_store = object()
    fake_chunks = [
        Chunk(
            chunk_id="pilot-chunk-1",
            path="/办事指南.md",
            start_line=1,
            end_line=2,
            text="# 办事指南\n目标内容",
        )
    ]
    calls: list[tuple[object, ...]] = []

    def fake_load_chunks(
        project_root: Path,
        source_spec_path: Path,
    ) -> list[Chunk]:
        calls.append(("load", project_root, source_spec_path))
        return fake_chunks

    def fake_build_chroma_index(
        chunks: list[Chunk],
        target_directory: Path,
        embedding: object,
    ) -> object:
        calls.append(
            ("build", chunks, target_directory, embedding)
        )
        return fake_vector_store

    monkeypatch.setattr(
        pilot_loader,
        "load_pilot_chunks",
        fake_load_chunks,
    )
    monkeypatch.setattr(
        pilot_loader,
        "build_chroma_index",
        fake_build_chroma_index,
        raising=False,
    )

    result = pilot_loader.build_pilot_vector_store(
        project_root=tmp_path,
        spec_path=spec_path,
        persist_directory=persist_directory,
        embedding=fake_embedding,
    )

    assert result is fake_vector_store
    assert calls == [
        ("load", tmp_path, spec_path),
        (
            "build",
            fake_chunks,
            persist_directory,
            fake_embedding,
        ),
    ]
