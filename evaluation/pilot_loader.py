"""独立 Pilot 规范与显式声明语料加载。"""

import json
from pathlib import Path

from langchain_core.embeddings import Embeddings
from langchain_chroma import Chroma

from rag_core.knowledge.indexer import (
    build_chroma_index,
    chunk_markdown_lines,
)
from rag_core.knowledge.store import (
    normalize_virtual_path,
    read_markdown_lines,
    resolve_knowledge_path,
)
from rag_core.models import Chunk


def load_pilot_chunks(
    project_root: Path,
    spec_path: Path,
) -> list[Chunk]:
    """只读取 Pilot 明确声明的 Markdown，并按声明顺序生成 Chunk。"""

    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    if not isinstance(spec, dict):
        raise ValueError("pilot specification must be an object")

    corpus = spec.get("corpus")
    if not isinstance(corpus, dict):
        raise ValueError("pilot corpus configuration must be an object")
    root_setting = corpus.get("root")
    if not isinstance(root_setting, str) or not root_setting.strip():
        raise ValueError("pilot corpus root must be a non-empty string")

    isolation = corpus.get("isolation")
    if isolation == "single_document":
        source_setting = corpus.get("source_path")
        if not isinstance(source_setting, str) or not source_setting.strip():
            raise ValueError("pilot source_path must be a non-empty string")
        source_settings = [source_setting]
    elif isolation == "declared_documents":
        declared_sources = corpus.get("source_paths")
        if (
            not isinstance(declared_sources, list)
            or not declared_sources
            or not all(
                isinstance(source, str) and source.strip()
                for source in declared_sources
            )
        ):
            raise ValueError(
                "pilot source_paths must be a non-empty string list"
            )
        source_settings = declared_sources
    else:
        raise ValueError(
            "pilot corpus isolation must be single_document "
            "or declared_documents"
        )

    corpus_root = (project_root / root_setting.strip()).resolve(strict=False)
    virtual_paths = [
        normalize_virtual_path(source_setting)
        for source_setting in source_settings
    ]
    if len(set(virtual_paths)) != len(virtual_paths):
        raise ValueError("pilot source_paths must not contain duplicate paths")

    chunks: list[Chunk] = []
    for virtual_path in virtual_paths:
        source_path = resolve_knowledge_path(virtual_path, corpus_root)
        lines = read_markdown_lines(source_path)
        chunks.extend(chunk_markdown_lines(virtual_path, lines))
    return chunks


def build_pilot_vector_store(
    project_root: Path,
    spec_path: Path,
    persist_directory: Path,
    embedding: Embeddings,
) -> Chroma:
    """加载 Pilot 指定的独立语料，并写入独立的 Chroma 目录。"""

    chunks = load_pilot_chunks(project_root, spec_path)
    return build_chroma_index(chunks, persist_directory, embedding)
