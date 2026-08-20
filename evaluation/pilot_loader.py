"""独立 Pilot 规范与单文档语料加载。"""

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
    """只读取 Pilot 声明的单个 Markdown，并生成对应 Chunk。"""

    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    if not isinstance(spec, dict):
        raise ValueError("pilot specification must be an object")

    corpus = spec.get("corpus")
    if not isinstance(corpus, dict):
        raise ValueError("pilot corpus configuration must be an object")
    if corpus.get("isolation") != "single_document":
        raise ValueError("pilot corpus isolation must be single_document")

    root_setting = corpus.get("root")
    if not isinstance(root_setting, str) or not root_setting.strip():
        raise ValueError("pilot corpus root must be a non-empty string")

    source_setting = corpus.get("source_path")
    if not isinstance(source_setting, str) or not source_setting.strip():
        raise ValueError("pilot source_path must be a non-empty string")

    corpus_root = (project_root / root_setting.strip()).resolve(strict=False)
    virtual_path = normalize_virtual_path(source_setting)
    source_path = resolve_knowledge_path(virtual_path, corpus_root)
    lines = read_markdown_lines(source_path)
    return chunk_markdown_lines(virtual_path, lines)


def build_pilot_vector_store(
    project_root: Path,
    spec_path: Path,
    persist_directory: Path,
    embedding: Embeddings,
) -> Chroma:
    """加载 Pilot 指定的独立语料，并写入独立的 Chroma 目录。"""

    chunks = load_pilot_chunks(project_root, spec_path)
    return build_chroma_index(chunks, persist_directory, embedding)
