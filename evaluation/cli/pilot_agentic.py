"""长文档 Pilot Agentic 运行时接线。"""

import json
from collections.abc import Mapping
from pathlib import Path

from langchain_chroma import Chroma

from evaluation.cli.pilot_index import (
    PILOT_SPEC_RELATIVE_PATH,
    resolve_pilot_persist_directory,
)
from rag_core.agentic.runner import AgenticRuntime
from rag_core.knowledge.indexer import build_dashscope_embeddings
from rag_core.knowledge.store import build_knowledge_path_snapshot
from rag_core.settings import _required_setting
from rag_core.traditional.service import (
    TraditionalRagConfig,
    build_traditional_chat_model,
)


def build_pilot_agentic_runtime_from_project(
    project_root: Path,
    settings: Mapping[str, str],
) -> AgenticRuntime:
    """使用冻结 Pilot 语料与已有索引构建 Agentic 运行时。"""

    spec_path = project_root / PILOT_SPEC_RELATIVE_PATH
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

    knowledge_root = (
        project_root / Path(root_setting.strip())
    ).resolve(strict=False)
    if not knowledge_root.is_dir():
        raise FileNotFoundError(
            f"pilot corpus root does not exist: {knowledge_root}"
        )

    persist_directory = resolve_pilot_persist_directory(
        project_root,
        settings,
    )
    if not persist_directory.is_dir():
        raise FileNotFoundError(
            f"pilot index does not exist: {persist_directory}; "
            "run `python -m evaluation.cli.pilot_index` first"
        )

    api_key = _required_setting(settings, "DASHSCOPE_API_KEY")
    base_url = _required_setting(settings, "DASHSCOPE_BASE_URL")
    embedding_model = _required_setting(settings, "EMBEDDING_MODEL")
    embedding_dimensions = int(
        _required_setting(settings, "EMBEDDING_DIMENSIONS")
    )
    embedding_batch_size = int(
        _required_setting(settings, "EMBEDDING_BATCH_SIZE")
    )

    embedding = build_dashscope_embeddings(
        model=embedding_model,
        dimensions=embedding_dimensions,
        batch_size=embedding_batch_size,
        api_key=api_key,
        base_url=base_url,
    )
    vector_store = Chroma(
        collection_name="knowledge_chunks",
        persist_directory=str(persist_directory),
        embedding_function=embedding,
    )
    stored = vector_store.get(limit=1, include=[])
    stored_ids = stored.get("ids") if isinstance(stored, dict) else None
    if not isinstance(stored_ids, list) or not stored_ids:
        raise RuntimeError(
            "pilot index is empty; run "
            "`python -m evaluation.cli.pilot_index` first"
        )
    config = TraditionalRagConfig()
    chat_model = build_traditional_chat_model(
        config,
        api_key,
        base_url,
    )
    path_snapshot = build_knowledge_path_snapshot(knowledge_root)

    return AgenticRuntime(
        knowledge_root=knowledge_root,
        vector_store=vector_store,
        chat_model=chat_model,
        path_snapshot=path_snapshot,
    )
