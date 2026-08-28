"""独立多 Chunk Pilot 向量索引构建入口。"""

from collections.abc import Mapping
from pathlib import Path

from evaluation.pilot_loader import build_pilot_vector_store
from rag_core.knowledge.indexer import build_dashscope_embeddings
from rag_core.settings import _required_setting, load_settings_from_env

PILOT_SPEC_RELATIVE_PATH = Path(
    "evaluation/pilots/multi_chunk_guide_001.json"
)
DEFAULT_PILOT_PERSIST_RELATIVE_PATH = Path(
    "data/pilots/multi-chunk-guide-001"
)


def resolve_pilot_persist_directory(
    project_root: Path,
    settings: Mapping[str, str],
) -> Path:
    """解析 Pilot 本地索引目录；旧 .env 缺失配置时使用默认目录。"""

    configured_path = settings.get("PILOT_CHROMA_PERSIST_DIR")
    relative_path = (
        Path(configured_path.strip())
        if isinstance(configured_path, str) and configured_path.strip()
        else DEFAULT_PILOT_PERSIST_RELATIVE_PATH
    )
    return (project_root / relative_path).resolve(strict=False)


def build_pilot_index_from_project(
    project_root: Path,
    settings: Mapping[str, str],
) -> Path:
    """读取项目配置，为固定 Pilot 构建独立的 Chroma 索引。"""

    api_key = _required_setting(settings, "DASHSCOPE_API_KEY")
    base_url = _required_setting(settings, "DASHSCOPE_BASE_URL")
    embedding_model = _required_setting(settings, "EMBEDDING_MODEL")
    embedding_dimensions = int(
        _required_setting(settings, "EMBEDDING_DIMENSIONS")
    )
    embedding_batch_size = int(
        _required_setting(settings, "EMBEDDING_BATCH_SIZE")
    )
    persist_directory = resolve_pilot_persist_directory(
        project_root,
        settings,
    )
    embedding = build_dashscope_embeddings(
        model=embedding_model,
        dimensions=embedding_dimensions,
        batch_size=embedding_batch_size,
        api_key=api_key,
        base_url=base_url,
    )
    build_pilot_vector_store(
        project_root=project_root,
        spec_path=project_root / PILOT_SPEC_RELATIVE_PATH,
        persist_directory=persist_directory,
        embedding=embedding,
    )
    return persist_directory


def main(project_root: Path | None = None) -> None:
    """加载项目配置，构建独立 Pilot 索引并打印保存目录。"""

    resolved_project_root = (
        project_root
        if project_root is not None
        else Path(__file__).resolve().parent.parent.parent
    )
    settings = load_settings_from_env(resolved_project_root)
    persist_directory = build_pilot_index_from_project(
        project_root=resolved_project_root,
        settings=settings,
    )
    print(f"Pilot vector index saved to: {persist_directory}")


if __name__ == "__main__":
    main()
