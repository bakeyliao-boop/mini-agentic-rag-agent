"""独立多 Chunk Pilot 向量索引构建入口。"""

from collections.abc import Mapping
from pathlib import Path

from evaluation.pilot_loader import build_pilot_vector_store
from rag_core.knowledge.indexer import build_dashscope_embeddings
from rag_core.settings import _required_setting, load_settings_from_env

PILOT_SPEC_RELATIVE_PATH = Path(
    "evaluation/pilots/multi_chunk_guide_001.json"
)


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
    persist_setting = _required_setting(
        settings,
        "PILOT_CHROMA_PERSIST_DIR",
    )

    persist_directory = (
        project_root / Path(persist_setting)
    ).resolve(strict=False)
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
