"""水循环课前资源一致性 Demo 的独立索引构建入口。"""

import json
from collections.abc import Mapping
from pathlib import Path

from evaluation.pilot_loader import build_pilot_vector_store
from rag_core.knowledge.indexer import build_dashscope_embeddings
from rag_core.settings import _required_setting, load_settings_from_env


DEMO_SPEC_RELATIVE_PATH = Path(
    "evaluation/demos/teacher_water_cycle_consistency_001.json"
)


def resolve_demo_persist_directory(
    project_root: Path,
    spec_path: Path,
) -> Path:
    """从 Demo 规格读取独立索引目录，并限制在项目根目录中。"""

    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    if not isinstance(spec, dict):
        raise ValueError("demo specification must be an object")
    persist_setting = spec.get("persist_directory")
    if not isinstance(persist_setting, str) or not persist_setting.strip():
        raise ValueError("demo persist_directory must be a non-empty string")

    resolved_project_root = project_root.resolve(strict=False)
    persist_directory = (
        resolved_project_root / Path(persist_setting.strip())
    ).resolve(strict=False)
    try:
        persist_directory.relative_to(resolved_project_root)
    except ValueError as error:
        raise ValueError(
            "demo persist_directory must stay inside project root"
        ) from error
    return persist_directory


def build_demo_index_from_project(
    project_root: Path,
    settings: Mapping[str, str],
) -> Path:
    """按冻结 Demo 规格加载四份 Markdown 并构建独立索引。"""

    resolved_project_root = project_root.resolve(strict=False)
    spec_path = (
        resolved_project_root / DEMO_SPEC_RELATIVE_PATH
    ).resolve(strict=False)
    persist_directory = resolve_demo_persist_directory(
        resolved_project_root,
        spec_path,
    )
    embedding = build_dashscope_embeddings(
        model=_required_setting(settings, "EMBEDDING_MODEL"),
        dimensions=int(_required_setting(settings, "EMBEDDING_DIMENSIONS")),
        batch_size=int(_required_setting(settings, "EMBEDDING_BATCH_SIZE")),
        api_key=_required_setting(settings, "DASHSCOPE_API_KEY"),
        base_url=_required_setting(settings, "DASHSCOPE_BASE_URL"),
    )
    build_pilot_vector_store(
        project_root=resolved_project_root,
        spec_path=spec_path,
        persist_directory=persist_directory,
        embedding=embedding,
    )
    return persist_directory


def main(project_root: Path | None = None) -> None:
    """加载本地配置，构建水循环 Demo 索引并打印保存目录。"""

    resolved_project_root = (
        project_root.resolve(strict=False)
        if project_root is not None
        else Path(__file__).resolve().parent.parent.parent
    )
    settings = load_settings_from_env(resolved_project_root)
    persist_directory = build_demo_index_from_project(
        project_root=resolved_project_root,
        settings=settings,
    )
    print(f"Demo vector index saved to: {persist_directory}")


if __name__ == "__main__":
    main()
