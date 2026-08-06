"""传统 RAG 固定评测的执行入口。"""

from collections.abc import Mapping
from pathlib import Path

from dotenv import dotenv_values

from app.evaluation import (
    load_evaluation_questions,
    run_traditional_baseline,
    save_evaluation_result,
)
from app.indexer import (
    build_dashscope_embeddings,
    build_knowledge_index,
)
from app.traditional_rag import (
    TraditionalRagConfig,
    build_traditional_chat_model,
    resolve_traditional_corpus_root,
)


def load_settings_from_env(project_root: Path) -> dict[str, str]:
    """从项目根目录的 .env 读取非空字符串配置。"""

    env_path = project_root / ".env"
    if not env_path.is_file():
        raise FileNotFoundError(f"environment file does not exist: {env_path}")

    loaded_values = dotenv_values(env_path)
    return {
        name: value
        for name, value in loaded_values.items()
        if isinstance(value, str)
    }


def _required_setting(
    settings: Mapping[str, str],
    name: str,
) -> str:
    """读取必需配置，并拒绝缺失或空字符串。"""

    value = settings.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"missing required setting: {name}")
    return value.strip()


def build_traditional_baseline_result_filename(
    config: TraditionalRagConfig,
) -> str:
    """根据模型和思考模式生成不会覆盖旧基线的结果文件名。"""

    thinking_mode = (
        "thinking-on" if config.enable_thinking else "thinking-off"
    )
    return f"traditional-baseline-{config.model}-{thinking_mode}.json"


def run_traditional_baseline_from_project(
    project_root: Path,
    settings: Mapping[str, str],
) -> Path:
    """连接真实组件，运行固定传统 RAG 评测并返回结果路径。"""

    # settings 是从项目 .env 文件读取的本地运行配置。
    config = TraditionalRagConfig()
    api_key = _required_setting(settings, "DASHSCOPE_API_KEY")
    base_url = _required_setting(settings, "DASHSCOPE_BASE_URL")
    embedding_model = _required_setting(settings, "EMBEDDING_MODEL")
    embedding_dimensions = int(
        _required_setting(settings, "EMBEDDING_DIMENSIONS")
    )
    persist_setting = _required_setting(settings, "CHROMA_PERSIST_DIR")

    knowledge_root = resolve_traditional_corpus_root(project_root, config)
    # 将配置中的相对目录映射为 Chroma 索引的真实磁盘目录。
    persist_directory = (
        project_root / Path(persist_setting)
    ).resolve(strict=False)
    embeddings = build_dashscope_embeddings(
        model=embedding_model,
        dimensions=embedding_dimensions,
        api_key=api_key,
        base_url=base_url,
    )
    vector_store = build_knowledge_index(
        knowledge_root,
        persist_directory,
        embeddings,
    )
    chat_model = build_traditional_chat_model(
        config,
        api_key,
        base_url,
    )

    dataset = load_evaluation_questions(
        project_root / "evaluation" / "questions.json"
    )
    result = run_traditional_baseline(
        dataset,
        vector_store,
        chat_model,
        config,
    )
    output_path = (
        project_root
        / "evaluation"
        / "results"
        / build_traditional_baseline_result_filename(config)
    )
    save_evaluation_result(result, output_path)
    return output_path


def main(project_root: Path | None = None) -> None:
    """读取项目配置，运行传统 RAG baseline 并打印结果路径。"""

    resolved_project_root = (
        project_root
        if project_root is not None
        else Path(__file__).resolve().parent.parent
    )
    settings = load_settings_from_env(resolved_project_root)
    output_path = run_traditional_baseline_from_project(
        project_root=resolved_project_root,
        settings=settings,
    )
    print(f"Baseline result saved to: {output_path}")


if __name__ == "__main__":
    main()
