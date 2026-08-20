"""多 Chunk Pilot A 组实验入口。"""

import json
from collections.abc import Mapping
from pathlib import Path

from langchain_chroma import Chroma

from evaluation.cli.pilot_index import PILOT_SPEC_RELATIVE_PATH
from evaluation.dataset import save_evaluation_result
from evaluation.runners.pilot import (
    run_pilot_arm_a,
    run_pilot_arm_a_answer,
)
from evaluation.scorers.pilot import score_pilot_answer_coverage
from evaluation.scorers.semantic import (
    build_semantic_judge,
    build_semantic_judge_chat_model,
)
from rag_core.knowledge.indexer import build_dashscope_embeddings
from rag_core.settings import _required_setting, load_settings_from_env
from rag_core.traditional.service import (
    TraditionalRagConfig,
    build_traditional_chat_model,
)


def run_pilot_arm_a_from_project(
    project_root: Path,
    settings: Mapping[str, str],
) -> Path:
    """打开独立 Pilot 索引，运行 A 组检索并保存结果。"""

    spec_path = project_root / PILOT_SPEC_RELATIVE_PATH
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    if not isinstance(spec, dict):
        raise ValueError("pilot specification must be an object")

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
    vector_store = Chroma(
        collection_name="knowledge_chunks",
        persist_directory=str(persist_directory),
        embedding_function=embedding,
    )
    result = run_pilot_arm_a(spec, vector_store)

    pilot_id = spec.get("id")
    if not isinstance(pilot_id, str) or not pilot_id.strip():
        raise ValueError("pilot id must be a non-empty string")
    output_path = (
        project_root
        / "evaluation"
        / "results"
        / f"pilot-{pilot_id}-arm-a-retrieval.json"
    )
    save_evaluation_result(result, output_path)
    return output_path


def run_pilot_arm_a_answer_from_project(
    project_root: Path,
    settings: Mapping[str, str],
) -> Path:
    """打开独立 Pilot 索引，构建传统对话模型，运行 A 组回答并保存结果。"""

    spec_path = project_root / PILOT_SPEC_RELATIVE_PATH
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    if not isinstance(spec, dict):
        raise ValueError("pilot specification must be an object")

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
    vector_store = Chroma(
        collection_name="knowledge_chunks",
        persist_directory=str(persist_directory),
        embedding_function=embedding,
    )
    config = TraditionalRagConfig()
    chat_model = build_traditional_chat_model(
        config,
        api_key,
        base_url,
    )
    result = run_pilot_arm_a_answer(
        spec,
        vector_store,
        chat_model,
        config,
    )

    pilot_id = spec.get("id")
    if not isinstance(pilot_id, str) or not pilot_id.strip():
        raise ValueError("pilot id must be a non-empty string")
    output_path = (
        project_root
        / "evaluation"
        / "results"
        / f"pilot-{pilot_id}-arm-a-answer.json"
    )
    save_evaluation_result(result, output_path)
    return output_path


def score_pilot_arm_answer_from_project(
    project_root: Path,
    settings: Mapping[str, str],
) -> Path:
    """构建语义 Judge，对 A 组回答做答案点覆盖评分并保存结果。"""

    spec_path = project_root / PILOT_SPEC_RELATIVE_PATH
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    if not isinstance(spec, dict):
        raise ValueError("pilot specification must be an object")

    pilot_id = spec.get("id")
    if not isinstance(pilot_id, str) or not pilot_id.strip():
        raise ValueError("pilot id must be a non-empty string")
    result_filename = f"pilot-{pilot_id}-arm-a-answer.json"
    answer_path = (
        project_root / "evaluation" / "results" / result_filename
    )
    answer_result = json.loads(answer_path.read_text(encoding="utf-8"))
    if not isinstance(answer_result, dict):
        raise ValueError("pilot arm A answer must be an object")

    chat_model = build_semantic_judge_chat_model(settings)
    judge = build_semantic_judge(chat_model)
    score = score_pilot_answer_coverage(spec, answer_result, judge)

    output_filename = f"pilot-{pilot_id}-arm-a-answer-coverage.json"
    output_path = (
        project_root / "evaluation" / "results" / output_filename
    )
    save_evaluation_result(score, output_path)
    return output_path


def main(project_root:Path | None=None)->None:
    '''加载项目配置，运行Pilot A组检索与回答并打印结果文件路径'''
    resolved_project_root=(
        project_root
        if project_root is not None
        else Path(__file__).resolve().parent.parent.parent
    )
    settings = load_settings_from_env(resolved_project_root)
    retrieval_path = run_pilot_arm_a_from_project(
        project_root=resolved_project_root,
        settings=settings
    )
    answer_path = run_pilot_arm_a_answer_from_project(
        project_root=resolved_project_root,
        settings=settings
    )
    print(f'Pilot arm A retrieval saved to: {retrieval_path}')
    print(f'Pilot arm A answer saved to: {answer_path}')

if __name__ == '__main__':
    main()
