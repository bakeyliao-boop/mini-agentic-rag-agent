"""Agentic RAG 固定评测的命令行入口。"""

from collections.abc import Mapping
from pathlib import Path

from app.agent_runner import (
    build_agentic_runtime_from_project,
    run_agentic_question,
)
from app.baseline_runner import load_settings_from_env
from app.evaluation import (
    load_evaluation_questions,
    run_agentic_evaluation,
    save_evaluation_result,
)
from app.prompts import KNOWLEDGE_AGENT_PROMPT_VERSION
from app.traditional_rag import TraditionalRagConfig

def build_agentic_evaluation_result_filename(
    config: TraditionalRagConfig,
    prompt_version: str,
) -> str:
    """根据模型、思考模式和 Prompt 版本生成结果文件名。"""

    thinking_mode = (
        "thinking-on"
        if config.enable_thinking
        else "thinking-off"
    )
    normalized_prompt_version = prompt_version.lower()

    return (
        f"agentic-baseline-{config.model}-"
        f"{thinking_mode}-"
        f"{normalized_prompt_version}.json"
    )


def run_agentic_evaluation_from_project(
    project_root: Path,
    settings: Mapping[str, str],
) -> Path:
    """运行固定 Agentic RAG 评测并返回结果文件路径。"""

    config = TraditionalRagConfig()
    dataset = load_evaluation_questions(
        project_root / "evaluation" / "questions.json"
    )
    runtime = build_agentic_runtime_from_project(
        project_root=project_root,
        settings=settings,
    )

    def run_question(
        question: str,
        thread_id: str,
    ) -> dict[str, object]:
        return run_agentic_question(
            runtime=runtime,
            question=question,
            thread_id=thread_id,
        )
    result = run_agentic_evaluation(
        dataset=dataset,
        run_question=run_question,
        config=config,
        prompt_version=KNOWLEDGE_AGENT_PROMPT_VERSION,
    )

    output_path = (
        project_root
        / "evaluation"
        / "results"
        / build_agentic_evaluation_result_filename(
            config,
            KNOWLEDGE_AGENT_PROMPT_VERSION,
        )
    )

    save_evaluation_result(
        result=result,
        output_path=output_path,
    )
    return output_path


def main(project_root: Path | None = None) -> None:
    """读取项目配置，运行 Agentic 评测并打印结果路径。"""

    resolved_project_root = (
        project_root
        if project_root is not None
        else Path(__file__).parent.parent
    )
    settings = load_settings_from_env(
        resolved_project_root
    )
    output_path = run_agentic_evaluation_from_project(
        project_root=resolved_project_root,
        settings=settings,
    )
    print(
        "Agentic evaluation result saved to: "
        f"{output_path}"
    )


if __name__ == "__main__":
    main()
