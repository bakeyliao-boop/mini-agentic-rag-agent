"""兼容原有命令的 Agentic RAG 单问题命令行入口。"""

import argparse
import json
from pathlib import Path

from rag_core.settings import load_settings_from_env
from rag_core.agentic.runner import run_agentic_question_from_project


def main(
    project_root: Path | None = None,
    question: str | None = None,
    thread_id: str = "smoke-thread",
) -> None:
    """读取项目配置，运行一个问题并输出 JSON 结果。"""

    if question is None:
        parser = argparse.ArgumentParser(
            description="运行一个 Agentic RAG 问题。",
        )
        parser.add_argument("question", help="需要向知识库提问的问题。")
        parser.add_argument(
            "--thread-id",
            default=thread_id,
            help="用于保存临时对话状态的会话 ID。",
        )
        arguments = parser.parse_args()
        question = arguments.question
        thread_id = arguments.thread_id

    resolved_project_root = (
        project_root
        if project_root is not None
        else Path(__file__).resolve().parent.parent
    )
    settings = load_settings_from_env(resolved_project_root)
    result = run_agentic_question_from_project(
        project_root=resolved_project_root,
        question=question,
        thread_id=thread_id,
        settings=settings,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
