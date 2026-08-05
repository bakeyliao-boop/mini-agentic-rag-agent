"""Agentic RAG 单问题运行入口。"""

import argparse
import json
from collections.abc import Mapping
from pathlib import Path

from app.agent import build_knowledge_agent, extract_tool_traces
from app.baseline_runner import _required_setting, load_settings_from_env
from app.indexer import build_dashscope_embeddings, build_knowledge_index
from app.tools import build_knowledge_tools
from app.traditional_rag import (
    TraditionalRagConfig,
    build_traditional_chat_model,
    resolve_traditional_corpus_root,
)


def run_agentic_question_from_project(
    project_root: Path,
    question: str,
    thread_id: str,
    settings: Mapping[str, str],
) -> dict[str, object]:
    """连接真实组件，运行一个知识库问题并返回回答和工具轨迹。"""

    config = TraditionalRagConfig()
    api_key = _required_setting(settings, "DASHSCOPE_API_KEY")
    base_url = _required_setting(settings, "DASHSCOPE_BASE_URL")
    embedding_model = _required_setting(settings, "EMBEDDING_MODEL")
    embedding_dimensions = int(
        _required_setting(settings, "EMBEDDING_DIMENSIONS")
    )
    persist_setting = _required_setting(settings, "CHROMA_PERSIST_DIR")

    knowledge_root = resolve_traditional_corpus_root(project_root, config)
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
    tools = build_knowledge_tools(knowledge_root, vector_store)
    knowledge_agent = build_knowledge_agent(chat_model, tools)

    agent_result = knowledge_agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": question,
                }
            ]
        },
        config={"configurable": {"thread_id": thread_id}},
    )
    messages = agent_result["messages"]
    answer = messages[-1].content

    return {
        "answer": answer,
        "tool_traces": extract_tool_traces(messages),
        "thread_id": thread_id,
    }


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
