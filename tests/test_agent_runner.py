import json
from importlib import import_module
from pathlib import Path

from langchain_core.messages import AIMessage


def test_run_agentic_question_from_project_wires_all_components(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """真实运行入口应按顺序连接索引、模型、工具和 Agent。"""

    agent_runner = import_module("app.agent_runner")
    settings = {
        "DASHSCOPE_API_KEY": "test-key",
        "DASHSCOPE_BASE_URL": (
            "https://dashscope.aliyuncs.com/compatible-mode/v1"
        ),
        "EMBEDDING_MODEL": "text-embedding-v4",
        "EMBEDDING_DIMENSIONS": "1024",
        "CHROMA_PERSIST_DIR": "./data/chroma",
    }
    knowledge_root = tmp_path / "knowledge" / "education-v1"
    knowledge_root.mkdir(parents=True)
    fake_embeddings = object()
    fake_vector_store = object()
    fake_chat_model = object()
    fake_tools = [object(), object(), object()]
    fake_messages = [AIMessage(content="气象站可以采集环境数据。")]
    fake_traces = [
        {
            "step": 1,
            "tool_call_id": "search-call",
            "name": "search",
            "args": {"query": "气象站", "path": "/", "limit": 2},
            "status": "success",
        }
    ]
    events: list[tuple[object, ...]] = []

    def fake_build_embeddings(**options):
        events.append(("embedding", options))
        return fake_embeddings

    def fake_build_index(root, persist_directory, embedding):
        events.append(("index", root, persist_directory, embedding))
        return fake_vector_store

    def fake_build_chat_model(config, api_key, base_url):
        events.append(("chat", config, api_key, base_url))
        return fake_chat_model

    def fake_build_tools(knowledge_root, vector_store):
        events.append(("tools", knowledge_root, vector_store))
        return fake_tools

    def fake_build_agent(chat_model, tools):
        events.append(("agent", chat_model, tools))

        class FakeAgent:
            def invoke(self, input_data, config):
                events.append(("invoke", input_data, config))
                return {"messages": fake_messages}

        return FakeAgent()

    def fake_extract_traces(messages):
        events.append(("trace", messages))
        return fake_traces

    monkeypatch.setattr(
        agent_runner,
        "build_dashscope_embeddings",
        fake_build_embeddings,
    )
    monkeypatch.setattr(
        agent_runner,
        "build_knowledge_index",
        fake_build_index,
    )
    monkeypatch.setattr(
        agent_runner,
        "build_traditional_chat_model",
        fake_build_chat_model,
    )
    monkeypatch.setattr(
        agent_runner,
        "build_knowledge_tools",
        fake_build_tools,
    )
    monkeypatch.setattr(
        agent_runner,
        "build_knowledge_agent",
        fake_build_agent,
    )
    monkeypatch.setattr(
        agent_runner,
        "extract_tool_traces",
        fake_extract_traces,
    )

    result = agent_runner.run_agentic_question_from_project(
        project_root=tmp_path,
        question="气象站能做什么？",
        thread_id="smoke-thread",
        settings=settings,
    )

    assert result == {
        "answer": "气象站可以采集环境数据。",
        "tool_traces": fake_traces,
        "thread_id": "smoke-thread",
    }
    assert [event[0] for event in events] == [
        "embedding",
        "index",
        "chat",
        "tools",
        "agent",
        "invoke",
        "trace",
    ]
    assert events[3] == ("tools", knowledge_root.resolve(), fake_vector_store)
    assert events[5] == (
        "invoke",
        {"messages": [{"role": "user", "content": "气象站能做什么？"}]},
        {"configurable": {"thread_id": "smoke-thread"}},
    )


def test_main_loads_settings_runs_question_and_prints_json(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    """命令行入口应读取配置、运行问题并输出 JSON。"""

    agent_runner = import_module("app.agent_runner")
    main = getattr(agent_runner, "main", None)
    assert main is not None, "agent_runner.main 尚未实现"

    settings = {"DASHSCOPE_API_KEY": "test-key"}
    fake_result = {
        "answer": "气象站可以采集环境数据。",
        "tool_traces": [],
        "thread_id": "smoke-thread",
    }
    calls: list[tuple[object, ...]] = []

    def fake_load_settings(project_root: Path) -> dict[str, str]:
        calls.append(("load", project_root))
        return settings

    def fake_run_question(
        project_root: Path,
        question: str,
        thread_id: str,
        settings: dict[str, str],
    ) -> dict[str, object]:
        calls.append(
            ("run", project_root, question, thread_id, settings)
        )
        return fake_result

    monkeypatch.setattr(
        agent_runner,
        "load_settings_from_env",
        fake_load_settings,
    )
    monkeypatch.setattr(
        agent_runner,
        "run_agentic_question_from_project",
        fake_run_question,
    )

    main(
        project_root=tmp_path,
        question="气象站能做什么？",
        thread_id="smoke-thread",
    )

    assert calls == [
        ("load", tmp_path),
        (
            "run",
            tmp_path,
            "气象站能做什么？",
            "smoke-thread",
            settings,
        ),
    ]
    assert json.loads(capsys.readouterr().out) == fake_result
