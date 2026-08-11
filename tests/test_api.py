from importlib import import_module

from fastapi.testclient import TestClient


def test_health_returns_ok() -> None:
    """健康检查接口应返回可用于探活的固定响应。"""

    main_module = import_module("app.main")
    client = TestClient(main_module.app)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_traditional_chat_returns_rag_result(monkeypatch) -> None:
    """传统 RAG 接口应接收问题和路径，并返回现有流程的结果。"""

    api_module = import_module("app.api")
    main_module = import_module("app.main")
    received_requests: list[tuple[str, str]] = []
    fake_result = {
        "answer": "气象站可以实时监测天气状况。",
        "hits": [
            {
                "path": "/课程资源/智慧农场.md",
                "start_line": 27,
                "end_line": 27,
                "preview": "气象站可以实时监测天气状况。",
            }
        ],
        "latency_ms": 12.5,
        "token_usage": {
            "input_tokens": 20,
            "output_tokens": 8,
            "total_tokens": 28,
        },
    }

    def fake_run_traditional_chat(
        question: str,
        path: str,
    ) -> dict[str, object]:
        received_requests.append((question, path))
        return fake_result

    monkeypatch.setattr(
        api_module,
        "run_traditional_chat",
        fake_run_traditional_chat,
        raising=False,
    )
    client = TestClient(main_module.app)

    response = client.post(
        "/chat/traditional",
        json={
            "question": "气象站能做什么？",
            "path": "/课程资源",
        },
    )

    assert response.status_code == 200
    assert received_requests == [
        ("气象站能做什么？", "/课程资源")
    ]
    assert response.json() == fake_result


def test_agentic_chat_returns_agent_result(monkeypatch) -> None:
    """Agentic 接口应转交问题和线程编号，并返回带证据的结果。"""

    api_module = import_module("app.api")
    main_module = import_module("app.main")
    received_requests: list[tuple[str, str]] = []
    fake_result = {
        "answer_type": "knowledge",
        "answer": "气象站可以实时监测天气状况。",
        "citations": [
            {
                "path": "/课程资源/智慧农场.md",
                "start_line": 27,
                "end_line": 27,
                "quote": "气象站可以实时监测天气状况。",
            }
        ],
        "tool_traces": [
            {
                "step": 1,
                "tool_call_id": "call-001",
                "name": "read",
                "args": {"path": "/课程资源/智慧农场.md"},
                "status": "success",
            }
        ],
        "token_usage": {
            "input_tokens": 20,
            "output_tokens": 8,
            "total_tokens": 28,
        },
        "thread_id": "api-thread-001",
    }

    def fake_run_agentic_chat(
        question: str,
        thread_id: str,
    ) -> dict[str, object]:
        received_requests.append((question, thread_id))
        return fake_result

    monkeypatch.setattr(
        api_module,
        "run_agentic_chat",
        fake_run_agentic_chat,
        raising=False,
    )
    client = TestClient(main_module.app)

    response = client.post(
        "/chat/agentic",
        json={
            "question": "气象站能做什么？",
            "thread_id": "api-thread-001",
        },
    )

    assert response.status_code == 200
    assert received_requests == [
        ("气象站能做什么？", "api-thread-001")
    ]
    assert response.json() == fake_result


def test_chat_endpoints_return_unified_runtime_error(
    monkeypatch,
) -> None:
    """两个聊天接口未配置运行时时应返回统一错误结构。"""

    api_module = import_module("app.api")
    main_module = import_module("app.main")

    monkeypatch.setattr(
        api_module,
        "_traditional_chat_handler",
        None,
    )
    monkeypatch.setattr(
        api_module,
        "_agentic_chat_handler",
        None,
    )

    client = TestClient(main_module.app)

    traditional_response = client.post(
        "/chat/traditional",
        json={
            "question": "气象站能做什么？",
            "path": "/",
        },
    )
    agentic_response = client.post(
        "/chat/agentic",
        json={
            "question": "气象站能做什么？",
            "thread_id": "thread-001",
        },
    )

    assert traditional_response.status_code == 503
    assert agentic_response.status_code == 503
    assert traditional_response.json() == {
        "error": {
            "code": "runtime_not_configured",
            "message": "traditional RAG runtime is not configured",
        }
    }
    assert agentic_response.json() == {
        "error": {
            "code": "runtime_not_configured",
            "message": "agentic RAG runtime is not configured",
        }
    }


def test_configured_agentic_chat_handler_is_reused(
    monkeypatch,
) -> None:
    """配置一次的 Agentic 处理器应能连续处理多个线程的请求。"""

    api_module = import_module("app.api")
    received_requests: list[tuple[str, str]] = []

    def fake_handler(
        question: str,
        thread_id: str,
    ) -> dict[str, object]:
        received_requests.append((question, thread_id))
        return {
            "answer_type": "knowledge",
            "answer": f"回答：{question}",
            "citations": [],
            "tool_traces": [],
            "token_usage": {},
            "thread_id": thread_id,
        }

    monkeypatch.setattr(
        api_module,
        "_agentic_chat_handler",
        None,
    )

    api_module.configure_agentic_chat(fake_handler)

    first_result = api_module.run_agentic_chat(
        "气象站能做什么？",
        "thread-001",
    )
    second_result = api_module.run_agentic_chat(
        "智慧农场如何灌溉？",
        "thread-002",
    )

    assert received_requests == [
        ("气象站能做什么？", "thread-001"),
        ("智慧农场如何灌溉？", "thread-002"),
    ]
    assert first_result["thread_id"] == "thread-001"
    assert second_result["thread_id"] == "thread-002"


def test_build_agentic_chat_handler_reuses_runtime(
    monkeypatch,
) -> None:
    """Agentic handler 应复用已有运行时并转交问题和线程编号。"""

    api_module = import_module("app.api")
    fake_runtime = object()
    fake_result = {
        "answer_type": "knowledge",
        "answer": "气象站可以实时监测天气状况。",
        "citations": [],
        "tool_traces": [],
        "token_usage": {},
        "thread_id": "thread-001",
    }
    received_calls: list[dict[str, object]] = []

    def fake_run_agentic_question(
        *,
        runtime: object,
        question: str,
        thread_id: str,
    ) -> dict[str, object]:
        received_calls.append(
            {
                "runtime": runtime,
                "question": question,
                "thread_id": thread_id,
            }
        )
        return fake_result

    monkeypatch.setattr(
        api_module,
        "run_agentic_question",
        fake_run_agentic_question,
        raising=False,
    )

    handler = api_module.build_agentic_chat_handler(fake_runtime)
    result = handler("气象站能做什么？", "thread-001")

    assert result is fake_result
    assert received_calls == [
        {
            "runtime": fake_runtime,
            "question": "气象站能做什么？",
            "thread_id": "thread-001",
        }
    ]


def test_configured_traditional_chat_handler_is_reused(
    monkeypatch,
) -> None:
    """配置一次的传统 RAG 处理器应能连续处理多个请求。"""

    api_module = import_module("app.api")
    received_requests: list[tuple[str, str]] = []

    def fake_handler(
        question: str,
        path: str,
    ) -> dict[str, object]:
        received_requests.append((question, path))
        return {
            "answer": f"回答：{question}",
            "hits": [],
        }

    monkeypatch.setattr(
        api_module,
        "_traditional_chat_handler",
        None,
        raising=False,
    )

    api_module.configure_traditional_chat(fake_handler)

    first_result = api_module.run_traditional_chat(
        "气象站能做什么？",
        "/课程资源",
    )
    second_result = api_module.run_traditional_chat(
        "自动灌溉有什么作用？",
        "/课程资源",
    )

    assert received_requests == [
        ("气象站能做什么？", "/课程资源"),
        ("自动灌溉有什么作用？", "/课程资源"),
    ]
    assert first_result["answer"] == "回答：气象站能做什么？"
    assert second_result["answer"] == "回答：自动灌溉有什么作用？"


def test_build_traditional_chat_handler_reuses_rag_components(
    monkeypatch,
) -> None:
    """处理器工厂应复用既有索引和模型调用传统 RAG。"""

    api_module = import_module("app.api")
    fake_vector_store = object()
    fake_chat_model = object()
    received_calls: list[dict[str, object]] = []

    def fake_answer_with_traditional_rag(
        *,
        question: str,
        vector_store: object,
        chat_model: object,
        path: str,
    ) -> dict[str, object]:
        received_calls.append(
            {
                "question": question,
                "vector_store": vector_store,
                "chat_model": chat_model,
                "path": path,
            }
        )
        return {
            "answer": f"回答：{question}",
            "hits": [],
        }

    monkeypatch.setattr(
        api_module,
        "answer_with_traditional_rag",
        fake_answer_with_traditional_rag,
        raising=False,
    )

    handler = api_module.build_traditional_chat_handler(
        vector_store=fake_vector_store,
        chat_model=fake_chat_model,
    )

    first_result = handler("气象站能做什么？", "/课程资源")
    second_result = handler("自动灌溉有什么作用？", "/课程资源")

    assert received_calls == [
        {
            "question": "气象站能做什么？",
            "vector_store": fake_vector_store,
            "chat_model": fake_chat_model,
            "path": "/课程资源",
        },
        {
            "question": "自动灌溉有什么作用？",
            "vector_store": fake_vector_store,
            "chat_model": fake_chat_model,
            "path": "/课程资源",
        },
    ]
    assert first_result["answer"] == "回答：气象站能做什么？"
    assert second_result["answer"] == "回答：自动灌溉有什么作用？"


def test_create_app_configures_traditional_handler_once(
    monkeypatch,
) -> None:
    """一次应用生命周期中只应构建并配置一次传统处理器。"""

    main_module = import_module("app.main")
    events: list[tuple[object, ...]] = []

    def fake_handler(
        question: str,
        path: str,
    ) -> dict[str, object]:
        return {
            "answer": question,
            "hits": [],
        }

    def fake_handler_factory():
        events.append(("build_handler",))
        return fake_handler

    def fake_configure(handler) -> None:
        events.append(("configure_handler", handler))

    monkeypatch.setattr(
        main_module,
        "configure_traditional_chat",
        fake_configure,
        raising=False,
    )

    application = main_module.create_app(
        traditional_handler_factory=fake_handler_factory,
    )

    with TestClient(application) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/health").status_code == 200

    assert events == [
        ("build_handler",),
        ("configure_handler", fake_handler),
    ]


def test_create_app_configures_shared_chat_handlers_once(
    monkeypatch,
) -> None:
    """应用启动时应调用一次共享工厂，并配置两个聊天处理器。"""

    main_module = import_module("app.main")
    fake_traditional_handler = object()
    fake_agentic_handler = object()
    events: list[tuple[object, ...]] = []

    def fake_chat_handlers_factory():
        events.append(("build_chat_handlers",))
        return (
            fake_traditional_handler,
            fake_agentic_handler,
        )

    def fake_configure_traditional(handler) -> None:
        events.append(("configure_traditional", handler))

    def fake_configure_agentic(handler) -> None:
        events.append(("configure_agentic", handler))

    monkeypatch.setattr(
        main_module,
        "configure_traditional_chat",
        fake_configure_traditional,
    )
    monkeypatch.setattr(
        main_module,
        "configure_agentic_chat",
        fake_configure_agentic,
        raising=False,
    )

    application = main_module.create_app(
        chat_handlers_factory=fake_chat_handlers_factory,
    )

    with TestClient(application) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/health").status_code == 200

    assert events == [
        ("build_chat_handlers",),
        (
            "configure_traditional",
            fake_traditional_handler,
        ),
        ("configure_agentic", fake_agentic_handler),
    ]


def test_build_traditional_handler_factory_defers_runtime_until_startup(
    tmp_path,
    monkeypatch,
) -> None:
    """创建工厂时不应初始化资源，调用工厂时才构建运行时。"""

    main_module = import_module("app.main")
    settings = {"DASHSCOPE_API_KEY": "test-key"}
    fake_vector_store = object()
    fake_chat_model = object()
    fake_handler = object()
    events: list[tuple[object, ...]] = []

    class FakeRuntime:
        vector_store = fake_vector_store
        chat_model = fake_chat_model

    def fake_load_settings(project_root):
        events.append(("load_settings", project_root))
        return settings

    def fake_build_runtime(project_root, received_settings):
        events.append(
            ("build_runtime", project_root, received_settings)
        )
        return FakeRuntime()

    def fake_build_handler(vector_store, chat_model):
        events.append(
            ("build_handler", vector_store, chat_model)
        )
        return fake_handler

    monkeypatch.setattr(
        main_module,
        "load_settings_from_env",
        fake_load_settings,
        raising=False,
    )
    monkeypatch.setattr(
        main_module,
        "build_traditional_runtime_from_project",
        fake_build_runtime,
        raising=False,
    )
    monkeypatch.setattr(
        main_module,
        "build_traditional_chat_handler",
        fake_build_handler,
        raising=False,
    )

    handler_factory = main_module.build_traditional_handler_factory(
        project_root=tmp_path,
    )

    assert events == []

    result = handler_factory()

    assert result is fake_handler
    assert events == [
        ("load_settings", tmp_path),
        ("build_runtime", tmp_path, settings),
        ("build_handler", fake_vector_store, fake_chat_model),
    ]


def test_build_chat_handlers_factory_reuses_one_runtime(
    tmp_path,
    monkeypatch,
) -> None:
    """共享工厂应只构建一次运行时，并生成传统和 Agentic 处理器。"""

    main_module = import_module("app.main")
    settings = {"DASHSCOPE_API_KEY": "test-key"}
    fake_vector_store = object()
    fake_chat_model = object()
    fake_traditional_handler = object()
    fake_agentic_handler = object()
    events: list[tuple[object, ...]] = []

    class FakeRuntime:
        vector_store = fake_vector_store
        chat_model = fake_chat_model

    fake_runtime = FakeRuntime()

    def fake_load_settings(project_root):
        events.append(("load_settings", project_root))
        return settings

    def fake_build_runtime(project_root, received_settings):
        events.append(
            ("build_runtime", project_root, received_settings)
        )
        return fake_runtime

    def fake_build_traditional_handler(vector_store, chat_model):
        events.append(
            (
                "build_traditional_handler",
                vector_store,
                chat_model,
            )
        )
        return fake_traditional_handler

    def fake_build_agentic_handler(runtime):
        events.append(("build_agentic_handler", runtime))
        return fake_agentic_handler

    monkeypatch.setattr(
        main_module,
        "load_settings_from_env",
        fake_load_settings,
    )
    monkeypatch.setattr(
        main_module,
        "build_agentic_runtime_from_project",
        fake_build_runtime,
        raising=False,
    )
    monkeypatch.setattr(
        main_module,
        "build_traditional_chat_handler",
        fake_build_traditional_handler,
    )
    monkeypatch.setattr(
        main_module,
        "build_agentic_chat_handler",
        fake_build_agentic_handler,
        raising=False,
    )

    handlers_factory = main_module.build_chat_handlers_factory(tmp_path)

    assert events == []

    result = handlers_factory()

    assert result == (
        fake_traditional_handler,
        fake_agentic_handler,
    )
    assert events == [
        ("load_settings", tmp_path),
        ("build_runtime", tmp_path, settings),
        (
            "build_traditional_handler",
            fake_vector_store,
            fake_chat_model,
        ),
        ("build_agentic_handler", fake_runtime),
    ]


def test_create_default_app_wires_shared_chat_handlers_factory(
    tmp_path,
    monkeypatch,
) -> None:
    """默认应用应把共享聊天处理器工厂接入 FastAPI。"""

    main_module = import_module("app.main")
    fake_chat_handlers_factory = object()
    fake_application = object()
    events: list[tuple[object, ...]] = []

    def fake_build_chat_handlers_factory(project_root):
        events.append(("build_chat_handlers_factory", project_root))
        return fake_chat_handlers_factory

    def fake_create_app(*, chat_handlers_factory=None):
        events.append(("create_app", chat_handlers_factory))
        return fake_application

    monkeypatch.setattr(
        main_module,
        "build_chat_handlers_factory",
        fake_build_chat_handlers_factory,
    )
    monkeypatch.setattr(
        main_module,
        "create_app",
        fake_create_app,
    )

    result = main_module.create_default_app(tmp_path)

    assert result is fake_application
    assert events == [
        ("build_chat_handlers_factory", tmp_path),
        ("create_app", fake_chat_handlers_factory),
    ]
