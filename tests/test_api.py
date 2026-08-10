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
