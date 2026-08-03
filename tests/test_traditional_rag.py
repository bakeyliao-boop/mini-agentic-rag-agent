import importlib
from pathlib import Path


class RecordingChatModel:
    """记录收到的提示词，并返回固定答案。"""

    def __init__(self) -> None:
        self.received_prompt: object | None = None

    def invoke(self, prompt: object) -> object:
        self.received_prompt = prompt
        return type(
            "ChatResponse",
            (),
            {"content": "智慧农场使用自动灌溉系统。"},
        )()


def test_answer_with_traditional_rag_uses_search_hits_in_prompt(
    monkeypatch,
) -> None:
    """传统 RAG 应把问题和 search 候选放入提示词后调用模型。"""

    traditional_rag = importlib.import_module("app.traditional_rag")
    vector_store = object()
    chat_model = RecordingChatModel()
    hits = [
        {
            "path": "/课程资源/智慧农场.md",
            "start_line": 1,
            "end_line": 3,
            "score": 1.0,
            "preview": "# 智慧农场\n\n自动灌溉系统可以分类灌溉。",
        }
    ]
    search_calls: list[dict[str, object]] = []

    def fake_search(
        received_store: object,
        query: str,
        path: str = "/",
        limit: int = 5,
    ) -> dict[str, object]:
        search_calls.append(
            {
                "vector_store": received_store,
                "query": query,
                "path": path,
                "limit": limit,
            }
        )
        return {"hits": hits, "usage": "candidate_only"}

    monkeypatch.setattr(traditional_rag, "search_chroma_index", fake_search)

    result = traditional_rag.answer_with_traditional_rag(
        question="智慧农场使用什么灌溉方式？",
        vector_store=vector_store,
        chat_model=chat_model,
        path="/课程资源",
        config=traditional_rag.TraditionalRagConfig(top_k=1),
    )

    assert search_calls == [
        {
            "vector_store": vector_store,
            "query": "智慧农场使用什么灌溉方式？",
            "path": "/课程资源",
            "limit": 1,
        }
    ]
    prompt_text = str(chat_model.received_prompt)
    assert "智慧农场使用什么灌溉方式？" in prompt_text
    assert "自动灌溉系统可以分类灌溉" in prompt_text
    assert result["answer"] == "智慧农场使用自动灌溉系统。"
    assert result["hits"] == hits
    assert isinstance(result["latency_ms"], float)


def test_answer_with_traditional_rag_skips_model_when_no_hits(
    monkeypatch,
) -> None:
    """没有 search 候选时，不应调用对话模型编写答案。"""

    traditional_rag = importlib.import_module("app.traditional_rag")

    def fake_search(*args, **kwargs) -> dict[str, object]:
        return {"hits": [], "usage": "candidate_only"}

    class UnexpectedChatModel:
        def invoke(self, prompt: object) -> object:
            raise AssertionError("没有候选内容时不应调用对话模型")

    monkeypatch.setattr(traditional_rag, "search_chroma_index", fake_search)

    result = traditional_rag.answer_with_traditional_rag(
        question="知识库中没有答案的问题",
        vector_store=object(),
        chat_model=UnexpectedChatModel(),
    )

    assert result == {
        "answer": "",
        "hits": [],
    }


def test_answer_with_traditional_rag_returns_latency_ms(
    monkeypatch,
) -> None:
    """传统 RAG 应记录从检索到生成答案的整体耗时。"""

    traditional_rag = importlib.import_module("app.traditional_rag")
    hits = [
        {
            "path": "/智慧农场.md",
            "start_line": 1,
            "end_line": 1,
            "score": 1.0,
            "preview": "自动灌溉系统可以分类灌溉。",
        }
    ]

    def fake_search(*args, **kwargs) -> dict[str, object]:
        return {"hits": hits, "usage": "candidate_only"}

    timestamps = iter([10.0, 10.125])
    monkeypatch.setattr(traditional_rag, "search_chroma_index", fake_search)
    monkeypatch.setattr(
        traditional_rag,
        "perf_counter",
        lambda: next(timestamps),
        raising=False,
    )

    result = traditional_rag.answer_with_traditional_rag(
        question="智慧农场如何灌溉？",
        vector_store=object(),
        chat_model=RecordingChatModel(),
    )

    assert result["latency_ms"] == 125.0


def test_answer_with_traditional_rag_returns_token_usage(
    monkeypatch,
) -> None:
    """传统 RAG 应保存对话模型返回的 token 使用量。"""

    traditional_rag = importlib.import_module("app.traditional_rag")
    hits = [
        {
            "path": "/智慧农场.md",
            "start_line": 1,
            "end_line": 1,
            "score": 1.0,
            "preview": "自动灌溉系统可以分类灌溉。",
        }
    ]

    def fake_search(*args, **kwargs) -> dict[str, object]:
        return {"hits": hits, "usage": "candidate_only"}

    class UsageChatModel:
        def invoke(self, prompt: object) -> object:
            return type(
                "ChatResponse",
                (),
                {
                    "content": "智慧农场使用自动灌溉系统。",
                    "usage_metadata": {
                        "input_tokens": 120,
                        "output_tokens": 30,
                        "total_tokens": 150,
                    },
                },
            )()

    monkeypatch.setattr(traditional_rag, "search_chroma_index", fake_search)

    result = traditional_rag.answer_with_traditional_rag(
        question="智慧农场如何灌溉？",
        vector_store=object(),
        chat_model=UsageChatModel(),
    )

    assert result["token_usage"] == {
        "input_tokens": 120,
        "output_tokens": 30,
        "total_tokens": 150,
    }


def test_traditional_rag_config_uses_fixed_baseline_defaults() -> None:
    """传统 RAG 基线应固定模型、温度、top-k 和语料版本。"""

    traditional_rag = importlib.import_module("app.traditional_rag")

    config = traditional_rag.TraditionalRagConfig()

    assert config.model == "qwen3.6-flash"
    assert config.temperature == 0
    assert config.top_k == 5
    assert config.corpus_version == "education-v1"


def test_answer_with_traditional_rag_uses_config_top_k(
    monkeypatch,
) -> None:
    """传统 RAG 检索数量应来自配置中的 top_k。"""

    traditional_rag = importlib.import_module("app.traditional_rag")
    config = traditional_rag.TraditionalRagConfig(top_k=3)
    received_limits: list[int] = []

    def fake_search(
        received_store: object,
        query: str,
        path: str = "/",
        limit: int = 5,
    ) -> dict[str, object]:
        received_limits.append(limit)
        return {"hits": [], "usage": "candidate_only"}

    monkeypatch.setattr(traditional_rag, "search_chroma_index", fake_search)

    traditional_rag.answer_with_traditional_rag(
        question="智慧农场如何灌溉？",
        vector_store=object(),
        chat_model=RecordingChatModel(),
        config=config,
    )

    assert received_limits == [3]


def test_build_traditional_chat_model_uses_config_and_dashscope(
    monkeypatch,
) -> None:
    """模型工厂应把固定配置和百炼连接参数传给 ChatOpenAI。"""

    traditional_rag = importlib.import_module("app.traditional_rag")
    config = traditional_rag.TraditionalRagConfig()
    received_options: list[dict[str, object]] = []
    fake_model = object()

    def fake_chat_openai(**options):
        received_options.append(options)
        return fake_model

    monkeypatch.setattr(
        traditional_rag,
        "ChatOpenAI",
        fake_chat_openai,
        raising=False,
    )

    result = traditional_rag.build_traditional_chat_model(
        config=config,
        api_key="test-key",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    assert result is fake_model
    assert received_options == [
        {
            "model": "qwen3.6-flash",
            "temperature": 0,
            "api_key": "test-key",
            "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        }
    ]


def test_resolve_traditional_corpus_root_uses_config_version(
    tmp_path: Path,
) -> None:
    """传统 RAG 应根据语料版本定位固定的知识库目录。"""

    traditional_rag = importlib.import_module("app.traditional_rag")
    config = traditional_rag.TraditionalRagConfig(
        corpus_version="education-v1",
    )
    expected_root = tmp_path / "knowledge" / "education-v1"
    expected_root.mkdir(parents=True)

    result = traditional_rag.resolve_traditional_corpus_root(
        project_root=tmp_path,
        config=config,
    )

    assert result == expected_root.resolve(strict=False)
