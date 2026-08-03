import importlib


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
        limit=1,
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
