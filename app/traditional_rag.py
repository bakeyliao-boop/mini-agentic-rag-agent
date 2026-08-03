"""传统 RAG 基线流程。"""

from time import perf_counter

from langchain_chroma import Chroma

from app.indexer import search_chroma_index


def answer_with_traditional_rag(
    question: str,
    vector_store: Chroma,
    chat_model: object,
    path: str = "/",
    limit: int = 5,
) -> dict[str, object]:
    """检索固定数量的候选内容，将其放入提示词并调用对话模型。"""

    started_at = perf_counter()
    search_result = search_chroma_index(
        vector_store,
        question,
        path=path,
        limit=limit,
    )
    hits = search_result["hits"]
    if not hits:
        return {
            "answer": "",
            "hits": [],
        }

    context = "\n\n".join(
        hit["preview"]
        for hit in hits
    )
    prompt = (
        "请仅根据下面的知识库候选内容回答问题。\n\n"
        f"候选内容：\n{context}\n\n"
        f"问题：{question}"
    )

    response = chat_model.invoke(prompt)
    latency_ms = (perf_counter() - started_at) * 1_000
    token_usage = getattr(response, "usage_metadata", None) or {}
    return {
        "answer": response.content,
        "hits": hits,
        "latency_ms": latency_ms,
        "token_usage": token_usage,
    }
