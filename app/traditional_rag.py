"""传统 RAG 基线流程。"""

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI

from app.indexer import search_chroma_index


@dataclass(frozen=True, slots=True)
class TraditionalRagConfig:
    """传统 RAG 对照实验使用的固定配置。"""

    model: str = "qwen3.6-flash"
    temperature: float = 0
    top_k: int = 5
    corpus_version: str = "education-v1"


def resolve_traditional_corpus_root(
    project_root: Path,
    config: TraditionalRagConfig,
) -> Path:
    """根据固定语料版本返回规范的真实知识库目录。"""

    return (
        project_root / "knowledge" / config.corpus_version
    ).resolve(strict=False)


def build_traditional_chat_model(
    config: TraditionalRagConfig,
    api_key: str,
    base_url: str,
) -> ChatOpenAI:
    """使用固定基线配置创建百炼 OpenAI-compatible 对话模型。"""

    return ChatOpenAI(
        model=config.model,
        temperature=config.temperature,
        api_key=api_key,
        base_url=base_url,
        extra_body={"enable_thinking": False},
    )


def answer_with_traditional_rag(
    question: str,
    vector_store: Chroma,
    chat_model: object,
    path: str = "/",
    config: TraditionalRagConfig = TraditionalRagConfig(),
) -> dict[str, object]:
    """检索固定数量的候选内容，将其放入提示词并调用对话模型。"""

    started_at = perf_counter()
    search_result = search_chroma_index(
        vector_store,
        question,
        path=path,
        limit=config.top_k,
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
