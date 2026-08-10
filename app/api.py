from fastapi import APIRouter, HTTPException
from collections.abc import Callable
from app.api_models import TraditionalChatRequest, TraditionalChatResponse
from app.traditional_rag import answer_with_traditional_rag



TraditionalChatHandler = Callable[[str,str],dict[str,object]]
router = APIRouter()
_traditional_chat_handler: TraditionalChatHandler | None = None


def build_traditional_chat_handler(
        vector_store:object,
        chat_model:object,
)->TraditionalChatHandler:
    """
    使用现有索引和模型创建可重复调用的传统RAG处理器
    """
    def handler(
            question:str,
            path:str,
    )->dict[str,object]:
        return answer_with_traditional_rag(
            question=question,
            vector_store=vector_store,
            chat_model=chat_model,
            path=path,
        )
    return handler

def configure_traditional_chat(
        handler: TraditionalChatHandler,
    ) -> None:
    """配置传统 RAG 处理器，供 HTTP 请求转发使用。"""
    global _traditional_chat_handler
    _traditional_chat_handler = handler


def run_traditional_chat(
    question: str,
    path: str,
) -> dict[str, object]:
    """传统 RAG 运行入口"""

    if _traditional_chat_handler is None:

        raise HTTPException(
        status_code=503,
        detail="traditional RAG runtime is not configured",
    )
    return _traditional_chat_handler(question, path)

@router.get("/health")
def health_check() -> dict[str, str]:
    """返回固定响应，表示服务可以正常工作。"""

    return {"status": "ok"}

@router.post(
    "/chat/traditional",
    response_model=TraditionalChatResponse,
)
def traditional_chat(
    request: TraditionalChatRequest,
) -> dict[str, object]:
    """把 HTTP 请求转交给已经配置好的传统 RAG 处理器。"""

    return run_traditional_chat(request.question, request.path)
