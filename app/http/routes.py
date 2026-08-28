"""FastAPI 健康检查与聊天路由。"""

from fastapi import APIRouter
from app.http.errors import ApiError
from collections.abc import Callable
from app.http.models import (
    AgenticChatRequest,
    AgenticChatResponse,
    TraditionalChatRequest,
    TraditionalChatResponse,
)
from rag_core.traditional.service import answer_with_traditional_rag
from rag_core.agentic.runner import AgenticRuntime, run_agentic_question



TraditionalChatHandler = Callable[[str,str],dict[str,object]]
AgenticChatHandler = Callable[[str,str],dict[str,object]]

router = APIRouter()
_traditional_chat_handler: TraditionalChatHandler | None = None
_agentic_chat_handler:AgenticChatHandler | None = None


def build_traditional_chat_handler( #优先配置 对应的知识库和模型  1.记录
        vector_store:object,    #接收应用启动时候创建的索引和模型
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

def configure_traditional_chat( #2.登记
        handler: TraditionalChatHandler,
    ) -> None:
    """配置传统 RAG 处理器，供 HTTP 请求转发使用。"""
    global _traditional_chat_handler
    _traditional_chat_handler = handler


def run_traditional_chat( #3.执行
    question: str,
    path: str,
) -> dict[str, object]:
    """传统 RAG 运行入口"""

    if _traditional_chat_handler is None:   #检查启动时是否已经保存handler

        raise ApiError(
            status_code=503,
            code="runtime_not_configured",
            message="traditional RAG runtime is not configured",
        )
    return _traditional_chat_handler(question, path)

def configure_agentic_chat(
        handler:AgenticChatHandler,
)-> None:
    '''保存Agentic RAG处理器，供HTTP请求使用'''
    global _agentic_chat_handler
    _agentic_chat_handler = handler

def run_agentic_chat(
        question:str,
        thread_id:str,
)->dict[str,object]:
    '''调用已经配置好的Agentic RAG处理器'''
    if _agentic_chat_handler is None:
        raise ApiError(
            status_code=503,
            code="runtime_not_configured",
            message="agentic RAG runtime is not configured",
        )
    return _agentic_chat_handler(question,thread_id)

def build_agentic_chat_handler(
        runtime:AgenticRuntime,
)->AgenticChatHandler:
    '''使用已有运行时创建可重复调用的Agentic RAG处理器'''
    def handler(
            question:str,
            thread_id:str
    )->dict[str,object]:
        return run_agentic_question( #固定保存runtime-每次直接收question、thread_id-调用agentic-rag
            runtime=runtime,
            question=question,
            thread_id=thread_id,
        )
    return handler




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

@router.post("/chat/agentic",response_model=AgenticChatResponse,)
def agentic_chat(request:AgenticChatRequest,)->dict[str,object]:
    '''把HTTP请求转交给Agentic RAG处理器'''
    return run_agentic_chat(
        request.question,
        request.thread_id,
    )
