from collections.abc import Callable
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI

from app.api import (
    TraditionalChatHandler,
    configure_traditional_chat,
    build_traditional_chat_handler,
    router,
)

from app.agent_runner import (
    build_agentic_runtime_from_project
    as build_traditional_runtime_from_project,
)
from app.baseline_runner import load_settings_from_env

TraditionalChatHandlerFactory = Callable[[], TraditionalChatHandler]

def build_traditional_handler_factory(
        project_root:Path,
)->TraditionalChatHandlerFactory:
    """
    创建一个在应用启动时才初始化传统RAG的工厂
    """
    def factory()->TraditionalChatHandler:
        settings = load_settings_from_env(project_root)
        runtime = build_traditional_runtime_from_project(
            project_root,
            settings,
        )
        return build_traditional_chat_handler(
            runtime.vector_store,
            runtime.chat_model,
        )
    return factory

def create_app(
    traditional_handler_factory: TraditionalChatHandlerFactory | None = None,
) -> FastAPI:
    """创建 FastAPI，并在启动时配置一次传统 RAG 处理器。"""

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        if traditional_handler_factory is not None:
            handler = traditional_handler_factory()
            configure_traditional_chat(handler)

        yield

    application = FastAPI(lifespan=lifespan)
    application.include_router(router)
    return application


app = create_app()
