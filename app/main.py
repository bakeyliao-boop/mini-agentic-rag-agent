from collections.abc import Callable
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from app.api_errors import ApiError, api_error_handler
from app.api import (
    AgenticChatHandler,
    TraditionalChatHandler,
    build_agentic_chat_handler,
    build_traditional_chat_handler,
    configure_traditional_chat,
    configure_agentic_chat,
    router,
)

from app.agent_runner import (
    build_agentic_runtime_from_project,
    build_agentic_runtime_from_project
    as build_traditional_runtime_from_project,
)
from app.baseline_runner import load_settings_from_env

TraditionalChatHandlerFactory = Callable[[], TraditionalChatHandler]

ChatHandlers = tuple[
    TraditionalChatHandler,
    AgenticChatHandler,
]
ChatHandlersFactory = Callable[[],ChatHandlers,]

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

def build_chat_handlers_factory(
        project_root:Path,
)->ChatHandlersFactory:
    '''创建共享一个运行时的传统和Agentic处理器工厂'''
    def factory()->ChatHandlers:
        settings = load_settings_from_env(project_root)
        runtime =  build_agentic_runtime_from_project(
            project_root,
            settings,
        )
        traditional_handler = build_traditional_chat_handler(
            runtime.vector_store,
            runtime.chat_model
        )
        agentic_handler = build_agentic_chat_handler(
            runtime,
        )
        return traditional_handler,agentic_handler
    return factory

def create_app(
        traditional_handler_factory:(
            TraditionalChatHandlerFactory | None
        ) = None,
        chat_handlers_factory:ChatHandlersFactory | None = None,
)->FastAPI:
    '''创建FastAPI，并在启动时配置聊天处理器'''
    @asynccontextmanager
    async def lifespan(application:FastAPI):
        if chat_handlers_factory is not None:
            (
                traditional_handler,
                agentic_handler,
            )=chat_handlers_factory()
            configure_traditional_chat(
                traditional_handler,
            )
            configure_agentic_chat(
                agentic_handler,
            )
        elif traditional_handler_factory is not None:
            traditional_handler = (
                traditional_handler_factory()
            )
            configure_traditional_chat(
                traditional_handler,
            )

        yield

    application = FastAPI(lifespan=lifespan)
    application.add_exception_handler(
        ApiError,
        api_error_handler,
    )
    application.include_router(router)
    return application

def create_default_app(project_root:Path)->FastAPI:
    '''创建接入共享聊天处理器工厂的默认应用'''
    chat_handlers_factory = build_chat_handlers_factory(
        project_root,
    )
    return create_app(
        chat_handlers_factory=chat_handlers_factory
    )

PROJECT_ROOT = Path(__file__).resolve().parent.parent
app = create_default_app(PROJECT_ROOT)
