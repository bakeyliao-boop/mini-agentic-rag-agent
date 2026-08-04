from importlib import import_module

from langchain.agents.middleware.tool_call_limit import (
    ToolCallLimitMiddleware,
)
from langchain_core.language_models.fake_chat_models import (
    FakeMessagesListChatModel,
)
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import StructuredTool


class ToolCallingFakeModel(FakeMessagesListChatModel):
    """按测试预设顺序返回工具调用的离线对话模型。"""

    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        return self


def test_build_knowledge_agent_registers_model_tools_and_system_prompt(
    monkeypatch,
) -> None:
    """知识库 Agent 应使用指定模型、三个工具和固定系统规则创建。"""

    agent_module = import_module("app.agent")
    build_knowledge_agent = getattr(
        agent_module,
        "build_knowledge_agent",
        None,
    )
    assert build_knowledge_agent is not None, "build_knowledge_agent 尚未实现"

    fake_agent = object()
    fake_model = object()
    fake_tools = [object(), object(), object()]
    received_options: list[dict[str, object]] = []

    def fake_create_agent(**options):
        received_options.append(options)
        return fake_agent

    monkeypatch.setattr(agent_module, "create_agent", fake_create_agent)

    result = build_knowledge_agent(
        chat_model=fake_model,
        tools=fake_tools,
    )

    assert result is fake_agent
    assert len(received_options) == 1
    assert received_options[0]["model"] is fake_model
    assert received_options[0]["tools"] is fake_tools
    assert (
        received_options[0]["system_prompt"]
        == agent_module.KNOWLEDGE_AGENT_SYSTEM_PROMPT
    )


def test_build_knowledge_agent_limits_each_run_to_six_tool_calls(
    monkeypatch,
) -> None:
    """每次运行最多允许执行 6 次工具调用。"""

    agent_module = import_module("app.agent")
    received_options: dict[str, object] = {}

    def fake_create_agent(**options):
        received_options.update(options)
        return object()

    monkeypatch.setattr(agent_module, "create_agent", fake_create_agent)

    agent_module.build_knowledge_agent(
        chat_model=object(),
        tools=[],
    )

    middleware = received_options.get("middleware")
    assert isinstance(middleware, list), "尚未配置工具调用次数限制"
    assert len(middleware) == 1

    tool_call_limiter = middleware[0]
    assert isinstance(tool_call_limiter, ToolCallLimitMiddleware)
    assert tool_call_limiter.run_limit == 6
    assert tool_call_limiter.exit_behavior == "error"


def test_knowledge_agent_executes_search_then_read() -> None:
    """知识库 Agent 应能执行 search、read 后再生成回答。"""

    tool_call_order: list[str] = []

    def fake_ls(path: str = "/") -> dict[str, object]:
        tool_call_order.append("ls")
        return {"path": path, "entries": []}

    def fake_search(
        query: str,
        path: str = "/",
        limit: int = 5,
    ) -> dict[str, object]:
        tool_call_order.append("search")
        return {
            "hits": [
                {
                    "path": "/课程资源/智慧农场.md",
                    "start_line": 3,
                    "end_line": 3,
                    "score": 0.9,
                    "preview": "气象站采集环境数据。",
                }
            ],
            "usage": "candidate_only",
        }

    def fake_read(
        path: str,
        start_line: int = 1,
        limit: int = 80,
    ) -> dict[str, object]:
        tool_call_order.append("read")
        return {
            "path": path,
            "lines": [
                {
                    "line": start_line,
                    "text": "气象站采集环境数据。",
                }
            ],
            "next_line": None,
        }

    tools = [
        StructuredTool.from_function(
            func=fake_ls,
            name="ls",
            description="浏览知识库目录。",
        ),
        StructuredTool.from_function(
            func=fake_search,
            name="search",
            description="搜索知识库候选内容。",
        ),
        StructuredTool.from_function(
            func=fake_read,
            name="read",
            description="读取 Markdown 原文。",
        ),
    ]
    fake_model = ToolCallingFakeModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "search",
                        "args": {
                            "query": "气象站能做什么",
                            "path": "/",
                            "limit": 1,
                        },
                        "id": "search-call",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "read",
                        "args": {
                            "path": "/课程资源/智慧农场.md",
                            "start_line": 3,
                            "limit": 1,
                        },
                        "id": "read-call",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="气象站可以采集环境数据。"),
        ]
    )

    agent_module = import_module("app.agent")
    knowledge_agent = agent_module.build_knowledge_agent(
        chat_model=fake_model,
        tools=tools,
    )

    result = knowledge_agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "气象站能做什么？",
                }
            ]
        }
    )

    assert tool_call_order == ["search", "read"]
    assert result["messages"][-1].content == "气象站可以采集环境数据。"


def test_extract_tool_traces_matches_calls_with_results() -> None:
    """工具轨迹应配对调用和结果，但不重复保存完整工具结果。"""

    agent_module = import_module("app.agent")
    extract_tool_traces = getattr(
        agent_module,
        "extract_tool_traces",
        None,
    )
    assert extract_tool_traces is not None, "extract_tool_traces 尚未实现"

    messages = [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "search",
                    "args": {
                        "query": "气象站",
                        "path": "/课程资源",
                        "limit": 2,
                    },
                    "id": "search-call",
                    "type": "tool_call",
                }
            ],
        ),
        ToolMessage(
            content='{"hits": []}',
            tool_call_id="search-call",
            status="success",
        ),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "read",
                    "args": {
                        "path": "/课程资源/智慧农场.md",
                        "start_line": 3,
                        "limit": 1,
                    },
                    "id": "read-call",
                    "type": "tool_call",
                }
            ],
        ),
        ToolMessage(
            content='{"lines": []}',
            tool_call_id="read-call",
            status="success",
        ),
    ]

    result = extract_tool_traces(messages)

    assert result == [
        {
            "step": 1,
            "tool_call_id": "search-call",
            "name": "search",
            "args": {
                "query": "气象站",
                "path": "/课程资源",
                "limit": 2,
            },
            "status": "success",
        },
        {
            "step": 2,
            "tool_call_id": "read-call",
            "name": "read",
            "args": {
                "path": "/课程资源/智慧农场.md",
                "start_line": 3,
                "limit": 1,
            },
            "status": "success",
        },
    ]


def test_extract_tool_traces_marks_missing_tool_result() -> None:
    """找不到对应 ToolMessage 时应标记为 missing_result。"""

    agent_module = import_module("app.agent")
    messages = [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "search",
                    "args": {
                        "query": "气象站",
                        "path": "/",
                        "limit": 2,
                    },
                    "id": "search-call",
                    "type": "tool_call",
                }
            ],
        )
    ]

    result = agent_module.extract_tool_traces(messages)

    assert result == [
        {
            "step": 1,
            "tool_call_id": "search-call",
            "name": "search",
            "args": {
                "query": "气象站",
                "path": "/",
                "limit": 2,
            },
            "status": "missing_result",
        }
    ]


def test_extract_tool_traces_records_tool_error() -> None:
    """工具返回错误消息时，轨迹状态应记录为 error。"""

    agent_module = import_module("app.agent")
    messages = [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "read",
                    "args": {
                        "path": "/课程资源/不存在.md",
                        "start_line": 1,
                        "limit": 1,
                    },
                    "id": "read-call",
                    "type": "tool_call",
                }
            ],
        ),
        ToolMessage(
            content="文件不存在",
            tool_call_id="read-call",
            status="error",
        ),
    ]

    result = agent_module.extract_tool_traces(messages)

    assert result == [
        {
            "step": 1,
            "tool_call_id": "read-call",
            "name": "read",
            "args": {
                "path": "/课程资源/不存在.md",
                "start_line": 1,
                "limit": 1,
            },
            "status": "error",
        }
    ]
