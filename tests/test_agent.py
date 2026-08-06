from importlib import import_module

from langchain.agents.middleware.tool_call_limit import (
    ToolCallLimitMiddleware,
)
from langchain_core.language_models.fake_chat_models import (
    FakeMessagesListChatModel,
)
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langgraph.checkpoint.memory import InMemorySaver

from app.models import GroundedAnswer


class ToolCallingFakeModel(FakeMessagesListChatModel):
    """按测试预设顺序返回工具调用的离线对话模型。"""

    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        return self


def structured_answer_message(answer: str, call_id: str) -> AIMessage:
    """生成供离线 Agent 测试使用的 GroundedAnswer 工具调用。"""

    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": "GroundedAnswer",
                "args": {
                    "answer_type": "conversation",
                    "answer": answer,
                    "evidence_ids": [],
                },
                "id": call_id,
                "type": "tool_call",
            }
        ],
    )


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


def test_system_prompt_requires_scoped_search_after_ls() -> None:
    """通过 ls 定位目录后，系统规则应要求在该目录内搜索。"""

    agent_module = import_module("app.agent")

    assert (
        "通过 ls 找到具体目录后，必须将该目录作为 search 的 path 参数"
        in agent_module.KNOWLEDGE_AGENT_SYSTEM_PROMPT
    )


def test_knowledge_agent_stops_searching_after_sufficient_read() -> None:
    """read 已提供充分证据后，Agent 应立即提交结构化回答。"""

    agent_module = import_module("app.agent")
    system_prompt = agent_module.KNOWLEDGE_AGENT_SYSTEM_PROMPT

    assert (
        "read 已返回足以回答问题的原文时，禁止继续调用 search 或 ls"
        in system_prompt
    )
    assert (
        "必须立即提交 GroundedAnswer，并引用 read 返回的 evidence_id"
        in system_prompt
    )
    assert "禁止使用相似关键词重复 search" in system_prompt


def test_knowledge_agent_uses_explicit_prompt_version() -> None:
    """当前 Agent Prompt 应具有可追踪的独立版本号。"""

    agent_module = import_module("app.agent")

    assert agent_module.KNOWLEDGE_AGENT_PROMPT_VERSION == "Prompt-V1.2"


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
    assert tool_call_limiter.exit_behavior == "end"


def test_build_knowledge_agent_does_not_count_grounded_answer_as_tool(
    monkeypatch,
) -> None:
    """结构化 GroundedAnswer 不应占用知识库工具调用次数。"""

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

    middleware = received_options["middleware"]
    tool_call_limiter = middleware[0]

    assert tool_call_limiter._matches_tool_filter({"name": "ls"}) is True
    assert (
        tool_call_limiter._matches_tool_filter(
            {"name": "GroundedAnswer"}
        )
        is False
    )


def test_build_knowledge_agent_uses_in_memory_checkpointer(
    monkeypatch,
) -> None:
    """知识库 Agent 应使用内存保存器维护临时会话状态。"""

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

    checkpointer = received_options.get("checkpointer")
    assert isinstance(checkpointer, InMemorySaver), "尚未配置内存会话状态"


def test_build_knowledge_agent_uses_grounded_answer_response_format(
    monkeypatch,
) -> None:
    """知识库 Agent 应使用 GroundedAnswer 作为结构化输出格式。"""

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

    assert received_options.get("response_format") is GroundedAnswer


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
        },
        config={"configurable": {"thread_id": "test-thread"}},
    )

    assert tool_call_order == ["search", "read"]
    assert result["messages"][-1].content == "气象站可以采集环境数据。"


def test_knowledge_agent_remembers_messages_in_same_thread() -> None:
    """使用相同 thread_id 时，第二轮应保留第一轮对话消息。"""

    fake_model = ToolCallingFakeModel(
        responses=[
            structured_answer_message("我记住了。", "answer-call-1"),
            structured_answer_message(
                "你刚才说项目代号是小云。",
                "answer-call-2",
            ),
        ]
    )
    agent_module = import_module("app.agent")
    knowledge_agent = agent_module.build_knowledge_agent(
        chat_model=fake_model,
        tools=[],
    )
    thread_config = {
        "configurable": {
            "thread_id": "memory-test-thread",
        }
    }

    knowledge_agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "项目代号是小云。",
                }
            ]
        },
        config=thread_config,
    )
    second_result = knowledge_agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "我刚才说的项目代号是什么？",
                }
            ]
        },
        config=thread_config,
    )

    assert [
        message.content
        for message in second_result["messages"]
        if isinstance(message, HumanMessage)
    ] == [
        "项目代号是小云。",
        "我刚才说的项目代号是什么？",
    ]
    assert second_result["structured_response"] == GroundedAnswer(
        answer_type="conversation",
        answer="你刚才说项目代号是小云。",
        evidence_ids=[],
    )


def test_knowledge_agent_isolates_messages_between_threads() -> None:
    """不同 thread_id 之间不应共享对话消息。"""

    fake_model = ToolCallingFakeModel(
        responses=[
            structured_answer_message(
                "已收到 A 会话消息。",
                "thread-a-answer",
            ),
            structured_answer_message(
                "已收到 B 会话消息。",
                "thread-b-answer",
            ),
        ]
    )
    agent_module = import_module("app.agent")
    knowledge_agent = agent_module.build_knowledge_agent(
        chat_model=fake_model,
        tools=[],
    )

    knowledge_agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "A 会话的项目代号是小云。",
                }
            ]
        },
        config={"configurable": {"thread_id": "thread-A"}},
    )
    thread_b_result = knowledge_agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "这是 B 会话。",
                }
            ]
        },
        config={"configurable": {"thread_id": "thread-B"}},
    )

    assert [
        message.content
        for message in thread_b_result["messages"]
        if isinstance(message, HumanMessage)
    ] == [
        "这是 B 会话。",
    ]
    assert thread_b_result["structured_response"] == GroundedAnswer(
        answer_type="conversation",
        answer="已收到 B 会话消息。",
        evidence_ids=[],
    )


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


def test_extract_tool_traces_ignores_grounded_answer() -> None:
    """工具轨迹应忽略 LangChain 内部的 GroundedAnswer 调用。"""

    agent_module = import_module("app.agent")
    messages = [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "search",
                    "args": {"query": "气象站"},
                    "id": "search-call",
                    "type": "tool_call",
                },
                {
                    "name": "GroundedAnswer",
                    "args": {
                        "answer_type": "knowledge",
                        "answer": "气象站可以监测天气。",
                        "evidence_ids": ["run-001:evidence-1"],
                    },
                    "id": "answer-call",
                    "type": "tool_call",
                },
            ],
        ),
        ToolMessage(
            content='{"hits": []}',
            tool_call_id="search-call",
            name="search",
            status="success",
        ),
        ToolMessage(
            content="Returning structured response",
            tool_call_id="answer-call",
            name="GroundedAnswer",
            status="success",
        ),
    ]

    result = agent_module.extract_tool_traces(messages)

    assert result == [
        {
            "step": 1,
            "tool_call_id": "search-call",
            "name": "search",
            "args": {"query": "气象站"},
            "status": "success",
        }
    ]
