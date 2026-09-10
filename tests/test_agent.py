import json
from importlib import import_module
from pathlib import Path

from langchain.agents.middleware.tool_call_limit import (
    ToolCallLimitMiddleware,
)
from langchain_core.language_models.fake_chat_models import (
    FakeMessagesListChatModel,
)
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langgraph.checkpoint.memory import InMemorySaver

from rag_core.models import GroundedAnswer
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

    agent_module = import_module("rag_core.agentic.agent")
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


def test_agent_receives_content_location_rule_before_reading(monkeypatch) -> None:
    """验证定位规则和直接读取例外传入 Agent，不代表模型一定遵守。"""

    agent_module = import_module("rag_core.agentic.agent")
    received_options: dict[str, object] = {}

    def fake_create_agent(**options):
        received_options.update(options)
        return object()

    monkeypatch.setattr(agent_module, "create_agent", fake_create_agent)
    agent_module.build_knowledge_agent(chat_model=object(), tools=[])

    system_prompt = received_options["system_prompt"]
    required_rules = [
        "查询文件中的具体内容时，只有文件路径而没有相关内容的位置，"
        "应先以该文件路径作为 search 或 grep 的 path 定位，再根据返回行号调用 read。",
        "已有可靠的目标行号时，可以直接 read 对应范围。",
        "用户明确要求通读全文时，可以从头顺序 read。",
    ]
    missing_rules = [rule for rule in required_rules if rule not in system_prompt]
    assert missing_rules == [], "Agent 未收到以下定位规则：" + "；".join(missing_rules)


def test_system_prompt_answers_pure_directory_question_after_ls() -> None:
    """纯目录题在 ls 返回直接子项后应立即回答。"""

    agent_module = import_module("rag_core.agentic.agent")
    system_prompt = agent_module.KNOWLEDGE_AGENT_SYSTEM_PROMPT

    assert (
        "只有用户询问某个目录有哪些直接子项、有哪些文件或目录、或者路径是否存在时，才属于目录题"
        in system_prompt
    )
    assert (
        "ls 已返回目标目录所需的直接子项时，立即提交 answer_type=directory"
        in system_prompt
    )
    assert "目录题不需要调用 search 或 read" in system_prompt
    assert "自然语言中的“目录”不能直接拼进虚拟路径" in system_prompt


def test_system_prompt_uses_only_ls_for_pure_directory_questions() -> None:
    """纯目录题只能使用 ls，不应混入文件定位或语义检索工具。"""

    agent_module = import_module("rag_core.agentic.agent")
    system_prompt = agent_module.KNOWLEDGE_AGENT_SYSTEM_PROMPT

    assert (
        "纯目录题只能使用 ls，禁止调用 glob、search 或 read"
        in system_prompt
    )


def test_system_prompt_searches_when_directory_is_only_scope() -> None:
    """目录名只限定知识题范围时，不应从根目录逐层猜路径。"""

    agent_module = import_module("rag_core.agentic.agent")
    system_prompt = agent_module.KNOWLEDGE_AGENT_SYSTEM_PROMPT

    assert (
        "目录名只用于限定问题范围，而用户询问文件内容、属性或资源类型时，仍属于知识题"
        in system_prompt
    )
    assert (
        "完整虚拟路径未知时，禁止猜测路径或从根目录选择一个分支逐层试探"
        in system_prompt
    )
    assert "应先在根路径 / 使用 search 定位" in system_prompt
    assert "只使用工具返回的完整虚拟路径" in system_prompt


def test_system_prompt_keeps_glob_path_discovery_with_grep() -> None:
    """增加 grep 后仍应保留 glob 的路径发现和证据边界。"""

    agent_module = import_module("rag_core.agentic.agent")
    system_prompt = agent_module.KNOWLEDGE_AGENT_SYSTEM_PROMPT

    assert "你可以使用 ls、glob、search、read 和 grep 五个工具" in system_prompt
    assert (
        "完整虚拟路径未知但已知目录名或文件名时，应先使用 glob 定位"
        in system_prompt
    )
    assert (
        "glob 只用于定位文件路径，返回结果不能直接作为回答证据"
        in system_prompt
    )
    assert (
        "只能使用 glob 返回的完整虚拟路径继续调用 search 或 read"
        in system_prompt
    )
    assert "禁止猜测或自行拼接虚拟路径" in system_prompt


def test_prompt_v15_uses_structured_glob_arguments() -> None:
    """Prompt-V1.5 应要求 Agent 使用结构化 glob 参数。"""

    agent_module = import_module("rag_core.agentic.agent")
    system_prompt = agent_module.KNOWLEDGE_AGENT_SYSTEM_PROMPT

    assert (
        "glob 只接受 target、target_type 和可选的 path，不要传入 pattern"
        in system_prompt
    )
    assert (
        "问题表述为“某目录中的文件或资源”时，target_type 使用 directory"
        in system_prompt
    )
    assert "按文件名或标题定位时，target_type 使用 filename" in system_prompt
    assert "禁止先把目录名拼成 ls 路径" in system_prompt


def test_knowledge_agent_stops_searching_after_sufficient_read() -> None:
    """read 已提供充分证据后，Agent 应立即提交结构化回答。"""

    agent_module = import_module("rag_core.agentic.agent")
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


def test_prompt_v17_stops_out_of_scope_question_after_first_search() -> None:
    """V1.7 应声明超范围内容题一次 search 后立即拒答。"""

    prompts_module = import_module("rag_core.agentic.prompts")
    system_prompt = prompts_module.KNOWLEDGE_AGENT_PROMPTS["Prompt-V1.7"]

    assert "对于询问知识库是否包含某项内容的问题，只允许调用一次 search" in system_prompt
    assert (
        "首次 search 未找到足以支持回答的候选时，立即提交 answer_type=insufficient"
        in system_prompt
    )
    assert "此后禁止继续调用 search、glob、ls 或 read" in system_prompt


def test_knowledge_agent_uses_explicit_prompt_version() -> None:
    """当前 Agent Prompt 应具有可追踪的独立版本号。"""

    agent_module = import_module("rag_core.agentic.agent")

    assert agent_module.KNOWLEDGE_AGENT_PROMPT_VERSION == "Prompt-V1.9"


def test_prompt_version_log_records_problem_experiment_and_result() -> None:
    """Prompt 版本记录应保存问题、实验假设和真实结果。"""

    project_root = Path(__file__).resolve().parent.parent
    version_log = (project_root / "PROMPT_VERSIONS.md").read_text(
        encoding="utf-8"
    )

    assert "## Prompt-V1.2" in version_log
    assert "directory-001" in version_log
    assert "ambiguity-001" in version_log
    assert "## Prompt-V1.3" in version_log
    assert "已运行新基线验证" in version_log
    assert "回答类型准确率" in version_log
    assert "out-of-scope-001" in version_log
    assert "## Prompt-V1.4" in version_log
    assert "路径猜测" in version_log
    assert "## Prompt-V1.5" in version_log
    assert "结构化 glob 参数" in version_log
    assert "尚未运行 Prompt-V1.5 真实基线" in version_log
    assert "## Prompt-V1.6" in version_log
    assert "纯目录题只能使用 ls" in version_log
    assert "已运行 Prompt-V1.6 完整真实基线" in version_log
    assert "## Prompt-V1.7" in version_log
    assert "只允许调用一次 search" in version_log
    assert "Prompt-V1.7 真实冒烟失败" in version_log
    assert "当前生效版本已恢复为 Prompt-V1.6" in version_log


def test_build_knowledge_agent_limits_each_run_to_six_tool_calls(
    monkeypatch,
) -> None:
    """每次运行最多允许执行 6 次工具调用。"""

    agent_module = import_module("rag_core.agentic.agent")
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
    assert len(middleware) == 2

    tool_call_limiter = middleware[0]
    assert isinstance(tool_call_limiter, ToolCallLimitMiddleware)
    assert tool_call_limiter.run_limit == 6
    assert tool_call_limiter.exit_behavior == "end"

    retrieval_stop = middleware[1]
    assert isinstance(
        retrieval_stop,
        agent_module.KnowledgeRetrievalStopMiddleware,
    )


def test_build_knowledge_agent_does_not_count_grounded_answer_as_tool(
    monkeypatch,
) -> None:
    """结构化 GroundedAnswer 不应占用知识库工具调用次数。"""

    agent_module = import_module("rag_core.agentic.agent")
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


def test_knowledge_tool_call_limiter_counts_glob() -> None:
    """glob 应与 ls、search、read 一样占用工具调用次数。"""

    agent_module = import_module("rag_core.agentic.agent")
    tool_call_limiter = agent_module.KnowledgeToolCallLimitMiddleware(
        run_limit=6,
        exit_behavior="end",
    )

    assert tool_call_limiter._matches_tool_filter({"name": "glob"}) is True


def test_build_knowledge_agent_uses_in_memory_checkpointer(
    monkeypatch,
) -> None:
    """知识库 Agent 应使用内存保存器维护临时会话状态。"""

    agent_module = import_module("rag_core.agentic.agent")
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

    agent_module = import_module("rag_core.agentic.agent")
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

    agent_module = import_module("rag_core.agentic.agent")
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
    agent_module = import_module("rag_core.agentic.agent")
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
    agent_module = import_module("rag_core.agentic.agent")
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

    agent_module = import_module("rag_core.agentic.agent")
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


def test_extract_tool_traces_records_glob() -> None:
    """工具轨迹必须记录 glob 的参数和执行状态。"""

    agent_module = import_module("rag_core.agentic.agent")
    messages = [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "glob",
                    "args": {
                        "path": "/课程资源",
                        "pattern": "**/自动控制系统/*.md",
                    },
                    "id": "glob-call",
                    "type": "tool_call",
                }
            ],
        ),
        ToolMessage(
            content='{"matches": []}',
            tool_call_id="glob-call",
            status="success",
        ),
    ]

    result = agent_module.extract_tool_traces(messages)

    assert result == [
        {
            "step": 1,
            "tool_call_id": "glob-call",
            "name": "glob",
            "args": {
                "path": "/课程资源",
                "pattern": "**/自动控制系统/*.md",
            },
            "status": "success",
        }
    ]


def test_extract_tool_traces_marks_missing_tool_result() -> None:
    """找不到对应 ToolMessage 时应标记为 missing_result。"""

    agent_module = import_module("rag_core.agentic.agent")
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

    agent_module = import_module("rag_core.agentic.agent")
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

    agent_module = import_module("rag_core.agentic.agent")
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


def test_extract_token_usage_sums_all_ai_message_usage() -> None:
    """Agent token 统计应累加每一轮模型调用，并忽略非模型消息。"""

    agent_module = import_module("rag_core.agentic.agent")
    messages = [
        HumanMessage(content="气象站能做什么？"),
        AIMessage(
            content="",
            usage_metadata={
                "input_tokens": 100,
                "output_tokens": 20,
                "total_tokens": 120,
                "output_token_details": {"reasoning": 0},
            },
        ),
        ToolMessage(
            content='{"hits": []}',
            tool_call_id="search-call",
            status="success",
        ),
        AIMessage(
            content="根据知识库内容回答。",
            usage_metadata={
                "input_tokens": 80,
                "output_tokens": 15,
                "total_tokens": 95,
                "output_token_details": {"reasoning": 0},
            },
        ),
    ]

    result = agent_module.extract_token_usage(messages)

    assert result == {
        "input_tokens": 180,
        "output_tokens": 35,
        "total_tokens": 215,
        "output_token_details": {"reasoning": 0},
    }


def _search_tool_message(retrieval_status: str) -> ToolMessage:
    """构造一条 search 工具返回的 ToolMessage，内容为 JSON 格式。"""

    return ToolMessage(
        content=json.dumps(
            {
                "hits": [{"path": "/课程资源/示例.md", "score": 0.5}],
                "usage": "candidate_only",
                "retrieval_status": retrieval_status,
            },
            ensure_ascii=False,
        ),
        name="search",
        tool_call_id="search-call",
    )


def test_retrieval_stop_middleware_ends_agent_when_search_returns_none() -> None:
    """search 返回 none 时，应在模型再次生成前确定性结束 Agent。"""

    agent_module = import_module("rag_core.agentic.agent")
    middleware = agent_module.KnowledgeRetrievalStopMiddleware()
    state = {
        "messages": [
            _search_tool_message("none"),
        ]
    }

    result = middleware.before_model(state, None)

    assert result == {"jump_to": "end"}


def test_retrieval_stop_middleware_lets_relevant_search_continue() -> None:
    """search 返回 relevant 时，应放行模型继续生成。"""

    agent_module = import_module("rag_core.agentic.agent")
    middleware = agent_module.KnowledgeRetrievalStopMiddleware()
    state = {
        "messages": [
            _search_tool_message("relevant"),
        ]
    }

    assert middleware.before_model(state, None) is None


def test_retrieval_stop_middleware_lets_uncertain_search_continue() -> None:
    """search 返回 uncertain 时，应放行模型继续生成。"""

    agent_module = import_module("rag_core.agentic.agent")
    middleware = agent_module.KnowledgeRetrievalStopMiddleware()
    state = {
        "messages": [
            _search_tool_message("uncertain"),
        ]
    }

    assert middleware.before_model(state, None) is None


def test_retrieval_stop_middleware_ignores_non_search_tool_result() -> None:
    """最后一条是 read 等非 search 工具结果时，应放行。"""

    agent_module = import_module("rag_core.agentic.agent")
    middleware = agent_module.KnowledgeRetrievalStopMiddleware()
    state = {
        "messages": [
            ToolMessage(
                content="{\"path\": \"/课程资源/示例.md\", \"lines\": []}",
                name="read",
                tool_call_id="read-call",
            ),
        ]
    }

    assert middleware.before_model(state, None) is None


def test_retrieval_stop_middleware_ignores_ai_message() -> None:
    """最后一条是模型消息（没有新工具结果）时，应放行。"""

    agent_module = import_module("rag_core.agentic.agent")
    middleware = agent_module.KnowledgeRetrievalStopMiddleware()
    state = {
        "messages": [
            AIMessage(content="我需要先搜索知识库。"),
        ]
    }

    assert middleware.before_model(state, None) is None
