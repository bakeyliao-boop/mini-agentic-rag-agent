from importlib import import_module

import pytest
from langchain_core.documents import Document
from pydantic import ValidationError


def test_build_knowledge_tools_returns_expected_names(tmp_path) -> None:
    """生成的知识库工具名称应固定为 ls、search、read。"""

    knowledge_tools = import_module("app.tools")
    build_knowledge_tools = getattr(
        knowledge_tools,
        "build_knowledge_tools",
        None,
    )
    assert build_knowledge_tools is not None, "build_knowledge_tools 尚未实现"

    generated_tools = build_knowledge_tools(
        knowledge_root=tmp_path / "knowledge",
        vector_store=object(),
    )

    assert [tool.name for tool in generated_tools] == ["ls", "search", "read"]


def test_ls_tool_lists_direct_children(tmp_path) -> None:
    """ls 工具应返回指定目录的直接子项。"""

    knowledge_root = tmp_path / "knowledge"
    course_directory = knowledge_root / "课程资源"
    course_directory.mkdir(parents=True)
    (knowledge_root / "说明.md").write_text(
        "# 知识库说明\n",
        encoding="utf-8",
    )
    (course_directory / "智慧农场.md").write_text(
        "# 智慧农场\n",
        encoding="utf-8",
    )

    knowledge_tools = import_module("app.tools")
    generated_tools = knowledge_tools.build_knowledge_tools(
        knowledge_root=knowledge_root,
        vector_store=object(),
    )
    ls_tool = generated_tools[0]

    result = ls_tool.invoke({"path": "/"})

    assert result == {
        "path": "/",
        "entries": [
            {"path": "/课程资源", "type": "directory"},
            {"path": "/说明.md", "type": "file"},
        ],
    }


def test_ls_tool_rejects_parent_path_traversal(tmp_path) -> None:
    """ls 工具应拒绝使用 .. 访问知识库外部。"""

    knowledge_tools = import_module("app.tools")
    generated_tools = knowledge_tools.build_knowledge_tools(
        knowledge_root=tmp_path / "knowledge",
        vector_store=object(),
    )
    ls_tool = generated_tools[0]

    with pytest.raises(
        ValueError,
        match=r"virtual path must not contain '\.' or '\.\.' segments",
    ):
        ls_tool.invoke({"path": "/../secret"})


def test_search_tool_returns_candidate_hits(tmp_path) -> None:
    """search 工具应返回只能用于定位的候选内容。"""

    class FakeVectorStore:
        def similarity_search_with_relevance_scores(
            self,
            query: str,
            k: int,
            filter,
        ):
            assert query == "自动灌溉"
            assert k == 2
            assert filter is None
            return [
                (
                    Document(
                        page_content="系统可以根据环境数据控制灌溉。",
                        metadata={
                            "path": "/课程资源/智慧农场.md",
                            "start_line": 4,
                            "end_line": 4,
                        },
                    ),
                    0.9,
                )
            ]

    knowledge_tools = import_module("app.tools")
    generated_tools = knowledge_tools.build_knowledge_tools(
        knowledge_root=tmp_path / "knowledge",
        vector_store=FakeVectorStore(),
    )
    search_tool = generated_tools[1]

    result = search_tool.invoke(
        {
            "query": "自动灌溉",
            "path": "/",
            "limit": 2,
        }
    )

    assert result == {
        "hits": [
            {
                "path": "/课程资源/智慧农场.md",
                "start_line": 4,
                "end_line": 4,
                "score": 0.9,
                "preview": "系统可以根据环境数据控制灌溉。",
            }
        ],
        "usage": "candidate_only",
    }


def test_search_tool_rejects_limit_greater_than_five(tmp_path) -> None:
    """search 工具不应允许一次返回超过 5 个候选。"""

    knowledge_tools = import_module("app.tools")
    generated_tools = knowledge_tools.build_knowledge_tools(
        knowledge_root=tmp_path / "knowledge",
        vector_store=object(),
    )
    search_tool = generated_tools[1]

    with pytest.raises(
        ValidationError,
        match="Input should be less than or equal to 5",
    ):
        search_tool.invoke(
            {
                "query": "自动灌溉",
                "path": "/",
                "limit": 6,
            }
        )


def test_search_tool_schema_exposes_limit_range(tmp_path) -> None:
    """search Schema 应明确告诉模型 limit 只能位于 1 到 5。"""

    knowledge_tools = import_module("app.tools")
    generated_tools = knowledge_tools.build_knowledge_tools(
        knowledge_root=tmp_path / "knowledge",
        vector_store=object(),
    )
    search_tool = generated_tools[1]

    schema = search_tool.args_schema.model_json_schema()
    limit_schema = schema["properties"]["limit"]

    assert limit_schema["minimum"] == 1
    assert limit_schema["maximum"] == 5


def test_read_tool_returns_requested_markdown_page(tmp_path) -> None:
    """read 工具应从指定位置读取一页 Markdown 原文。"""

    knowledge_root = tmp_path / "knowledge"
    source_path = knowledge_root / "课程资源" / "智慧农场.md"
    source_path.parent.mkdir(parents=True)
    source_path.write_text(
        "# 智慧农场\n\n气象站采集环境数据。\n自动灌溉系统控制水泵。\n",
        encoding="utf-8",
    )

    knowledge_tools = import_module("app.tools")
    generated_tools = knowledge_tools.build_knowledge_tools(
        knowledge_root=knowledge_root,
        vector_store=object(),
    )
    read_tool = generated_tools[2]

    result = read_tool.invoke(
        {
            "path": "/课程资源/智慧农场.md",
            "start_line": 3,
            "limit": 1,
        }
    )

    assert result == {
        "path": "/课程资源/智慧农场.md",
        "lines": [
            {
                "line": 3,
                "text": "气象站采集环境数据。",
            }
        ],
        "next_line": 4,
    }


def test_read_tool_rejects_limit_greater_than_eighty(tmp_path) -> None:
    """read 工具不应允许一次读取超过 80 行。"""

    knowledge_tools = import_module("app.tools")
    generated_tools = knowledge_tools.build_knowledge_tools(
        knowledge_root=tmp_path / "knowledge",
        vector_store=object(),
    )
    read_tool = generated_tools[2]

    with pytest.raises(
        ValueError,
        match="limit must be less than or equal to 80",
    ):
        read_tool.invoke(
            {
                "path": "/课程资源/智慧农场.md",
                "start_line": 1,
                "limit": 81,
            }
        )
