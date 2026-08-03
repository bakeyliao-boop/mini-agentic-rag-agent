from app import indexer
from app.models import Chunk


def test_chunk_markdown_lines_returns_single_chunk_for_short_document() -> None:
    """较短的 Markdown 应生成一个保留原文位置的 Chunk。"""

    lines = [
        (1, "# 智慧农场"),
        (2, ""),
        (3, "自动灌溉系统可以根据不同的农作物进行分类灌溉。"),
    ]

    result = indexer.chunk_markdown_lines(
        "/课程资源/智慧农场.md",
        lines,
    )

    assert result == [
        Chunk(
            chunk_id="/课程资源/智慧农场.md#L1-L3",
            path="/课程资源/智慧农场.md",
            start_line=1,
            end_line=3,
            text=(
                "# 智慧农场\n"
                "\n"
                "自动灌溉系统可以根据不同的农作物进行分类灌溉。"
            ),
        )
    ]


def test_chunk_markdown_lines_returns_empty_list_for_empty_document() -> None:
    """空 Markdown 不应生成 Chunk。"""

    result = indexer.chunk_markdown_lines(
        "/课程资源/空文件.md",
        [],
    )

    assert result == []


def test_chunk_markdown_lines_splits_long_document_with_overlap() -> None:
    """长文档应按完整段落切块，并让相邻 Chunk 重叠一个段落。"""

    irrigation = "自动灌溉系统可以根据农作物进行分类灌溉。" * 15
    weather = "气象站可以实时监测天气并预测农业灾害。" * 15
    fertilizer = "计算机可以自动控制肥料的种类和用量。" * 15
    lines = [
        (1, "# 智慧农场"),
        (2, ""),
        (3, irrigation),
        (4, ""),
        (5, weather),
        (6, ""),
        (7, fertilizer),
    ]

    result = indexer.chunk_markdown_lines(
        "/课程资源/智慧农场.md",
        lines,
    )

    assert result == [
        Chunk(
            chunk_id="/课程资源/智慧农场.md#L1-L5",
            path="/课程资源/智慧农场.md",
            start_line=1,
            end_line=5,
            text=f"# 智慧农场\n\n{irrigation}\n\n{weather}",
        ),
        Chunk(
            chunk_id="/课程资源/智慧农场.md#L5-L7",
            path="/课程资源/智慧农场.md",
            start_line=5,
            end_line=7,
            text=f"{weather}\n\n{fertilizer}",
        ),
    ]
    assert all(len(chunk.text) <= 800 for chunk in result)
