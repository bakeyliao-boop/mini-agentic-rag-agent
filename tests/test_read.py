from pathlib import Path

import pytest

from app.knowledge_store import read_markdown_lines


def test_read_markdown_lines_preserves_one_based_line_numbers(
    tmp_path: Path,
) -> None:
    """中文 Markdown 的空行和 1-based 行号应保持稳定。"""

    source_path = tmp_path / "智慧农场.md"
    source_path.write_text(
        "# 智慧农场\n\n气象站可以采集环境数据。\n",
        encoding="utf-8",
        newline="\n",
    )

    result = read_markdown_lines(source_path)

    assert result == [
        (1, "# 智慧农场"),
        (2, ""),
        (3, "气象站可以采集环境数据。"),
    ]


def test_read_markdown_lines_returns_same_result_when_repeated(
    tmp_path: Path,
) -> None:
    """同一个 Markdown 文件重复读取时应返回完全相同的结果。"""

    source_path = tmp_path / "课程介绍.md"
    source_path.write_text(
        "# 课程介绍\n\n学习自动控制系统。\n",
        encoding="utf-8",
        newline="\n",
    )

    first_result = read_markdown_lines(source_path)
    second_result = read_markdown_lines(source_path)

    assert second_result == first_result


def test_read_markdown_lines_returns_empty_list_for_empty_file(
    tmp_path: Path,
) -> None:
    """内容为空的 Markdown 文件应返回空列表。"""

    source_path = tmp_path / "空文件.md"
    source_path.write_text(
        "",
        encoding="utf-8",
        newline="\n",
    )

    result = read_markdown_lines(source_path)

    assert result == []


def test_read_markdown_lines_rejects_missing_file(
    tmp_path: Path,
) -> None:
    """不存在的 Markdown 文件应抛出 FileNotFoundError。"""

    source_path = tmp_path / "不存在.md"
    assert not source_path.exists()

    with pytest.raises(FileNotFoundError):
        read_markdown_lines(source_path)


def test_read_markdown_lines_rejects_directory(
    tmp_path: Path,
) -> None:
    """传入目录时应抛出 IsADirectoryError。"""

    source_path = tmp_path / "课程资源"
    source_path.mkdir()

    with pytest.raises(IsADirectoryError):
        read_markdown_lines(source_path)


def test_read_markdown_lines_rejects_non_markdown_file(
    tmp_path: Path,
) -> None:
    """存在但不是 Markdown 的文件应抛出 ValueError。"""

    source_path = tmp_path / "课程介绍.txt"
    source_path.write_text(
        "这不是 Markdown 文件。\n",
        encoding="utf-8",
        newline="\n",
    )
    assert source_path.exists()

    with pytest.raises(ValueError):
        read_markdown_lines(source_path)


def test_read_markdown_lines_rejects_non_path_source() -> None:
    """source_path 不是 Path 对象时应抛出 TypeError。"""

    with pytest.raises(TypeError):
        read_markdown_lines("./课程介绍.md")  # type: ignore[arg-type]


def test_read_markdown_lines_rejects_invalid_utf8(
    tmp_path: Path,
) -> None:
    """内容不是合法 UTF-8 时应抛出 UnicodeDecodeError。"""

    source_path = tmp_path / "错误编码.md"
    source_path.write_bytes(b"\xff\xfe\xfa")

    with pytest.raises(UnicodeDecodeError):
        read_markdown_lines(source_path)
