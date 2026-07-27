from pathlib import Path

import pytest

from app import knowledge_store


def test_list_knowledge_entries_returns_direct_children(
    tmp_path: Path,
) -> None:
    """应只返回指定目录的直接子项，并区分文件和目录。"""

    knowledge_root = tmp_path / "knowledge"
    course_directory = knowledge_root / "课程资源"
    course_directory.mkdir(parents=True)  #如果父目录不存在自动创建
    (course_directory / "智慧农场.md").write_text(
        "# 智慧农场\n",  #文件中的内容也是 ‘智慧农场’
        encoding="utf-8",
    )
    (knowledge_root / "说明.md").write_text(
        "# 知识库说明\n",
        encoding="utf-8",
    )

    result = knowledge_store.list_knowledge_entries("/", knowledge_root)

    assert result == [
        {"path": "/课程资源", "type": "directory"},
        {"path": "/说明.md", "type": "file"},
    ]


def test_list_knowledge_entries_explores_child_directory(
    tmp_path: Path,
) -> None:
    """应能从非根虚拟目录继续查看下一层。"""

    knowledge_root = tmp_path / "knowledge"
    course_directory = knowledge_root / "课程资源"
    course_directory.mkdir(parents=True)
    (course_directory / "智慧农场.md").write_text(
        "# 智慧农场\n",
        encoding="utf-8",
    )

    result = knowledge_store.list_knowledge_entries(
        "/课程资源",
        knowledge_root,
    )

    assert result == [
        {
            "path": "/课程资源/智慧农场.md",
            "type": "file",
        }
    ]


def test_list_knowledge_entries_returns_stable_order(
    tmp_path: Path,
) -> None:
    """创建顺序不同时也应稳定地先返回目录，再返回文件。"""

    knowledge_root = tmp_path / "knowledge"
    knowledge_root.mkdir()
    (knowledge_root / "B说明.md").write_text("", encoding="utf-8")
    (knowledge_root / "B目录").mkdir()
    (knowledge_root / "A说明.md").write_text("", encoding="utf-8")
    (knowledge_root / "A目录").mkdir()

    first_result = knowledge_store.list_knowledge_entries("/", knowledge_root)
    second_result = knowledge_store.list_knowledge_entries("/", knowledge_root)

    expected = [
        {"path": "/A目录", "type": "directory"},
        {"path": "/B目录", "type": "directory"},
        {"path": "/A说明.md", "type": "file"},
        {"path": "/B说明.md", "type": "file"},
    ]
    assert first_result == expected
    assert second_result == expected


def test_list_knowledge_entries_returns_empty_list_for_empty_directory(
    tmp_path: Path,
) -> None:
    """存在但没有子项的目录应返回空列表。"""

    knowledge_root = tmp_path / "knowledge"
    empty_directory = knowledge_root / "空目录"
    empty_directory.mkdir(parents=True)

    result = knowledge_store.list_knowledge_entries(
        "/空目录",
        knowledge_root,
    )

    assert result == []


def test_list_knowledge_entries_rejects_missing_directory(
    tmp_path: Path,
) -> None:
    """不存在的虚拟目录应抛出 FileNotFoundError。"""

    knowledge_root = tmp_path / "knowledge"
    knowledge_root.mkdir()

    with pytest.raises(FileNotFoundError):
        knowledge_store.list_knowledge_entries(
            "/不存在",
            knowledge_root,
        )


def test_list_knowledge_entries_rejects_file_path(
    tmp_path: Path,
) -> None:
    """把文件路径当作目录传入时应抛出 NotADirectoryError。"""

    knowledge_root = tmp_path / "knowledge"
    knowledge_root.mkdir()
    (knowledge_root / "说明.md").write_text(
        "# 知识库说明\n",
        encoding="utf-8",
    )

    with pytest.raises(NotADirectoryError):
        knowledge_store.list_knowledge_entries(
            "/说明.md",
            knowledge_root,
        )


def test_list_knowledge_entries_rejects_path_traversal(
    tmp_path: Path,
) -> None:
    """包含 .. 的虚拟路径不能进入目录查询。"""

    knowledge_root = tmp_path / "knowledge"
    knowledge_root.mkdir()

    with pytest.raises(ValueError):
        knowledge_store.list_knowledge_entries(
            "/课程资源/../外部目录",
            knowledge_root,
        )
