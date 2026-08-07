from pathlib import Path

from app import knowledge_store


def test_glob_knowledge_paths_matches_nested_markdown_files(
    tmp_path: Path,
) -> None:
    """应递归匹配指定目录模式，并稳定返回完整虚拟路径。"""

    knowledge_root = tmp_path / "knowledge"
    target_directory = (
        knowledge_root / "课程资源" / "自动控制系统"
    )
    target_directory.mkdir(parents=True)
    (target_directory / "课程介绍.md").write_text(
        "# 课程介绍\n",
        encoding="utf-8",
    )
    (target_directory / "智慧农场.md").write_text(
        "# 智慧农场\n",
        encoding="utf-8",
    )

    result = knowledge_store.glob_knowledge_paths(
        "/",
        "**/自动控制系统/*.md", #*.md 只匹配“自动控制系统”目录的直接文件
#         **/          自动控制系统/          *.md
#       任意层级目录   名字必须完全匹配        直接 Markdown 文件
        knowledge_root,
    )

    assert result == [
        "/课程资源/自动控制系统/智慧农场.md",
        "/课程资源/自动控制系统/课程介绍.md",
    ]


def test_glob_knowledge_paths_stays_inside_virtual_subdirectory(
    tmp_path: Path,
) -> None:
    """从子目录搜索时不应越界，并仍应返回完整虚拟路径。"""

    knowledge_root = tmp_path / "knowledge"
    target_directory = (
        knowledge_root / "课程资源" / "自动控制系统"
    )
    target_directory.mkdir(parents=True)
    (target_directory / "智慧农场.md").write_text(
        "# 智慧农场\n",
        encoding="utf-8",
    )

    outside_directory = (
        knowledge_root / "其他资源" / "自动控制系统"
    )
    outside_directory.mkdir(parents=True)
    (outside_directory / "外部文件.md").write_text(
        "# 外部文件\n",
        encoding="utf-8",
    )

    result = knowledge_store.glob_knowledge_paths(
        "/课程资源",
        "**/自动控制系统/*.md",
        knowledge_root,
    )

    assert result == [
        "/课程资源/自动控制系统/智慧农场.md",
    ]
# C:\Users\hyf\Desktop\mini-agentic-rag-agent\knowledge\课程资源\自动控制系统\课程介绍.md || 智慧农场.md
