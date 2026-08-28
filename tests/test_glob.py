from pathlib import Path

from rag_core.knowledge import store as knowledge_store


def test_build_knowledge_path_snapshot_records_complete_tree(
    tmp_path: Path,
) -> None:
    """路径快照应同时记录知识库中的目录和 Markdown 文件。"""

    knowledge_root = tmp_path / "knowledge"
    digital_directory = (
        knowledge_root / "课程资源" / "小学" / "四年级" / "信息科技"
    )
    digital_directory.mkdir(parents=True)
    (digital_directory / "初探数字化.md").write_text(
        "# 初探数字化\n",
        encoding="utf-8",
    )

    result = knowledge_store.build_knowledge_path_snapshot(knowledge_root)

    assert result == [
        {"path": "/", "type": "directory"},
        {"path": "/课程资源", "type": "directory"},
        {"path": "/课程资源/小学", "type": "directory"},
        {"path": "/课程资源/小学/四年级", "type": "directory"},
        {
            "path": "/课程资源/小学/四年级/信息科技",
            "type": "directory",
        },
        {
            "path": "/课程资源/小学/四年级/信息科技/初探数字化.md",
            "type": "file",
        },
    ]


def test_glob_knowledge_snapshot_paths_does_not_access_disk(
    monkeypatch,
) -> None:
    """内存快照匹配应返回完整路径，并且不再扫描真实磁盘。"""

    snapshot = [
        {"path": "/", "type": "directory"},
        {"path": "/课程资源", "type": "directory"},
        {"path": "/课程资源/小学", "type": "directory"},
        {
            "path": "/课程资源/小学/四年级/信息科技/初探数字化.md",
            "type": "file",
        },
        {
            "path": "/其他资源/初探数字化.md",
            "type": "file",
        },
    ]

    def reject_disk_scan(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("snapshot glob must not scan the filesystem")

    monkeypatch.setattr(Path, "rglob", reject_disk_scan)

    result = knowledge_store.glob_knowledge_snapshot_paths(
        "/课程资源",
        "**/*初探数字化*.md",
        snapshot,
    )

    assert result == [
        "/课程资源/小学/四年级/信息科技/初探数字化.md",
    ]


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
