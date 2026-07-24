from pathlib import Path

import pytest

from app.knowledge_store import (
    normalize_virtual_path,
    resolve_knowledge_path,
)


@pytest.mark.parametrize(
    ("raw_path", "expected"),
    [
        ("/", "/"),
        ("/课程资源/小学", "/课程资源/小学"),
        ("/课程资源/小学/", "/课程资源/小学"),
        ("/.hidden", "/.hidden"),
        ("/a..b", "/a..b"),
    ],
)
def test_normalize_virtual_path_returns_canonical_path(
    raw_path: str,
    expected: str,
) -> None:
    assert normalize_virtual_path(raw_path) == expected


@pytest.mark.parametrize(
    "invalid_path",
    [
        "",
        "课程资源/小学",
        "//课程资源",
        "/课程资源//小学",
        "/课程资源/./小学",
        "/课程资源/../小学",
        "/..",
        "\\\\server\\share",
        "/课程资源\\小学",
        "C:/Users/hyf",
        "C:\\Users\\hyf",
        "/C:/Users/hyf",
        "/C:secret",
        "/课程资源/\x00小学",
    ],
)
def test_normalize_virtual_path_rejects_invalid_strings(
    invalid_path: str,
) -> None:
    with pytest.raises(ValueError):
        normalize_virtual_path(invalid_path)


@pytest.mark.parametrize(
    "invalid_path",
    [None, 123, b"/knowledge", Path("/课程资源")],
)
def test_normalize_virtual_path_rejects_non_strings(
    invalid_path: object,
) -> None:
    with pytest.raises(TypeError):
        normalize_virtual_path(invalid_path)  # type: ignore[arg-type]


def test_resolve_knowledge_root(tmp_path: Path) -> None:
    knowledge_root = tmp_path / "knowledge"
    knowledge_root.mkdir()

    result = resolve_knowledge_path("/", knowledge_root)

    assert result == knowledge_root.resolve()


def test_resolve_knowledge_child_path(tmp_path: Path) -> None:
    knowledge_root = tmp_path / "knowledge"
    expected_path = knowledge_root / "课程资源" / "小学"
    expected_path.mkdir(parents=True)

    result = resolve_knowledge_path("/课程资源/小学", knowledge_root)

    assert result == expected_path.resolve()


def test_resolve_nonexistent_knowledge_path(tmp_path: Path) -> None:
    knowledge_root = tmp_path / "knowledge"
    knowledge_root.mkdir()
    expected_path = knowledge_root / "课程资源" / "不存在.md"
    assert not expected_path.exists()   #文件确实不存在

    result = resolve_knowledge_path("/课程资源/不存在.md", knowledge_root)

    assert result == expected_path.resolve(strict=False)
    assert not result.exists()


def test_resolve_rejects_invalid_virtual_path(tmp_path: Path) -> None:
    """
    非法虚拟路径不能进入映射流程
    """
    knowledge_root = tmp_path / "knowledge"
    knowledge_root.mkdir()

    with pytest.raises(ValueError):
        resolve_knowledge_path("/课程资源/../secret.md", knowledge_root)


def test_resolve_rejects_non_path_knowledge_root() -> None:
    with pytest.raises(TypeError):
        resolve_knowledge_path("/", "./knowledge")  # type: ignore[arg-type]


def test_resolve_rejects_symlink_escape(tmp_path: Path) -> None:
    """知识库内的符号链接不能指向根目录外部。"""

    knowledge_root = tmp_path / "knowledge"
    outside_root = tmp_path / "outside"
    knowledge_root.mkdir()
    outside_root.mkdir()
    (knowledge_root / "escape").symlink_to(
        outside_root,
        target_is_directory=True,
    )

    with pytest.raises(ValueError):
        resolve_knowledge_path("/escape/secret.md", knowledge_root)
