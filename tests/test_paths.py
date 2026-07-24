from pathlib import Path

import pytest

from app.knowledge_store import normalize_virtual_path


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
