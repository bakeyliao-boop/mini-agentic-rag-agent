"""虚拟知识空间的路径处理工具。"""

from pathlib import Path


def normalize_virtual_path(path: str) -> str:
    """校验虚拟 POSIX 路径并返回规范形式。

    本函数只处理路径语法，不访问文件系统，不检查路径是否存在，
    也不将路径映射到 ``KNOWLEDGE_ROOT``。
    """

    if not isinstance(path, str):
        raise TypeError("virtual path must be a string")
    if not path:
        raise ValueError("virtual path must not be empty")
    if "\x00" in path:
        raise ValueError("virtual path must not contain a NUL character")
    if "\\" in path:
        raise ValueError("virtual path must use forward slashes")
    if not path.startswith("/"):
        raise ValueError("virtual path must start with '/'")
    if "//" in path:
        raise ValueError("virtual path must not contain repeated slashes")

    canonical_path = path[:-1] if path != "/" and path.endswith("/") else path
    parts = canonical_path.split("/")[1:]

    if any(part in {".", ".."} for part in parts):
        raise ValueError("virtual path must not contain '.' or '..' segments")

    first_part = parts[0] if parts else ""
    if (
        len(first_part) >= 2
        and first_part[0].isascii()
        and first_part[0].isalpha()
        and first_part[1] == ":"
    ):
        raise ValueError("virtual path must not contain a Windows drive")

    return canonical_path


def resolve_knowledge_path(
    virtual_path: str,
    knowledge_root: Path,
) -> Path:  #虚拟路径，真实知识库目录
    """将虚拟路径映射为知识库根目录内的真实路径。

    映射后的路径可以不存在。执行包含关系检查前会解析已存在的
    符号链接，防止路径逃逸到知识库根目录之外。
    """

    normalized_path = normalize_virtual_path(virtual_path)

    if not isinstance(knowledge_root, Path):
        raise TypeError("knowledge root must be a Path")

    resolved_root = knowledge_root.resolve(strict=False)
    relative_path = normalized_path[1:]
    resolved_path = (resolved_root / relative_path).resolve(strict=False)

    if not resolved_path.is_relative_to(resolved_root):
        raise ValueError("virtual path resolves outside the knowledge root")

    return resolved_path
