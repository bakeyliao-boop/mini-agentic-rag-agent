"""虚拟知识空间的路径处理工具。"""

from pathlib import Path, PurePosixPath

MAX_READ_CHARACTERS = 8_000


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
    """
    虚拟路径 -> 真实数据所在路径
    
    将虚拟路径映射为知识库根目录内的真实路径。
    映射后的路径可以不存在。执行包含关系检查前会解析已存在的
    符号链接，防止路径逃逸到知识库根目录之外。
    """

    normalized_path = normalize_virtual_path(virtual_path)

    if not isinstance(knowledge_root, Path):
        raise TypeError("knowledge root must be a Path")

    resolved_root = knowledge_root.resolve(strict=False)    #先处理知识库根目录 得到A/B
    relative_path = normalized_path[1:] #再去掉虚拟路径开头的 “/” 得到C/D.md 
    resolved_path = (resolved_root / relative_path).resolve(strict=False) #最后拼接 得到A/B/C/D.md

    if not resolved_path.is_relative_to(resolved_root):
        raise ValueError("virtual path resolves outside the knowledge root")

    return resolved_path


def glob_knowledge_paths(
    virtual_path: str,
    pattern: str,
    knowledge_root: Path,
) -> list[str]:
    """在虚拟目录范围内递归匹配文件，并返回完整虚拟路径。"""

    normalized_path = normalize_virtual_path(virtual_path)
    if not isinstance(pattern, str) or not pattern.strip():
        raise ValueError("glob pattern must not be empty")

    directory_path = resolve_knowledge_path(
        normalized_path,
        knowledge_root,
    )
    if not directory_path.exists():
        raise FileNotFoundError(
            f"knowledge directory does not exist: {normalized_path}"
        )
    if not directory_path.is_dir():
        raise NotADirectoryError(
            f"knowledge path is not a directory: {normalized_path}"
        )

    normalized_pattern = pattern.strip().lstrip("/")
    matched_paths: list[str] = []
    for source_path in sorted(
        directory_path.rglob("*"),  #递归查所有内容 recrusive glob
        key=lambda path: path.relative_to(directory_path).as_posix(),
    ):
        if not source_path.is_file():
            continue

        relative_path = source_path.relative_to(directory_path).as_posix()
        #当前文件相对于‘本次’搜索起点的路径 相对于本次directory_path的临时路径
        candidate_path = PurePosixPath(relative_path)
        matches_pattern = candidate_path.match(normalized_pattern)
        if (
            not matches_pattern
            and normalized_pattern.startswith("**/")
        ):
            zero_level_pattern = normalized_pattern.removeprefix("**/")
            matches_pattern = candidate_path.match(zero_level_pattern)

        if not matches_pattern:
            continue

        matched_paths.append(
            f"/{relative_path}"
            if normalized_path == "/"
            else f"{normalized_path}/{relative_path}"
        )

    return matched_paths


def read_markdown_lines(source_path: Path) -> list[tuple[int, str]]:
    """按稳定的 1起始 行号读取 UTF-8 Markdown 文件。

    保留空行和行内空白，但不保留每行末尾的换行符。
    """

    if not isinstance(source_path, Path):
        raise TypeError("source path must be a Path")
    if not source_path.exists():
        raise FileNotFoundError(f"Markdown file does not exist: {source_path}")
    if not source_path.is_file():  # 是否为文件
        raise IsADirectoryError(f"source path is not a file: {source_path}")
    if source_path.suffix.lower() != ".md":  # 是否为 Markdown 文件
        raise ValueError("source path must point to a Markdown file")

    with source_path.open(
        mode="r",
        encoding="utf-8",
        newline=None,
    ) as source_file:
        lines = source_file.read().splitlines()

    return list(enumerate(lines, start=1))


def list_knowledge_entries(
    virtual_path: str,
    knowledge_root: Path,
) -> list[dict[str, str]]:
    """
    只返回当前目录(virtual_path)的下一层子项，并区分文件和目录。
    列出指定虚拟目录的直接子项，并返回虚拟路径和类型。
    虚拟路径--真实路径--虚拟路径
    """

    normalized_path = normalize_virtual_path(virtual_path)
    directory_path = resolve_knowledge_path(normalized_path, knowledge_root)

    if not directory_path.exists():
        raise FileNotFoundError(f"knowledge directory does not exist: {virtual_path}")
    if not directory_path.is_dir():
        raise NotADirectoryError(f"knowledge path is not a directory: {virtual_path}")

    entries: list[dict[str, str]] = []
    for child_path in sorted(
        directory_path.iterdir(),
        key=lambda path: (not path.is_dir(), path.name),    #先排目录，再排文件
    ):
        child_virtual_path = (
            f"/{child_path.name}"
            if normalized_path == "/"
            else f"{normalized_path}/{child_path.name}"
        )
        entry_type = "directory" if child_path.is_dir() else "file"
        entries.append(
            {
                "path": child_virtual_path,
                "type": entry_type,
            }
        )

    return entries


def read_knowledge_page(
    virtual_path: str,
    knowledge_root: Path,
    start_line: int = 1,
    limit: int = 80,
) -> dict[str, object]:
    """按起始行和行数读取一页 Markdown 内容。"""

    if start_line < 1:
        raise ValueError("start_line must be greater than or equal to 1")
    if limit < 1:
        raise ValueError("limit must be greater than or equal to 1")
    if limit > 80:
        raise ValueError("limit must be less than or equal to 80")

    normalized_path = normalize_virtual_path(virtual_path)
    source_path = resolve_knowledge_path(normalized_path, knowledge_root)

    all_lines = read_markdown_lines(source_path)
    if all_lines and start_line > len(all_lines):
        raise ValueError("start_line must not exceed the file line count")

    start_index = start_line - 1
    end_index = start_index + limit
    candidate_lines = all_lines[start_index:end_index]
    page_lines = []
    character_count = 0

    for line_number, text in candidate_lines:
        if len(text) > MAX_READ_CHARACTERS:
            raise ValueError(
                "a single Markdown line exceeds the character limit"
            )

        if character_count + len(text) > MAX_READ_CHARACTERS:
            break

        page_lines.append((line_number, text))
        character_count += len(text)

    next_index = start_index + len(page_lines)
    next_line = next_index + 1 if next_index < len(all_lines) else None

    return {
        "path": normalized_path,
        "lines": [
            {
                "line": line_number,
                "text": text,
            }
            for line_number, text in page_lines
        ],
        "next_line": next_line,
    }
