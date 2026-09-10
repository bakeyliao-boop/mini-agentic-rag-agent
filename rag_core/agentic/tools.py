"""供知识库 Agent 调用的 LangChain 工具封装。"""

from pathlib import Path, PurePosixPath
from typing import Annotated, Literal

from langchain_core.tools import BaseTool, StructuredTool, ToolException
from pydantic import AfterValidator, Field, StringConstraints

from rag_core.agentic.evidence import EvidenceRegistry
from rag_core.knowledge.indexer import search_chroma_index
from rag_core.knowledge.store import (
    glob_knowledge_paths,
    glob_knowledge_snapshot_paths,
    grep_knowledge_file,
    list_knowledge_entries,
    list_knowledge_snapshot_entries,
    normalize_virtual_path,
    read_knowledge_page,
)


def _validate_glob_target(target: str) -> str:
    """拒绝具有路径语义的 glob 目标名称。"""

    if target in {".", ".."}:
        raise ValueError("glob target must not be '.' or '..'")
    return target


def build_knowledge_tools(
    knowledge_root: Path,
    vector_store: object,
    evidence_registry: EvidenceRegistry,
    path_snapshot: list[dict[str, str]] | None = None,
) -> list[BaseTool]:
    """生成供 Agent 调用的 ls、glob、search、read 和 grep 工具。"""

    def ls_tool(path: str = "/") -> dict[str, object]:
        """列出指定虚拟目录的下一层文件和目录。"""

        normalized_path = normalize_virtual_path(path)
        try:
            if path_snapshot is None:
                entries = list_knowledge_entries(
                    normalized_path,
                    knowledge_root,
                )
            else:
                entries = list_knowledge_snapshot_entries(
                    normalized_path,
                    path_snapshot,
                )
        except (FileNotFoundError, NotADirectoryError) as error:
            raise ToolException(str(error)) from error

        return {
            "path": normalized_path,
            "entries": entries,
        }

    def glob_tool(
        target: Annotated[
            str,
            StringConstraints(
                strip_whitespace=True,
                min_length=1,
                pattern=r"^[^/\\*?\[\]]+$",
            ),
            AfterValidator(_validate_glob_target),
            Field(description="要查找的目录名或文件名关键词。"),
        ],
        target_type: Annotated[
            Literal["directory", "filename"],
            Field(description="目标是目录名还是文件名。"),
        ],
        path: str = "/",
    ) -> dict[str, object]:
        """根据目标名称和类型，在指定虚拟目录范围内递归查找文件。"""

        normalized_path = normalize_virtual_path(path)
        normalized_target = target.strip()
        if target_type == "directory":
            pattern = f"**/{normalized_target}/*.md"
        elif normalized_target.casefold().endswith(".md"):
            pattern = f"**/*{normalized_target}"
        else:
            pattern = f"**/*{normalized_target}*.md"
        try:
            if path_snapshot is None:
                matched_paths = glob_knowledge_paths(
                    virtual_path=normalized_path,
                    pattern=pattern,
                    knowledge_root=knowledge_root,
                )
            else:
                matched_paths = glob_knowledge_snapshot_paths(
                    virtual_path=normalized_path,
                    pattern=pattern,
                    snapshot=path_snapshot,
                )
        except (ValueError, FileNotFoundError, NotADirectoryError) as error:
            raise ToolException(str(error)) from error

        return {
            "matches": [
                {
                    "path": matched_path,
                    "name": PurePosixPath(matched_path).name,
                }
                for matched_path in matched_paths
            ],
        }

    def search_tool(
        query: str,
        path: str = "/",
        limit: Annotated[int, Field(ge=1, le=5)] = 5,
    ) -> dict[str, object]:
        """在指定虚拟路径范围内搜索候选内容。"""

        return search_chroma_index(
            vector_store=vector_store,
            query=query,
            path=path,
            limit=limit,
        )

    def read_tool(
        path: str,
        start_line: int = 1,
        limit: Annotated[int, Field(ge=1, le=80)] = 80,
    ) -> dict[str, object]:
        """按行读取指定 Markdown 文件的原文。"""

        read_result = read_knowledge_page(
            virtual_path=path,
            knowledge_root=knowledge_root,
            start_line=start_line,
            limit=limit,    #完整文件没结束+字符没超限 就一直读
        )
        evidences = evidence_registry.register_read_page(read_result)
        evidence_ids_by_line = {
            evidence.start_line: evidence.evidence_id
            for evidence in evidences
        }
        lines = read_result["lines"]
        if not isinstance(lines, list):
            raise ValueError("read result lines must be a list")

        return {
            "path": read_result["path"],
            "lines": [
                {
                    **line,
                    **(
                        {"evidence_id": evidence_ids_by_line[line["line"]]}
                        if line["line"] in evidence_ids_by_line
                        else {}
                    ),
                }
                for line in lines
            ],
            "next_line": read_result["next_line"],
        }

    def grep_tool(
        path: Annotated[str, Field(description="已知的单个 Markdown 完整虚拟路径，不接受目录。")],
        patterns: Annotated[
            list[Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=200)]],
            Field(min_length=1, max_length=10, description="原文词语列表，任意一个命中即可；字面匹配，不是正则或语义检索。"),
        ],
        limit: Annotated[int, Field(ge=1, le=20)] = 10,
    ) -> dict[str, object]:
        """定位单文件内的字面关键词；不登记证据，也不因空结果直接结束 Agent。"""
        try:
            return grep_knowledge_file(path, knowledge_root, patterns, limit)
        except (ValueError, OSError) as error:
            raise ToolException(str(error)) from error

    return [
        StructuredTool.from_function(
            func=ls_tool,
            name="ls",
            description="列出一个虚拟目录的直接子项，不递归读取正文。",
            handle_tool_error=True,
        ),
        StructuredTool.from_function(
            func=glob_tool,
            name="glob",
            description=(
                "按目录名或文件名递归查找 Markdown 文件；"
                "只用于定位文件，不读取正文。"
            ),
            handle_tool_error=True,
        ),
        StructuredTool.from_function(
            func=search_tool,
            name="search",
            description="搜索候选内容；结果只能用于定位，不能直接作为证据。",
        ),
        StructuredTool.from_function(
            func=read_tool,
            name="read",
            description="按行读取 Markdown 原文，供回答前核实知识。",
        ),
        StructuredTool.from_function(
            func=grep_tool,
            name="grep",
            description=(
                "在已知的单个 Markdown 文件中查找原文关键词，返回 path、1-based line 和命中文本。"
                "多个 patterns 按 OR 字面匹配，不走向量库；可命中目录或正文，需要判断读取位置。"
                "truncated 表示仍有命中未返回或长行片段已缩短。结果只用于定位，回答前须 read 核实；"
                "空结果只表示这些词未匹配，不代表没有相关内容。"
            ),
            handle_tool_error=True,
            handle_validation_error=True,
        ),
    ]
