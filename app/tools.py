from pathlib import Path
from typing import Annotated

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import Field

from app.evidence import EvidenceRegistry
from app.indexer import search_chroma_index
from app.knowledge_store import (
    list_knowledge_entries,
    normalize_virtual_path,
    read_knowledge_page,
)


def build_knowledge_tools(
    knowledge_root: Path,
    vector_store: object,
    evidence_registry: EvidenceRegistry,
) -> list[BaseTool]:
    """生成供 Agent 调用的 ls、search 和 read 工具。"""

    def ls_tool(path: str = "/") -> dict[str, object]:
        """列出指定虚拟目录的下一层文件和目录。"""

        normalized_path = normalize_virtual_path(path)
        return {
            "path": normalized_path,
            "entries": list_knowledge_entries(normalized_path, knowledge_root), #返回下一层路径和类型
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
        limit: int = 80,
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

    return [
        StructuredTool.from_function(
            func=ls_tool,
            name="ls",
            description="列出一个虚拟目录的直接子项，不递归读取正文。",
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
    ]
