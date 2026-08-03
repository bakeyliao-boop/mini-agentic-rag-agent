"""Markdown 切块与本地索引。"""

from pathlib import Path

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_chroma import Chroma

from app.models import Chunk

MAX_CHUNK_CHARACTERS = 800
CHUNK_OVERLAP_PARAGRAPHS = 1


def _find_paragraph_ranges(
    lines: list[tuple[int, str]],
) -> list[tuple[int, int]]:
    """
    返回每个非空段落在 lines 中的起止索引。
    """

    ranges: list[tuple[int, int]] = []
    start_index: int | None = None

    for index, (_, text) in enumerate(lines):
        if text.strip():
            if start_index is None:
                start_index = index
        elif start_index is not None:   #如果有起点并且当前行是空行，则结束段落
            ranges.append((start_index, index - 1))
            start_index = None

    if start_index is not None:
        ranges.append((start_index, len(lines) - 1))

    return ranges


def _build_chunk(
    path: str,
    lines: list[tuple[int, str]],
    start_index: int,
    end_index: int,
) -> Chunk:
    """根据原文索引范围创建一个 Chunk。"""

    selected_lines = lines[start_index : end_index + 1]
    start_line = selected_lines[0][0]
    end_line = selected_lines[-1][0]
    text = "\n".join(line_text for _, line_text in selected_lines)

    return Chunk(
        chunk_id=f"{path}#L{start_line}-L{end_line}",
        path=path,
        start_line=start_line,
        end_line=end_line,
        text=text,
    )


def _range_text_length(
    lines: list[tuple[int, str]],
    start_index: int,
    end_index: int,
) -> int:
    """计算原文索引范围重新连接后的字符数。"""

    return len(
        "\n".join(
            line_text
            for _, line_text in lines[start_index : end_index + 1]
        )
    )


def chunk_to_document(chunk: Chunk) -> Document:
    """将 Chunk 转为可写入向量索引的 LangChain Document。"""

    return Document(
        page_content=chunk.text,
        metadata={
            "chunk_id": chunk.chunk_id,
            "path": chunk.path,
            "start_line": chunk.start_line,
            "end_line": chunk.end_line,
        },
    )


def build_chroma_index(
    chunks: list[Chunk],
    persist_directory: Path,
    embedding: Embeddings,
) -> Chroma:
    """将 Chunk 写入指定目录中的本地 Chroma 索引。"""

    documents = [chunk_to_document(chunk) for chunk in chunks]
    ids = [chunk.chunk_id for chunk in chunks]

    return Chroma.from_documents(
        documents=documents,
        embedding=embedding,
        ids=ids,
        collection_name="knowledge_chunks",
        persist_directory=str(persist_directory),
    )


def chunk_markdown_lines(
    path: str,
    lines: list[tuple[int, str]],
) -> list[Chunk]:
    """按完整段落切分 Markdown，并保留路径和原始行号。"""

    if not lines:
        return []

    paragraph_ranges = _find_paragraph_ranges(lines)
    if not paragraph_ranges:
        return []

    chunks: list[Chunk] = []
    current_ranges: list[tuple[int, int]] = []

    for paragraph_range in paragraph_ranges:
        paragraph_length = _range_text_length(
            lines,
            paragraph_range[0],
            paragraph_range[1],
        )
        if paragraph_length > MAX_CHUNK_CHARACTERS:
            raise ValueError(
                "a single Markdown paragraph exceeds 800 characters"
            )

        candidate_ranges = [*current_ranges, paragraph_range]
        candidate_length = _range_text_length(
            lines,
            candidate_ranges[0][0],
            candidate_ranges[-1][1],
        )

        if candidate_length <= MAX_CHUNK_CHARACTERS:
            current_ranges = candidate_ranges
            continue

        if not current_ranges:
            raise ValueError("a single Markdown paragraph exceeds 800 characters")

        chunks.append(
            _build_chunk(
                path,
                lines,
                current_ranges[0][0],
                current_ranges[-1][1],
            )
        )

        overlap_ranges = current_ranges[-CHUNK_OVERLAP_PARAGRAPHS:]
        current_ranges = [*overlap_ranges, paragraph_range]

        if (
            _range_text_length(
                lines,
                current_ranges[0][0],
                current_ranges[-1][1],
            )
            > MAX_CHUNK_CHARACTERS
        ):
            current_ranges = [paragraph_range]

    chunks.append(
        _build_chunk(
            path,
            lines,
            current_ranges[0][0],
            current_ranges[-1][1],
        )
    )

    return chunks
