"""Markdown 切块与本地索引。"""

from pathlib import Path

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from app.knowledge_store import normalize_virtual_path, read_markdown_lines
from app.models import Chunk

MAX_CHUNK_CHARACTERS = 800
CHUNK_OVERLAP_PARAGRAPHS = 1


def build_dashscope_embeddings(
    model: str,
    dimensions: int,
    api_key: str,
    base_url: str,
) -> OpenAIEmbeddings:
    """使用百炼 OpenAI-compatible 接口创建文本向量模型。"""

    return OpenAIEmbeddings(
        model=model,
        dimensions=dimensions,
        api_key=api_key,
        base_url=base_url,
    )


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


def collect_knowledge_chunks(knowledge_root: Path) -> list[Chunk]:
    """
    寻找 .md → 排序 → 转虚拟路径 → 读取内容 → 切成 Chunk
    递归读取知识库中的 Markdown 文件，并按虚拟路径稳定生成 Chunk。
    """

    if not knowledge_root.exists():
        raise FileNotFoundError(f"knowledge root does not exist: {knowledge_root}")
    if not knowledge_root.is_dir():  # 根路径必须是目录
        raise NotADirectoryError(f"knowledge root is not a directory: {knowledge_root}")

    markdown_paths = sorted(
        knowledge_root.rglob("*.md"),
        key=lambda path: path.relative_to(knowledge_root).as_posix(),
    )
    chunks: list[Chunk] = []

    for source_path in markdown_paths:
        relative_path = source_path.relative_to(knowledge_root).as_posix()
        virtual_path = normalize_virtual_path(f"/{relative_path}")
        lines = read_markdown_lines(source_path)
        chunks.extend(chunk_markdown_lines(virtual_path, lines))

    return chunks


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
        collection_metadata={"hnsw:space": "cosine"},
        persist_directory=str(persist_directory),
    )


def build_knowledge_index(
    knowledge_root: Path,
    persist_directory: Path,
    embedding: Embeddings,
) -> Chroma:
    """收集知识库 Markdown、完成切块并构建本地 Chroma 索引。"""

    chunks = collect_knowledge_chunks(knowledge_root)
    return build_chroma_index(chunks, persist_directory, embedding)


def search_chroma_index(
    vector_store: Chroma,
    query: str,
    path: str = "/",
    limit: int = 5,
) -> dict[str, object]:
    """
    在指定虚拟路径范围内检索候选 Chunk。

    search 只负责定位候选内容，不注册 evidence。最终回答引用的原文
    必须由后续 read 工具重新读取和验证。
    """

    # 先校验调用参数，避免无意义查询或返回过多候选结果。
    if not isinstance(query, str):
        raise TypeError("query must be a string")
    if not query.strip():
        raise ValueError("query must not be empty")
    if limit < 1 or limit > 5:
        raise ValueError("limit must be between 1 and 5")

    # 将调用者传入的路径整理成统一的虚拟 POSIX 路径。
    normalized_path = normalize_virtual_path(path)
    search_filter: dict[str, object] | None = None

    # 根路径表示搜索整个知识库；非根路径需要限制搜索范围。
    if normalized_path != "/":
        # 先读取索引中已经保存的 metadata，找出位于指定范围内的文件。
        stored = vector_store.get(include=["metadatas"])    #调用 Chroma 的 get 方法获取索引中存储的 metadata
        metadatas = stored.get("metadatas") or []
        path_prefix = f"{normalized_path}/"
        allowed_paths: set[str] = set() #候选路径集合

        for metadata in metadatas:
            if not isinstance(metadata, dict):
                continue

            document_path = metadata.get("path")
            if not isinstance(document_path, str):
                continue

            if (
                document_path == normalized_path
                or document_path.startswith(path_prefix)
            ):
                allowed_paths.add(document_path)

        # 指定范围内没有已索引文件时，直接返回空候选结果。
        if not allowed_paths:
            return {
                "hits": [],
                "usage": "candidate_only",
            }

        # Chroma 使用 metadata 过滤器，只在允许的文件路径中检索。
        search_filter = {
            "path": {
                "$in": sorted(allowed_paths),
            }
        }

    # 根据查询文本生成向量并取得相关度最高的候选 Document。
    results = vector_store.similarity_search_with_relevance_scores(
        query,
        k=limit,
        filter=search_filter,
    )

    # 将 LangChain Document 转成项目约定的 search 输出格式。
    return {
        "hits": [
            {
                "path": document.metadata["path"],
                "start_line": document.metadata["start_line"],
                "end_line": document.metadata["end_line"],
                "score": float(score),
                "preview": document.page_content,
            }
            for document, score in results
        ],
        "usage": "candidate_only",
    }


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
