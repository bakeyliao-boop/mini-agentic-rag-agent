from pathlib import Path

import pytest
from langchain_core.embeddings import Embeddings

from app import indexer
from app.models import Chunk


class KeywordEmbeddings(Embeddings):
    """根据测试关键词生成固定向量，避免调用外部 Embedding 服务。"""

    @staticmethod
    def _embed(text: str) -> list[float]:
        return [
            float("灌溉" in text),
            float("物联网" in text or "万物互联" in text),
        ]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


def test_chunk_markdown_lines_returns_single_chunk_for_short_document() -> None:
    """较短的 Markdown 应生成一个保留原文位置的 Chunk。"""

    lines = [
        (1, "# 智慧农场"),
        (2, ""),
        (3, "自动灌溉系统可以根据不同的农作物进行分类灌溉。"),
    ]

    result = indexer.chunk_markdown_lines(
        "/课程资源/智慧农场.md",
        lines,
    )

    assert result == [
        Chunk(
            chunk_id="/课程资源/智慧农场.md#L1-L3",
            path="/课程资源/智慧农场.md",
            start_line=1,
            end_line=3,
            text=(
                "# 智慧农场\n"
                "\n"
                "自动灌溉系统可以根据不同的农作物进行分类灌溉。"
            ),
        )
    ]


def test_chunk_markdown_lines_returns_empty_list_for_empty_document() -> None:
    """空 Markdown 不应生成 Chunk。"""

    result = indexer.chunk_markdown_lines(
        "/课程资源/空文件.md",
        [],
    )

    assert result == []


def test_chunk_markdown_lines_splits_long_document_with_overlap() -> None:
    """长文档应按完整段落切块，并让相邻 Chunk 重叠一个段落。"""

    irrigation = "自动灌溉系统可以根据农作物进行分类灌溉。" * 15
    weather = "气象站可以实时监测天气并预测农业灾害。" * 15
    fertilizer = "计算机可以自动控制肥料的种类和用量。" * 15
    lines = [
        (1, "# 智慧农场"),
        (2, ""),
        (3, irrigation),
        (4, ""),
        (5, weather),
        (6, ""),
        (7, fertilizer),
    ]

    result = indexer.chunk_markdown_lines(
        "/课程资源/智慧农场.md",
        lines,
    )

    assert result == [
        Chunk(
            chunk_id="/课程资源/智慧农场.md#L1-L5",
            path="/课程资源/智慧农场.md",
            start_line=1,
            end_line=5,
            text=f"# 智慧农场\n\n{irrigation}\n\n{weather}",
        ),
        Chunk(
            chunk_id="/课程资源/智慧农场.md#L5-L7",
            path="/课程资源/智慧农场.md",
            start_line=5,
            end_line=7,
            text=f"{weather}\n\n{fertilizer}",
        ),
    ]
    assert all(len(chunk.text) <= 800 for chunk in result)


def test_chunk_markdown_lines_rejects_oversized_paragraph_after_heading() -> None:
    """标题后的单个段落超过 800 字符时应抛出 ValueError。"""

    oversized_paragraph = "自动灌溉系统" * 134
    lines = [
        (1, "# 智慧农场"),
        (2, ""),
        (3, oversized_paragraph),
    ]

    with pytest.raises(ValueError):
        indexer.chunk_markdown_lines(
            "/课程资源/智慧农场.md",
            lines,
        )


def test_chunk_markdown_lines_returns_same_result_when_repeated() -> None:
    """同一份 Markdown 重复切块时应返回完全相同的结果。"""

    lines = [
        (19, "# 智慧农场"),
        (20, ""),
        (21, "## 分类灌溉"),
        (22, ""),
        (23, "自动灌溉系统可以根据不同农作物进行分类灌溉。"),
    ]

    first_result = indexer.chunk_markdown_lines(
        "/课程资源/智慧农场.md",
        lines,
    )
    second_result = indexer.chunk_markdown_lines(
        "/课程资源/智慧农场.md",
        lines,
    )

    assert second_result == first_result


def test_chunk_to_document_preserves_text_and_metadata() -> None:
    """Chunk 转为 Document 后应保留正文和原文定位信息。"""

    chunk = Chunk(
        chunk_id="/课程资源/智慧农场.md#L19-L23",
        path="/课程资源/智慧农场.md",
        start_line=19,
        end_line=23,
        text="# 智慧农场\n\n自动灌溉系统可以分类灌溉。",
    )

    result = indexer.chunk_to_document(chunk)

    assert result.page_content == chunk.text
    assert result.metadata == {
        "chunk_id": chunk.chunk_id,
        "path": chunk.path,
        "start_line": chunk.start_line,
        "end_line": chunk.end_line,
    }


def test_build_chroma_index_returns_matching_chunk(
    tmp_path: Path,
) -> None:
    """本地 Chroma 应返回语义匹配的 Chunk 及其原文定位信息。"""

    chunks = [
        Chunk(
            chunk_id="/课程资源/智慧农场.md#L21-L23",
            path="/课程资源/智慧农场.md",
            start_line=21,
            end_line=23,
            text="自动灌溉系统可以根据农作物进行分类灌溉。",
        ),
        Chunk(
            chunk_id="/课程资源/互联网与物联网.md#L20-L24",
            path="/课程资源/互联网与物联网.md",
            start_line=20,
            end_line=24,
            text="物联网让学习者感受万物互联的场景和价值。",
        ),
    ]

    vector_store = indexer.build_chroma_index(
        chunks,
        tmp_path / "chroma",
        KeywordEmbeddings(),
    )
    results = vector_store.similarity_search("灌溉", k=1)

    assert len(results) == 1
    assert results[0].page_content == chunks[0].text
    assert results[0].metadata == {
        "chunk_id": chunks[0].chunk_id,
        "path": chunks[0].path,
        "start_line": chunks[0].start_line,
        "end_line": chunks[0].end_line,
    }
