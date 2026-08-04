from pathlib import Path

import pytest
from langchain_core.embeddings import Embeddings

from app import indexer, knowledge_store
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


def test_build_dashscope_embeddings_uses_openai_compatible_options(
    monkeypatch,
) -> None:
    """Embedding 工厂应把百炼连接参数传给 OpenAIEmbeddings。"""

    received_options: list[dict[str, object]] = []
    fake_embeddings = object()

    def fake_openai_embeddings(**options):
        received_options.append(options)
        return fake_embeddings

    monkeypatch.setattr(
        indexer,
        "OpenAIEmbeddings",
        fake_openai_embeddings,
    )

    result = indexer.build_dashscope_embeddings(
        model="text-embedding-v4",
        dimensions=1024,
        api_key="test-key",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    assert result is fake_embeddings
    assert received_options == [
        {
            "model": "text-embedding-v4",
            "dimensions": 1024,
            "api_key": "test-key",
            "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "check_embedding_ctx_length": False,
        }
    ]


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


def test_search_chroma_index_returns_candidate_only_hits(
    tmp_path: Path,
) -> None:
    """search 应返回带原文定位的候选结果，而不是 evidence。"""

    chunk = Chunk(
        chunk_id="/课程资源/智慧农场.md#L21-L23",
        path="/课程资源/智慧农场.md",
        start_line=21,
        end_line=23,
        text="自动灌溉系统可以根据农作物进行分类灌溉。",
    )
    vector_store = indexer.build_chroma_index(
        [chunk],
        tmp_path / "chroma",
        KeywordEmbeddings(),
    )

    result = indexer.search_chroma_index(
        vector_store,
        "灌溉",
        limit=1,
    )

    assert result["usage"] == "candidate_only"
    assert result["hits"] == [
        {
            "path": chunk.path,
            "start_line": chunk.start_line,
            "end_line": chunk.end_line,
            "score": pytest.approx(1.0),
            "preview": chunk.text,
        }
    ]


def test_search_chroma_index_filters_by_virtual_path(
    tmp_path: Path,
) -> None:
    """指定目录搜索时，不应返回目录范围外的 Chunk。"""

    primary_chunk = Chunk(
        chunk_id="/课程资源/小学/智慧农场.md#L21-L23",
        path="/课程资源/小学/智慧农场.md",
        start_line=21,
        end_line=23,
        text="小学智慧农场使用自动灌溉系统。",
    )
    junior_chunk = Chunk(
        chunk_id="/课程资源/初中/智慧温室.md#L10-L12",
        path="/课程资源/初中/智慧温室.md",
        start_line=10,
        end_line=12,
        text="初中智慧温室也使用自动灌溉系统。",
    )
    vector_store = indexer.build_chroma_index(
        [primary_chunk, junior_chunk],
        tmp_path / "chroma",
        KeywordEmbeddings(),
    )

    result = indexer.search_chroma_index(
        vector_store,
        "灌溉",
        path="/课程资源/初中",
        limit=5,
    )

    assert [hit["path"] for hit in result["hits"]] == [junior_chunk.path]


def test_collect_knowledge_chunks_discovers_markdown_files_in_stable_order(
    tmp_path: Path,
) -> None:
    """应递归读取 Markdown，忽略其他文件，并按虚拟路径稳定排序。"""

    knowledge_root = tmp_path / "knowledge"
    primary_directory = knowledge_root / "小学"
    junior_directory = knowledge_root / "初中"
    primary_directory.mkdir(parents=True)
    junior_directory.mkdir(parents=True)

    (primary_directory / "智慧农场.md").write_text(
        "# 智慧农场\n\n自动灌溉系统可以分类灌溉。\n",
        encoding="utf-8",
        newline="\n",
    )
    (junior_directory / "物联网.md").write_text(
        "# 物联网\n\n物联网可以实现万物互联。\n",
        encoding="utf-8",
        newline="\n",
    )
    (knowledge_root / "说明.txt").write_text(
        "该文件不应进入索引。\n",
        encoding="utf-8",
        newline="\n",
    )

    result = indexer.collect_knowledge_chunks(knowledge_root)

    assert [chunk.path for chunk in result] == [
        "/初中/物联网.md",
        "/小学/智慧农场.md",
    ]
    assert all(chunk.path.endswith(".md") for chunk in result)


def test_collect_knowledge_chunks_returns_empty_list_for_empty_root(
    tmp_path: Path,
) -> None:
    """空知识库目录应返回空列表。"""

    knowledge_root = tmp_path / "knowledge"
    knowledge_root.mkdir()

    result = indexer.collect_knowledge_chunks(knowledge_root)

    assert result == []


def test_collect_knowledge_chunks_rejects_missing_root(
    tmp_path: Path,
) -> None:
    """知识库根目录不存在时应抛出 FileNotFoundError。"""

    knowledge_root = tmp_path / "不存在的知识库"
    assert not knowledge_root.exists()

    with pytest.raises(FileNotFoundError):
        indexer.collect_knowledge_chunks(knowledge_root)


def test_collect_knowledge_chunks_rejects_file_root(
    tmp_path: Path,
) -> None:
    """知识库根路径是文件时应抛出 NotADirectoryError。"""

    knowledge_root = tmp_path / "knowledge.md"
    knowledge_root.write_text(
        "# 这不是知识库目录\n",
        encoding="utf-8",
        newline="\n",
    )
    assert knowledge_root.is_file()

    with pytest.raises(NotADirectoryError):
        indexer.collect_knowledge_chunks(knowledge_root)


def test_build_knowledge_index_collects_and_indexes_markdown(
    tmp_path: Path,
) -> None:
    """应一次完成 Markdown 收集、切块和 Chroma 索引构建。"""

    knowledge_root = tmp_path / "knowledge"
    course_directory = knowledge_root / "课程资源"
    course_directory.mkdir(parents=True)
    (course_directory / "智慧农场.md").write_text(
        "# 智慧农场\n\n自动灌溉系统可以分类灌溉。\n",
        encoding="utf-8",
        newline="\n",
    )

    vector_store = indexer.build_knowledge_index(
        knowledge_root,
        tmp_path / "chroma",
        KeywordEmbeddings(),
    )
    results = vector_store.similarity_search("灌溉", k=1)

    assert len(results) == 1
    assert results[0].metadata["path"] == "/课程资源/智慧农场.md"
    assert "自动灌溉系统" in results[0].page_content


def test_search_hit_can_be_reproduced_by_read(
    tmp_path: Path,
) -> None:
    """search 候选应能通过 path 和行号被 read 精确复现。"""

    knowledge_root = tmp_path / "knowledge"
    course_directory = knowledge_root / "课程资源"
    course_directory.mkdir(parents=True)
    (course_directory / "智慧农场.md").write_text(
        "# 智慧农场\n\n自动灌溉系统可以分类灌溉。\n",
        encoding="utf-8",
        newline="\n",
    )
    vector_store = indexer.build_knowledge_index(
        knowledge_root,
        tmp_path / "chroma",
        KeywordEmbeddings(),
    )

    search_result = indexer.search_chroma_index(
        vector_store,
        "灌溉",
        limit=1,
    )
    hit = search_result["hits"][0]
    read_result = knowledge_store.read_knowledge_page(
        hit["path"],
        knowledge_root,
        start_line=hit["start_line"],
        limit=hit["end_line"] - hit["start_line"] + 1,
    )
    reproduced_text = "\n".join(
        line["text"] for line in read_result["lines"]
    )

    assert reproduced_text == hit["preview"]


def test_rebuilt_knowledge_index_returns_stable_search_result(
    tmp_path: Path,
) -> None:
    """相同语料重复构建索引后，应返回相同的候选结果结构。"""

    knowledge_root = tmp_path / "knowledge"
    knowledge_root.mkdir()
    (knowledge_root / "智慧农场.md").write_text(
        "# 智慧农场\n\n自动灌溉系统可以分类灌溉。\n",
        encoding="utf-8",
        newline="\n",
    )

    first_store = indexer.build_knowledge_index(
        knowledge_root,
        tmp_path / "chroma-first",
        KeywordEmbeddings(),
    )
    second_store = indexer.build_knowledge_index(
        knowledge_root,
        tmp_path / "chroma-second",
        KeywordEmbeddings(),
    )

    first_result = indexer.search_chroma_index(
        first_store,
        "灌溉",
        limit=1,
    )
    second_result = indexer.search_chroma_index(
        second_store,
        "灌溉",
        limit=1,
    )

    assert first_result == second_result
