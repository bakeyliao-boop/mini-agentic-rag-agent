import pytest
from pydantic import ValidationError

from app.models import Chunk


def test_chunk_accepts_valid_data() -> None:
    """合法数据应能创建 Chunk。"""

    chunk = Chunk(
        chunk_id="chunk-001",
        path="/课程资源/智慧农场.md",
        start_line=10,
        end_line=12,
        text="气象站可以采集环境数据。",
    )

    assert chunk.chunk_id == "chunk-001"
    assert chunk.path == "/课程资源/智慧农场.md"
    assert chunk.start_line == 10
    assert chunk.end_line == 12
    assert chunk.text == "气象站可以采集环境数据。"


@pytest.mark.parametrize(
    "field_name",
    ["chunk_id", "path", "text"],
)
def test_chunk_rejects_empty_required_text(field_name: str) -> None:
    """必填文本字段不能为空字符串。"""

    data = {
        "chunk_id": "chunk-001",
        "path": "/课程资源/智慧农场.md",
        "start_line": 10,
        "end_line": 12,
        "text": "气象站可以采集环境数据。",
    }
    data[field_name] = ""

    with pytest.raises(ValidationError):
        Chunk.model_validate(data)


@pytest.mark.parametrize(
    "field_name",
    ["start_line", "end_line"],
)
def test_chunk_rejects_line_number_less_than_one(
    field_name: str,
) -> None:
    """起始和结束行号都必须从 1 开始。"""

    data = {
        "chunk_id": "chunk-001",
        "path": "/课程资源/智慧农场.md",
        "start_line": 10,
        "end_line": 12,
        "text": "气象站可以采集环境数据。",
    }
    data[field_name] = 0

    with pytest.raises(ValidationError):
        Chunk.model_validate(data)


def test_chunk_rejects_end_line_before_start_line() -> None:
    """结束行号不能早于起始行号。"""

    with pytest.raises(ValidationError):
        Chunk(
            chunk_id="chunk-001",
            path="/课程资源/智慧农场.md",
            start_line=12,
            end_line=10,
            text="气象站可以采集环境数据。",
        )


def test_chunk_accepts_single_line_range() -> None:
    """起始和结束行号相同时表示单行 Chunk。"""

    chunk = Chunk(
        chunk_id="chunk-001",
        path="/课程资源/智慧农场.md",
        start_line=12,
        end_line=12,
        text="气象站可以采集环境数据。",
    )

    assert chunk.start_line == chunk.end_line == 12
