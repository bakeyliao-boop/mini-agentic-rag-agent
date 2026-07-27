import pytest
from pydantic import ValidationError

from app.models import Chunk, Citation, Evidence, GroundedAnswer


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


def test_evidence_accepts_valid_data() -> None:
    """合法数据应能创建 Evidence。"""

    evidence = Evidence(
        evidence_id="ev-001",
        path="/课程资源/智慧农场.md",
        start_line=12,
        end_line=13,
        quote="气象站可以采集环境数据。",
    )

    assert evidence.evidence_id == "ev-001"
    assert evidence.path == "/课程资源/智慧农场.md"
    assert evidence.start_line == 12
    assert evidence.end_line == 13
    assert evidence.quote == "气象站可以采集环境数据。"


@pytest.mark.parametrize(
    "field_name",
    ["evidence_id", "path", "quote"],
)
def test_evidence_rejects_empty_required_text(field_name: str) -> None:
    """Evidence 的必填文本字段不能为空。"""

    data = {
        "evidence_id": "ev-001",
        "path": "/课程资源/智慧农场.md",
        "start_line": 12,
        "end_line": 13,
        "quote": "气象站可以采集环境数据。",
    }
    data[field_name] = ""

    with pytest.raises(ValidationError):
        Evidence.model_validate(data)


@pytest.mark.parametrize(
    "field_name",
    ["start_line", "end_line"],
)
def test_evidence_rejects_line_number_less_than_one(
    field_name: str,
) -> None:
    """Evidence 的行号必须从 1 开始。"""

    data = {
        "evidence_id": "ev-001",
        "path": "/课程资源/智慧农场.md",
        "start_line": 12,
        "end_line": 13,
        "quote": "气象站可以采集环境数据。",
    }
    data[field_name] = 0

    with pytest.raises(ValidationError):
        Evidence.model_validate(data)


def test_evidence_rejects_end_line_before_start_line() -> None:
    """Evidence 的结束行号不能早于起始行号。"""

    with pytest.raises(ValidationError):
        Evidence(
            evidence_id="ev-001",
            path="/课程资源/智慧农场.md",
            start_line=13,
            end_line=12,
            quote="气象站可以采集环境数据。",
        )


def test_citation_accepts_valid_data() -> None:
    """合法数据应能创建 Citation。"""

    citation = Citation(
        path="/课程资源/智慧农场.md",
        start_line=12,
        end_line=13,
        quote="气象站可以采集环境数据。",
    )

    assert citation.path == "/课程资源/智慧农场.md"
    assert citation.start_line == 12
    assert citation.end_line == 13
    assert citation.quote == "气象站可以采集环境数据。"


@pytest.mark.parametrize(
    "field_name",
    ["path", "quote"],
)
def test_citation_rejects_empty_required_text(field_name: str) -> None:
    """Citation 的必填文本字段不能为空。"""

    data = {
        "path": "/课程资源/智慧农场.md",
        "start_line": 12,
        "end_line": 13,
        "quote": "气象站可以采集环境数据。",
    }
    data[field_name] = ""

    with pytest.raises(ValidationError):
        Citation.model_validate(data)


@pytest.mark.parametrize(
    "field_name",
    ["start_line", "end_line"],
)
def test_citation_rejects_line_number_less_than_one(
    field_name: str,
) -> None:
    """Citation 的行号必须从 1 开始。"""

    data = {
        "path": "/课程资源/智慧农场.md",
        "start_line": 12,
        "end_line": 13,
        "quote": "气象站可以采集环境数据。",
    }
    data[field_name] = 0

    with pytest.raises(ValidationError):
        Citation.model_validate(data)


def test_citation_rejects_end_line_before_start_line() -> None:
    """Citation 的结束行号不能早于起始行号。"""

    with pytest.raises(ValidationError):
        Citation(
            path="/课程资源/智慧农场.md",
            start_line=13,
            end_line=12,
            quote="气象站可以采集环境数据。",
        )


@pytest.mark.parametrize(
    "answer_type",
    [
        "knowledge",
        "directory",
        "conversation",
        "insufficient",
    ],
)
def test_grounded_answer_accepts_supported_answer_types(
    answer_type: str,
) -> None:
    """GroundedAnswer 应接受四种约定的回答类型。"""

    answer = GroundedAnswer.model_validate(
        {
            "answer_type": answer_type,
            "answer": "这是一个有效回答。",
            "evidence_ids": ["ev-001"],
        }
    )

    assert answer.answer_type == answer_type


def test_grounded_answer_rejects_unknown_answer_type() -> None:
    """未知回答类型应被拒绝。"""

    with pytest.raises(ValidationError):
        GroundedAnswer.model_validate(
            {
                "answer_type": "unknown",
                "answer": "这是一个无效回答。",
                "evidence_ids": [],
            }
        )


def test_grounded_answer_rejects_empty_answer() -> None:
    """回答正文不能为空。"""

    with pytest.raises(ValidationError):
        GroundedAnswer(
            answer_type="insufficient",
            answer="",
            evidence_ids=[],
        )


def test_grounded_answer_rejects_empty_evidence_id() -> None:
    """证据 ID 不能为空字符串。"""

    with pytest.raises(ValidationError):
        GroundedAnswer(
            answer_type="knowledge",
            answer="这是一个知识回答。",
            evidence_ids=[""],
        )


def test_grounded_answer_defaults_to_empty_evidence_ids() -> None:
    """未提供证据 ID 时应使用独立的空列表。"""

    answer = GroundedAnswer(
        answer_type="insufficient",
        answer="当前证据不足。",
    )

    assert answer.evidence_ids == []
