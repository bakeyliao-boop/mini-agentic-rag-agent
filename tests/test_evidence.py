from importlib import import_module
from pathlib import Path

import pytest

from app.models import Citation, Evidence, GroundedAnswer


def test_register_read_page_creates_evidence_for_nonempty_line() -> None:
    """read 返回的非空原文行应注册为本轮 Evidence。"""

    evidence_module = import_module("app.evidence")
    registry = evidence_module.EvidenceRegistry(run_id="run-001")
    read_result = {
        "path": "/课程资源/智慧农场.md",
        "lines": [
            {
                "line": 3,
                "text": "气象站可以采集环境数据。",
            }
        ],
        "next_line": None,
    }

    result = registry.register_read_page(read_result)

    expected = Evidence(
        evidence_id="run-001:evidence-1",
        path="/课程资源/智慧农场.md",
        start_line=3,
        end_line=3,
        quote="气象站可以采集环境数据。",
    )
    assert result == [expected]
    assert registry.get(expected.evidence_id) == expected


def test_validate_answer_evidence_rejects_unknown_id() -> None:
    """模型提交未在本轮注册的 evidence ID 时应拒绝回答。"""

    evidence_module = import_module("app.evidence")
    registry = evidence_module.EvidenceRegistry(run_id="run-001")
    answer = GroundedAnswer(
        answer_type="knowledge",
        answer="气象站可以采集环境数据。",
        evidence_ids=["run-001:evidence-999"],
    )

    with pytest.raises(ValueError, match="unknown evidence ID"):
        evidence_module.validate_answer_evidence(answer, registry)


def test_validate_answer_evidence_returns_registered_evidence() -> None:
    """模型提交本轮已登记的 ID 时应返回对应 Evidence。"""

    evidence_module = import_module("app.evidence")
    registry = evidence_module.EvidenceRegistry(run_id="run-001")   #区分不同运行，防止上一轮的Evidence被这一轮使用
    registered = registry.register_read_page(
        {
            "path": "/课程资源/智慧农场.md",
            "lines": [
                {
                    "line": 3,
                    "text": "气象站可以采集环境数据。",
                }
            ],
            "next_line": None,
        }
    )
    answer = GroundedAnswer(
        answer_type="knowledge",
        answer="气象站可以采集环境数据。",
        evidence_ids=[registered[0].evidence_id],
    )

    result = evidence_module.validate_answer_evidence(answer, registry)

    assert result == registered


def test_validate_answer_evidence_rejects_id_from_another_run() -> None:
    """其他运行登记的 evidence ID 不能用于当前运行。"""

    evidence_module = import_module("app.evidence")
    first_registry = evidence_module.EvidenceRegistry(run_id="run-001")
    second_registry = evidence_module.EvidenceRegistry(run_id="run-002")
    registered = first_registry.register_read_page(
        {
            "path": "/课程资源/智慧农场.md",
            "lines": [
                {
                    "line": 3,
                    "text": "气象站可以采集环境数据。",
                }
            ],
            "next_line": None,
        }
    )
    answer = GroundedAnswer(
        answer_type="knowledge",
        answer="气象站可以采集环境数据。",
        evidence_ids=[registered[0].evidence_id],
    )

    with pytest.raises(ValueError, match="unknown evidence ID"):
        evidence_module.validate_answer_evidence(answer, second_registry)


def test_build_citations_converts_validated_evidence() -> None:
    """已验证的 Evidence 应转换为可返回给用户的 Citation。"""

    evidence_module = import_module("app.evidence")
    evidence = Evidence(
        evidence_id="run-001:evidence-1",
        path="/课程资源/智慧农场.md",
        start_line=3,
        end_line=3,
        quote="气象站可以采集环境数据。",
    )

    result = evidence_module.build_citations([evidence])

    assert result == [
        Citation(
            path="/课程资源/智慧农场.md",
            start_line=3,
            end_line=3,
            quote="气象站可以采集环境数据。",
        )
    ]


def test_validate_evidence_sources_rejects_changed_markdown(
    tmp_path: Path,
) -> None:
    """登记后的 Markdown 原文发生变化时，旧 Evidence 应失效。"""

    evidence_module = import_module("app.evidence")
    knowledge_root = tmp_path / "knowledge"
    knowledge_root.mkdir()
    source_path = knowledge_root / "智慧农场.md"
    source_path.write_text(
        "气象站可以采集环境数据。\n",
        encoding="utf-8",
    )
    registry = evidence_module.EvidenceRegistry(run_id="run-001")
    registered = registry.register_read_page(
        {
            "path": "/智慧农场.md",
            "lines": [
                {
                    "line": 1,
                    "text": "气象站可以采集环境数据。",
                }
            ],
            "next_line": None,
        }
    )
    source_path.write_text(
        "气象站数据已经修改。\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="evidence source has changed"):
        evidence_module.validate_evidence_sources(
            registered,
            knowledge_root,
        )


def test_validate_evidence_sources_returns_unchanged_evidence(
    tmp_path: Path,
) -> None:
    """Markdown 原文未变化时，应返回通过校验的 Evidence。"""

    evidence_module = import_module("app.evidence")
    knowledge_root = tmp_path / "knowledge"
    knowledge_root.mkdir()
    source_path = knowledge_root / "智慧农场.md"
    source_path.write_text(
        "气象站可以采集环境数据。\n",
        encoding="utf-8",
    )
    registry = evidence_module.EvidenceRegistry(run_id="run-001")
    registered = registry.register_read_page(
        {
            "path": "/智慧农场.md",
            "lines": [
                {
                    "line": 1,
                    "text": "气象站可以采集环境数据。",
                }
            ],
            "next_line": None,
        }
    )

    result = evidence_module.validate_evidence_sources(
        registered,
        knowledge_root,
    )

    assert result == registered


def test_validate_answer_evidence_rejects_knowledge_without_ids() -> None:
    """knowledge 回答没有引用任何 Evidence 时应被拒绝。"""

    evidence_module = import_module("app.evidence")
    registry = evidence_module.EvidenceRegistry(run_id="run-001")
    answer = GroundedAnswer(
        answer_type="knowledge",
        answer="气象站可以采集环境数据。",
        evidence_ids=[],
    )

    with pytest.raises(
        ValueError,
        match="knowledge answer requires evidence",
    ):
        evidence_module.validate_answer_evidence(answer, registry)
