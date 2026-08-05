"""本轮 Agent 运行使用的原文证据登记册。"""

from collections.abc import Mapping
from pathlib import Path

from app.knowledge_store import read_markdown_lines, resolve_knowledge_path
from app.models import Citation, Evidence, GroundedAnswer


class EvidenceRegistry:
    """保存一次 Agent 运行中由 read 注册的可信原文。"""

    def __init__(self, run_id: str) -> None:
        if not isinstance(run_id, str) or not run_id.strip():
            raise ValueError("run_id must be a non-empty string")

        self.run_id = run_id.strip()
        self._evidences: dict[str, Evidence] = {}

    def register_read_page(
        self,
        read_result: Mapping[str, object],
    ) -> list[Evidence]:
        """将 read 返回页中的非空原文行注册为 Evidence。"""

        path = read_result.get("path")
        lines = read_result.get("lines")
        if not isinstance(path, str) or not path:
            raise ValueError("read result path must be a non-empty string")
        if not isinstance(lines, list):
            raise ValueError("read result lines must be a list")

        registered: list[Evidence] = [] #保存本次新生成的Evidence对象
        for line in lines:
            if not isinstance(line, dict):
                raise ValueError("each read result line must be a dictionary")

            line_number = line.get("line")
            text = line.get("text")
            if not isinstance(line_number, int) or line_number < 1:
                raise ValueError("read result line number must be positive")
            if not isinstance(text, str):
                raise ValueError("read result line text must be a string")
            if not text.strip():
                continue

            evidence_id = (
                f"{self.run_id}:evidence-{len(self._evidences) + 1}"
            )
            evidence = Evidence(
                evidence_id=evidence_id,
                path=path,
                start_line=line_number,
                end_line=line_number,
                quote=text,
            )
            self._evidences[evidence_id] = evidence
            registered.append(evidence)

        return registered

    def get(self, evidence_id: str) -> Evidence:
        """根据 evidence ID 返回本轮已经注册的证据。"""

        return self._evidences[evidence_id]


def validate_answer_evidence(
    answer: GroundedAnswer,
    registry: EvidenceRegistry,
) -> list[Evidence]:
    """校验回答引用的 ID，并返回本轮登记的可信 Evidence。"""

    if answer.answer_type == "knowledge" and not answer.evidence_ids:
        raise ValueError("knowledge answer requires evidence")

    evidences: list[Evidence] = []
    for evidence_id in answer.evidence_ids:
        try:
            evidence = registry.get(evidence_id)
        except KeyError as error:
            raise ValueError(
                f"unknown evidence ID: {evidence_id}"
            ) from error
        evidences.append(evidence)

    return evidences


def build_citations(evidences: list[Evidence]) -> list[Citation]:
    """将已验证的 Evidence 转换为可返回给用户的 Citation。"""

    citations: list[Citation] = []
    for evidence in evidences:
        citations.append(
            Citation(
                path=evidence.path,
                start_line=evidence.start_line,
                end_line=evidence.end_line,
                quote=evidence.quote,
            )
        )

    return citations


def validate_evidence_sources(
    evidences: list[Evidence],
    knowledge_root: Path,
) -> list[Evidence]:
    """重新读取 Markdown，确认 Evidence 的路径、行号和原文未变化。"""

    validated: list[Evidence] = []
    for evidence in evidences:
        try:
            source_path = resolve_knowledge_path(
                evidence.path,
                knowledge_root,
            )
            source_lines = read_markdown_lines(source_path)
        except (FileNotFoundError, IsADirectoryError, ValueError) as error:
            raise ValueError(
                f"evidence source has changed: {evidence.evidence_id}"
            ) from error

        selected_lines = [
            text
            for line_number, text in source_lines
            if evidence.start_line <= line_number <= evidence.end_line  #当前行号是否在 Evidence 记录的范围内
        ]
        expected_line_count = evidence.end_line - evidence.start_line + 1
        current_quote = "\n".join(selected_lines)

        if (
            len(selected_lines) != expected_line_count
            or current_quote != evidence.quote
        ):
            raise ValueError(
                f"evidence source has changed: {evidence.evidence_id}"
            )

        validated.append(evidence)

    return validated
