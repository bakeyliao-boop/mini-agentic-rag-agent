"""Agentic RAG 使用的数据模型。"""

from typing import Annotated, Literal, Self

from pydantic import BaseModel, Field, model_validator

EvidenceId = Annotated[str, Field(min_length=1)]


class Chunk(BaseModel):
    """表示一段可检索的 Markdown 文本及其原文位置。"""

    chunk_id: str = Field(min_length=1)
    path: str = Field(min_length=1)
    start_line: int = Field(ge=1)
    end_line: int = Field(ge=1)
    text: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_line_range(self) -> Self:
        """确保结束行号不早于起始行号。"""

        if self.end_line < self.start_line:
            raise ValueError("end_line must be greater than or equal to start_line")
        return self


class Evidence(BaseModel):
    """表示通过 read 读取并由服务端注册的可信原文。"""

    evidence_id: str = Field(min_length=1)
    path: str = Field(min_length=1)
    start_line: int = Field(ge=1)
    end_line: int = Field(ge=1)
    quote: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_line_range(self) -> Self:
        """确保结束行号不早于起始行号。"""

        if self.end_line < self.start_line:
            raise ValueError("end_line must be greater than or equal to start_line")
        return self


class Citation(BaseModel):
    """表示服务端验证后展示给用户的原文引用。"""

    path: str = Field(min_length=1)
    start_line: int = Field(ge=1)
    end_line: int = Field(ge=1)
    quote: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_line_range(self) -> Self:
        """确保结束行号不早于起始行号。"""

        if self.end_line < self.start_line:
            raise ValueError("end_line must be greater than or equal to start_line")
        return self


class GroundedAnswer(BaseModel):
    """表示模型返回、等待服务端验证的结构化回答。"""

    answer_type: Literal[
        "knowledge",
        "directory",
        "conversation",
        "insufficient",
    ]
    answer: str = Field(min_length=1)
    evidence_ids: list[EvidenceId] = Field(default_factory=list)
