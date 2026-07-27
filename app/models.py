"""Agentic RAG 使用的数据模型。"""

from typing import Self

from pydantic import BaseModel, Field, model_validator


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
