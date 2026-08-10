"""FastAPI 请求和响应模型，进入对应接口时再逐个添加。"""
from pydantic import BaseModel,Field

class TraditionalChatRequest(BaseModel):
    """
    传统RAG接口请求参数
    """
    question: str = Field(min_length=1, description="用户提出的问题")
    path: str = Field(default='/', min_length=1, description="知识库虚拟目录路径，默认为根目录")

class TraditionalChatResponse(BaseModel):
    """
    传统RAG接口响应数据
    """
    answer: str
    hits: list[dict[str,object]]
    latency_ms: float | None = None
    token_usage: dict[str,object] = Field(default_factory=dict)