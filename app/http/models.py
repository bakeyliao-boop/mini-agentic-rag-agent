"""FastAPI 请求和响应模型。"""
from pydantic import BaseModel,Field
from typing import Literal

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

class AgenticChatRequest(BaseModel):
    '''Agentic RAG接口请求参数'''
    question:str=Field(
        min_length=1,
        description='query from user',
    )
    thread_id:str=Field(
        min_length=1,
        description='thread_id of this turns',
    )
class AgenticChatResponse(BaseModel):
    '''Agentic RAG接口响应数据'''
    answer_type:Literal['knowledge','insufficient']
    answer:str
    citations:list[dict[str,object]]
    tool_traces:list[dict[str,object]]
    token_usage:dict[str,object]=Field(default_factory=dict)
    thread_id:str
