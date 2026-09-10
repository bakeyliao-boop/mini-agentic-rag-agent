"""预提供定位结果的诊断接线测试；不调用真实模型。"""

import json
from importlib import import_module

import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from rag_core.agentic.evidence import EvidenceRegistry
from rag_core.agentic.runner import AgenticRuntime
from rag_core.agentic.tools import build_knowledge_tools


class ScriptedModel(FakeMessagesListChatModel):
    """仅返回预设消息，用于验证执行器而非模型能力。"""

    def bind_tools(self, tools, **kwargs):
        return self


def test_prepare_handoff_uses_real_results_and_preserves_noise(tmp_path, monkeypatch):
    """真实匹配包含目录和正文，准备阶段不得读正文或登记证据。"""
    diagnostic = import_module("scripts.diagnose_grep_handoff")
    (tmp_path / "指南.md").write_text("目录：标题A\n\n## 标题A\n## 标题B", encoding="utf-8")
    registry = EvidenceRegistry("setup")

    def reject_read(*args):
        raise AssertionError("准备阶段不能登记 read 证据")

    monkeypatch.setattr(registry, "register_read_page", reject_read)
    tools = build_knowledge_tools(tmp_path, object(), registry)
    messages = diagnostic.prepare_handoff(tools, "原始问题", "指南", ["标题A", "标题B"])
    assert [type(message) for message in messages] == [HumanMessage, AIMessage, ToolMessage, AIMessage, ToolMessage]
    assert messages[0].content == "原始问题"
    for index in (1, 3):
        assert messages[index].tool_calls[0]["id"] == messages[index + 1].tool_call_id
    assert messages[3].tool_calls[0]["args"]["path"] == "/指南.md"
    grep_result = json.loads(messages[4].content)
    assert [hit["line"] for hit in grep_result["matches"]] == [1, 3, 4]
    assert all("evidence_id" not in hit for hit in grep_result["matches"])
    assert messages[1].additional_kwargs["diagnostic_source"] == "script_prepared"


def test_prepare_handoff_does_not_pick_a_file_when_ambiguous(tmp_path):
    """不允许利用金标从同名候选中偷偷挑选正确文件。"""
    diagnostic = import_module("scripts.diagnose_grep_handoff")
    for name in ("指南甲.md", "指南乙.md"):
        (tmp_path / name).write_text("标题", encoding="utf-8")
    tools = build_knowledge_tools(tmp_path, object(), EvidenceRegistry("setup"))
    with pytest.raises(ValueError, match="exactly one"):
        diagnostic.prepare_handoff(tools, "问题", "指南", ["标题"])


def test_handoff_sends_seed_to_model_and_uses_same_read_registry(tmp_path):
    """检查模型首轮确实收到真实定位结果，后续 read 可通过原证据闸门。"""
    diagnostic = import_module("scripts.diagnose_grep_handoff")
    (tmp_path / "指南.md").write_text("目录：标题\n\n标题下的真实正文", encoding="utf-8")
    received = []

    class CapturingModel(ScriptedModel):
        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            received.append(messages)
            return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)

    model = CapturingModel(responses=[
        AIMessage(content="", tool_calls=[{"name": "read", "args": {"path": "/指南.md", "start_line": 3, "limit": 1}, "id": "r1", "type": "tool_call"}]),
        AIMessage(content="", tool_calls=[{"name": "GroundedAnswer", "args": {"answer_type": "knowledge", "answer": "标题下的真实正文", "evidence_ids": ["offline:evidence-1"]}, "id": "answer", "type": "tool_call"}]),
    ])
    from rag_core.knowledge.store import build_knowledge_path_snapshot
    runtime = AgenticRuntime(tmp_path, object(), model, build_knowledge_path_snapshot(tmp_path))
    result = diagnostic.run_handoff(runtime, "问题", "指南", ["标题"], thread_id="offline")
    from rag_core.agentic.prompts import KNOWLEDGE_AGENT_PROMPTS
    assert received[0][0].content == KNOWLEDGE_AGENT_PROMPTS["Prompt-V1.9"]
    assert received[0][1].content == "问题"
    assert isinstance(received[0][-1], ToolMessage)
    assert json.loads(received[0][-1].content)["matches"][1]["line"] == 3
    assert result["final"]["answer_type"] == "knowledge"
    assert result["final"]["citations"][0]["start_line"] == 3
    assert [trace["source"] for trace in result["traces"]] == ["script_prepared", "script_prepared", "model"]


def test_handoff_only_allows_four_more_calls(tmp_path):
    """准备两次后只能再执行四次，不能因注入历史变相放宽预算。"""
    diagnostic = import_module("scripts.diagnose_grep_handoff")
    (tmp_path / "指南.md").write_text("标题正文", encoding="utf-8")
    model = ScriptedModel(responses=[AIMessage(content="", tool_calls=[{
        "name": "read", "args": {"path": "/指南.md", "limit": 1},
        "id": f"r{n}", "type": "tool_call",
    }]) for n in range(5)])
    from rag_core.knowledge.store import build_knowledge_path_snapshot
    runtime = AgenticRuntime(tmp_path, object(), model, build_knowledge_path_snapshot(tmp_path))
    result = diagnostic.run_handoff(runtime, "问题", "指南", ["标题"], thread_id="budget")
    traces = result["traces"]
    assert len(traces) == 7
    assert sum(trace["status"] == "success" for trace in traces) == 6
    assert traces[-1]["status"] == "error"
    assert result["remaining_tool_budget"] == 4
