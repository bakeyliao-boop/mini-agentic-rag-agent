"""预提供真实 glob/grep 结果，诊断模型是否会按位置读取并结束。

准备历史由脚本生成，不计作模型自主选择；正文仍只能由后续 read 登记。
正式 Agent 不变，诊断 Agent 的剩余预算为四次，准备两次后总预算仍为六次。
运行：python -m scripts.diagnose_grep_handoff
"""

import hashlib
import json
from pathlib import Path
from time import perf_counter
from uuid import uuid4

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver

from evaluation.cli.pilot_agentic import build_pilot_agentic_runtime_from_project
from evaluation.dataset import save_evaluation_result
from rag_core.agentic.agent import (
    KNOWLEDGE_TOOL_NAMES,
    KnowledgeRetrievalStopMiddleware,
    KnowledgeToolCallLimitMiddleware,
    extract_token_usage,
    extract_tool_traces,
)
from rag_core.agentic.evidence import EvidenceRegistry
from rag_core.agentic.prompts import KNOWLEDGE_AGENT_PROMPTS
from rag_core.agentic.runner import (
    AgenticRuntime,
    _stream_update_messages,
    _tool_result_summary,
    finalize_agent_result,
)
from rag_core.agentic.tools import build_knowledge_tools
from rag_core.knowledge.store import resolve_knowledge_path
from rag_core.models import GroundedAnswer
from rag_core.settings import load_settings_from_env

TITLE = "深圳市知识产权公共服务事项办事指南（第二版）"
PATTERNS = ["专利优先审查", "专利快速预审"]
PROMPT_VERSION = "Prompt-V1.9"


def prepare_handoff(tools, question: str, title: str, patterns: list[str]):
    """真实执行准备工具，并保留未经筛选的匹配结果与调用配对。"""
    by_name = {tool.name: tool for tool in tools}
    messages = [HumanMessage(content=question)]

    def execute(name, args):
        call = {"name": name, "args": args, "id": f"prepared-{name}-{uuid4().hex}", "type": "tool_call"}
        result = by_name[name].invoke(call)
        if not isinstance(result, ToolMessage) or result.status == "error":
            raise ValueError(f"diagnostic preparation failed: {name}")
        messages.extend([
            AIMessage(content="", tool_calls=[call], additional_kwargs={"diagnostic_source": "script_prepared"}),
            result,
        ])
        return json.loads(result.content)

    glob_result = execute("glob", {"target": title, "target_type": "filename", "path": "/"})
    matches = glob_result.get("matches", [])
    if len(matches) != 1:
        raise ValueError("diagnostic requires exactly one glob match")
    grep_result = execute("grep", {"path": matches[0]["path"], "patterns": list(patterns), "limit": 20})
    if not grep_result.get("matches"):
        raise ValueError("diagnostic requires real grep matches")
    return messages


def build_handoff_agent(chat_model, tools):
    """仅在诊断实例中扣除两次准备调用，不修改正式服务配置。"""
    return create_agent(
        model=chat_model,
        tools=tools,
        system_prompt=KNOWLEDGE_AGENT_PROMPTS[PROMPT_VERSION],
        middleware=[
            KnowledgeToolCallLimitMiddleware(run_limit=4, exit_behavior="end"),
            KnowledgeRetrievalStopMiddleware(),
        ],
        checkpointer=InMemorySaver(),
        response_format=GroundedAnswer,
    )


def run_handoff(
    runtime: AgenticRuntime,
    question: str,
    title: str,
    patterns: list[str],
    *,
    thread_id: str,
    on_event=None,
) -> dict:
    """将准备历史送入真实 Agent，收集模型后续事件与原证据闸门结果。"""
    started = perf_counter()
    registry = EvidenceRegistry(thread_id)
    tools = build_knowledge_tools(runtime.knowledge_root, runtime.vector_store, registry, path_snapshot=runtime.path_snapshot)
    seed = prepare_handoff(tools, question, title, patterns)
    prepared_ids = {message.tool_calls[0]["id"] for message in seed if isinstance(message, AIMessage)}
    preparation_ms = (perf_counter() - started) * 1000
    agent = build_handoff_agent(runtime.chat_model, tools)
    names = {message.tool_calls[0]["id"]: message.tool_calls[0]["name"] for message in seed if isinstance(message, AIMessage)}
    steps = {call_id: index for index, call_id in enumerate(names, 1)}
    completed_ids = set(prepared_ids)
    events = []
    final_state = None
    error_type = None

    def emit(event):
        events.append(event)
        if on_event is not None:
            on_event(event)

    model_started = perf_counter()
    try:
        for part in agent.stream(
            {"messages": seed},
            config={"configurable": {"thread_id": thread_id}},
            stream_mode=["updates", "values"],
            version="v2",
        ):
            if part.get("type") == "values":
                final_state = part["data"]
                continue
            for message in _stream_update_messages(part):
                if isinstance(message, AIMessage):
                    for call in message.tool_calls:
                        if call["name"] not in KNOWLEDGE_TOOL_NAMES or call["id"] in steps:
                            continue
                        steps[call["id"]] = len(steps) + 1
                        names[call["id"]] = call["name"]
                        emit({"event": "tool_started", "data": {
                            "step": steps[call["id"]], "tool_call_id": call["id"],
                            "name": call["name"], "args": call["args"], "source": "model",
                        }})
                elif isinstance(message, ToolMessage):
                    call_id = message.tool_call_id
                    if call_id not in steps or call_id in completed_ids:
                        continue
                    completed_ids.add(call_id)
                    named = message.model_copy(update={"name": names[call_id]})
                    emit({"event": "tool_completed", "data": {
                        "step": steps[call_id], "tool_call_id": call_id,
                        "name": names[call_id], "status": message.status,
                        "summary": _tool_result_summary(named), "source": "model",
                    }})
    except Exception as error:
        # 不把供应商异常中的认证信息或请求头写入报告。
        error_type = type(error).__name__
    model_ms = (perf_counter() - model_started) * 1000
    state = final_state or {"messages": seed}
    messages = state.get("messages", seed)
    traces = extract_tool_traces(messages)
    for trace in traces:
        trace["source"] = "script_prepared" if trace["tool_call_id"] in prepared_ids else "model"
    structured = state.get("structured_response")
    return {
        "thread_id": thread_id, "prompt_version": PROMPT_VERSION,
        "remaining_tool_budget": 4, "total_tool_budget": 6,
        "preparation_ms": preparation_ms, "model_ms": model_ms,
        "error_type": error_type,
        "seed_messages": [message.model_dump(mode="json") for message in seed],
        "messages": [message.model_dump(mode="json") for message in messages],
        "raw_structured_response": structured.model_dump() if isinstance(structured, GroundedAnswer) else None,
        "final": finalize_agent_result(state, registry, runtime.knowledge_root),
        "token_usage": extract_token_usage(messages), "traces": traces, "events": events,
    }


def summarize(result: dict, points: list[dict], source_path: str) -> dict:
    """运行结束后统计读取覆盖，不向模型提供金标行号。"""
    coverage = set()
    ranges = []
    complete_index = None
    for index, event in enumerate(result["events"]):
        data = event["data"]
        if event["event"] != "tool_completed" or data["name"] != "read" or data["status"] != "success":
            continue
        summary = data["summary"]
        if summary["path"] != source_path:
            continue
        start, end = summary["start_line"], summary["end_line"]
        if start is None or end is None:
            continue
        ranges.append([start, end])
        coverage.update(range(start, end + 1))
        if complete_index is None and all(set(range(p["evidence"]["start_line"], p["evidence"]["end_line"] + 1)) <= coverage for p in points):
            complete_index = index
    model_traces = [trace for trace in result["traces"] if trace["source"] == "model"]
    citations = result["final"].get("citations", [])
    return {
        "thread_id": result["thread_id"], "error_type": result["error_type"],
        "first_read_start": ranges[0][0] if ranges else None,
        "read_ranges": ranges, "model_calls": model_traces,
        "fully_read_answer_point_ids": [p["id"] for p in points if set(range(p["evidence"]["start_line"], p["evidence"]["end_line"] + 1)) <= coverage],
        "calls_after_complete_read": sum(event["event"] == "tool_started" for event in result["events"][complete_index + 1:]) if complete_index is not None else None,
        "answer_type": result["final"]["answer_type"], "citation_count": len(citations),
        "heading_only_citation_count": sum(all(line.lstrip().startswith("#") for line in citation["quote"].splitlines() if line.strip()) for citation in citations),
        "token_usage": result["token_usage"], "model_ms": result["model_ms"],
        "preparation_ms": result["preparation_ms"],
    }


def main():
    root = Path(__file__).resolve().parents[1]
    demo = json.loads((root / "evaluation/demos/patent_review_comparison_001.json").read_text(encoding="utf-8"))
    spec = json.loads((root / "evaluation/pilots/multi_chunk_guide_001.json").read_text(encoding="utf-8"))
    print("加载共享运行时；不重建索引，不修改正式服务", flush=True)
    runtime = build_pilot_agentic_runtime_from_project(root, load_settings_from_env(root))
    source = resolve_knowledge_path(spec["corpus"]["source_path"], runtime.knowledge_root)
    points = [p for p in spec["answer_points"] if p["id"].startswith(("priority-review-", "rapid-pre-review-"))]
    assert len(points) == 8
    directory = root / "data/demo-diagnostics" / f"grep-handoff-{uuid4().hex}"
    metadata = {
        "question": demo["question"], "title": TITLE, "patterns": PATTERNS,
        "prompt": KNOWLEDGE_AGENT_PROMPTS[PROMPT_VERSION], "prompt_version": PROMPT_VERSION,
        "model": runtime.chat_model.model_name, "temperature": runtime.chat_model.temperature,
        "enable_thinking": (runtime.chat_model.extra_body or {}).get("enable_thinking"),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "boundary": "诊断脚本预执行 glob/grep，不代表自主 Agent；准备两次加后续四次，总预算六次。",
    }
    summaries = []
    for number in range(1, 4):
        print(f"RUN {number}", flush=True)
        result = run_handoff(
            runtime, demo["question"], TITLE, PATTERNS,
            thread_id=f"handoff-{number}-{uuid4().hex}",
            on_event=lambda event: print(json.dumps(event, ensure_ascii=False), flush=True),
        )
        summary = {"round": number, **summarize(result, points, spec["corpus"]["source_path"])}
        save_evaluation_result({**metadata, **result, "summary": summary}, directory / f"round-{number}.json")
        summaries.append(summary)
        save_evaluation_result({**metadata, "runs": summaries}, directory / "summary.json")
        print("RESULT " + json.dumps(summary, ensure_ascii=False), flush=True)
    print(f"SAVED {directory}", flush=True)


if __name__ == "__main__":
    main()
