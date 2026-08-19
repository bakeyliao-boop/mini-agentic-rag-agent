import json
from pathlib import Path

from evaluation.dataset import load_tool_expectations
from evaluation.scorers.tool_expectations import score_tool_expectation


def test_score_tool_expectation_accepts_ideal_directory_flow() -> None:
    """理想的两次 ls 调用应通过 directory-001 的工具流程评分。"""

    expectation = {
        "id": "directory-001",
        "allowed_tools": ["ls"],
        "required_initial_call": {
            "name": "ls",
            "args": {"path": "/"},
            "status": "success",
        },
        "max_tool_calls": 2,
        "completion_call": {
            "name": "ls",
            "args": {"path": "/课程资源"},
            "status": "success",
        },
        "forbid_calls_after_completion": True,
        "require_all_calls_success": True,
    }
    tool_traces = [
        {
            "step": 1,
            "tool_call_id": "call-root",
            "name": "ls",
            "args": {"path": "/"},
            "status": "success",
        },
        {
            "step": 2,
            "tool_call_id": "call-course-resources",
            "name": "ls",
            "args": {"path": "/课程资源"},
            "status": "success",
        },
    ]

    result = score_tool_expectation(expectation, tool_traces)

    assert result == {
        "initial_call_matches": True,
        "allowed_tools_match": True,
        "unexpected_tools": [],
        "tool_call_count": 2,
        "within_tool_call_limit": True,
        "all_calls_successful": True,
        "error_call_count": 0,
        "required_calls_matched": True,
        "missing_required_calls": [],
        "completion_reached": True,
        "calls_after_completion": 0,
        "stopped_after_completion": True,
        "tool_flow_passed": True,
    }


def test_directory_expectation_accepts_direct_target_ls() -> None:
    """直接列出目标目录时，不应强制先调用根目录 ls。"""

    project_root = Path(__file__).parent.parent
    standards = load_tool_expectations(
        project_root / "evaluation" / "tool_expectations.json"
    )
    expectation = next(
        item
        for item in standards["expectations"]
        if item["id"] == "directory-001"
    )
    tool_traces = [
        {
            "step": 1,
            "tool_call_id": "call-course-resources",
            "name": "ls",
            "args": {"path": "/课程资源"},
            "status": "success",
        }
    ]

    result = score_tool_expectation(expectation, tool_traces)

    assert result["allowed_tools_match"] is True
    assert result["tool_call_count"] == 1
    assert result["within_tool_call_limit"] is True
    assert result["all_calls_successful"] is True
    assert result["completion_reached"] is True
    assert result["calls_after_completion"] == 0
    assert result["initial_call_matches"] is True
    assert result["tool_flow_passed"] is True


def test_directory_expectation_accepts_failed_probe_then_recovery() -> None:
    """目录题在全局阈值内从错误 ls 恢复后应通过工具流程评分。"""

    project_root = Path(__file__).parent.parent
    standards = load_tool_expectations(
        project_root / "evaluation" / "tool_expectations.json"
    )
    expectation = next(
        item
        for item in standards["expectations"]
        if item["id"] == "directory-001"
    )
    tool_traces = [
        {
            "step": 1,
            "tool_call_id": "call-unverified-directory",
            "name": "ls",
            "args": {"path": "/课程资源目录"},
            "status": "error",
        },
        {
            "step": 2,
            "tool_call_id": "call-root",
            "name": "ls",
            "args": {"path": "/"},
            "status": "success",
        },
        {
            "step": 3,
            "tool_call_id": "call-course-resources",
            "name": "ls",
            "args": {"path": "/课程资源"},
            "status": "success",
        },
    ]

    result = score_tool_expectation(expectation, tool_traces)

    assert result["allowed_tools_match"] is True
    assert result["tool_call_count"] == 3
    assert result["within_tool_call_limit"] is True
    assert result["all_calls_successful"] is False
    assert result["completion_reached"] is True
    assert result["calls_after_completion"] == 0
    assert result["stopped_after_completion"] is True
    assert result["tool_flow_passed"] is True


def test_score_tool_expectation_reports_error_call_count() -> None:
    """评分结果应单独记录恢复过程中出现了几次错误调用。"""

    expectation = {
        "allowed_tools": ["ls"],
        "required_initial_call": {"name": "ls"},
        "max_tool_calls": 6,
        "completion_call": {
            "name": "ls",
            "args": {"path": "/课程资源"},
            "status": "success",
        },
        "forbid_calls_after_completion": True,
        "require_all_calls_success": False,
    }
    tool_traces = [
        {
            "name": "ls",
            "args": {"path": "/课程资源目录"},
            "status": "error",
        },
        {
            "name": "ls",
            "args": {"path": "/"},
            "status": "success",
        },
        {
            "name": "ls",
            "args": {"path": "/课程资源"},
            "status": "success",
        },
    ]

    result = score_tool_expectation(expectation, tool_traces)

    assert result["error_call_count"] == 1


def test_score_tool_expectation_evaluates_v15_directory_trace() -> None:
    """V1.5 的 directory-001 真实轨迹应暴露冗余工具调用。"""

    project_root = Path(__file__).parent.parent
    standards = load_tool_expectations(
        project_root / "evaluation" / "tool_expectations.json"
    )
    run_result = json.loads(
        (
            project_root
            / "evaluation"
            / "results"
            / "agentic-baseline-qwen3.6-flash-thinking-off-prompt-v1.5.json"
        ).read_text(encoding="utf-8")
    )

    expectation = next(
        item
        for item in standards["expectations"]
        if item["id"] == "directory-001"
    )
    question_result = next(
        item
        for item in run_result["results"]
        if item["id"] == "directory-001"
    )

    result = score_tool_expectation(
        expectation,
        question_result["tool_traces"],
    )

    assert result == {
        "initial_call_matches": False,
        "allowed_tools_match": False,
        "unexpected_tools": ["glob", "search"],
        "tool_call_count": 3,
        "within_tool_call_limit": True,
        "all_calls_successful": True,
        "error_call_count": 0,
        "required_calls_matched": True,
        "missing_required_calls": [],
        "completion_reached": True,
        "calls_after_completion": 0,
        "stopped_after_completion": True,
        "tool_flow_passed": False,
    }


def test_score_tool_expectation_matches_required_calls_in_any_order() -> None:
    """双文档题的两个 read 可以按任意顺序完成。"""

    expectation = {
        "allowed_tools": ["glob", "read"],
        "required_initial_call": {
            "name": "glob",
            "args": {"target": "自动控制系统", "path": "/"},
            "status": "success",
        },
        "required_calls": [
            {
                "name": "read",
                "args": {"path": "/自动控制系统/课程介绍.md"},
                "status": "success",
            },
            {
                "name": "read",
                "args": {"path": "/自动控制系统/智慧农场.md"},
                "status": "success",
            },
        ],
        "max_tool_calls": 3,
        "forbid_calls_after_completion": True,
        "require_all_calls_success": True,
    }
    tool_traces = [
        {
            "name": "glob",
            "args": {
                "target": "自动控制系统",
                "target_type": "directory",
                "path": "/",
            },
            "status": "success",
        },
        {
            "name": "read",
            "args": {
                "path": "/自动控制系统/智慧农场.md",
                "limit": 40,
            },
            "status": "success",
        },
        {
            "name": "read",
            "args": {
                "path": "/自动控制系统/课程介绍.md",
                "start_line": 1,
            },
            "status": "success",
        },
    ]

    result = score_tool_expectation(expectation, tool_traces)

    assert result["initial_call_matches"] is True
    assert result["required_calls_matched"] is True
    assert result["missing_required_calls"] == []
    assert result["completion_reached"] is True
    assert result["calls_after_completion"] == 0
    assert result["tool_flow_passed"] is True


def test_out_of_scope_expectation_rejects_second_search() -> None:
    """本轮冒烟中，第二次 search 应被判定为超出工具预算。"""

    project_root = Path(__file__).parent.parent
    standards = load_tool_expectations(
        project_root / "evaluation" / "tool_expectations.json"
    )
    expectation = next(
        item
        for item in standards["expectations"]
        if item["id"] == "out-of-scope-001"
    )
    tool_traces = [
        {
            "step": 1,
            "tool_call_id": "call-first-search",
            "name": "search",
            "args": {
                "query": "量子计算机的工作原理",
                "path": "/",
                "limit": 5,
            },
            "status": "success",
        },
        {
            "step": 2,
            "tool_call_id": "call-second-search",
            "name": "search",
            "args": {
                "query": "量子计算机",
                "path": "/",
                "limit": 5,
            },
            "status": "success",
        },
    ]

    result = score_tool_expectation(expectation, tool_traces)

    assert result["initial_call_matches"] is True
    assert result["allowed_tools_match"] is True
    assert result["all_calls_successful"] is True
    assert result["tool_call_count"] == 2
    assert result["within_tool_call_limit"] is False
    assert result["tool_flow_passed"] is False


def test_v15_tool_expectations_cover_expected_pass_and_failure_ids() -> None:
    """V1.5 的十题工具流程应稳定得到八项通过和两项失败。"""

    project_root = Path(__file__).parent.parent
    standards = load_tool_expectations(
        project_root / "evaluation" / "tool_expectations.json"
    )
    run_result = json.loads(
        (
            project_root
            / "evaluation"
            / "results"
            / "agentic-baseline-qwen3.6-flash-thinking-off-prompt-v1.5.json"
        ).read_text(encoding="utf-8")
    )
    standards_by_id = {
        item["id"]: item for item in standards["expectations"]
    }

    scores_by_id = {
        item["id"]: score_tool_expectation(
            standards_by_id[item["id"]],
            item["tool_traces"],
        )
        for item in run_result["results"]
    }

    passed_ids = {
        question_id
        for question_id, score in scores_by_id.items()
        if score["tool_flow_passed"]
    }
    failed_ids = set(scores_by_id) - passed_ids

    assert passed_ids == {
        "exact-001",
        "exact-002",
        "exact-003",
        "exact-004",
        "exact-005",
        "exact-006",
        "ambiguity-001",
        "multi-document-001",
    }
    assert failed_ids == {
        "directory-001",
        "out-of-scope-001",
    }
    assert scores_by_id["out-of-scope-001"] == {
        "initial_call_matches": True,
        "allowed_tools_match": False,
        "unexpected_tools": ["glob"],
        "tool_call_count": 7,
        "within_tool_call_limit": False,
        "all_calls_successful": False,
        "error_call_count": 1,
        "required_calls_matched": True,
        "missing_required_calls": [],
        "completion_reached": None,
        "calls_after_completion": None,
        "stopped_after_completion": None,
        "tool_flow_passed": False,
    }
