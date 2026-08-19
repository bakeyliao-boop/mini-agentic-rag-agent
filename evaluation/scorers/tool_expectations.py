"""Agent 工具调用流程的规则评分。"""


def _matches_subset(expected: object, actual: object) -> bool:
    """判断期望值是否为实际值的递归子集。"""

    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            return False
        return all(
            key in actual and _matches_subset(value, actual[key])
            for key, value in expected.items()
        )
    return expected == actual


def _matches_call(
    expected_call: dict[str, object],
    actual_call: dict[str, object],
) -> bool:
    """匹配单个调用规则，并支持多个允许的首调用。"""

    alternatives = expected_call.get("any_of")
    if isinstance(alternatives, list):
        return any(
            isinstance(alternative, dict)
            and _matches_subset(alternative, actual_call)
            for alternative in alternatives
        )
    return _matches_subset(expected_call, actual_call)


def _match_required_calls(
    required_calls: list[dict[str, object]],
    tool_traces: list[dict[str, object]],
) -> tuple[list[int], list[dict[str, object]]]:
    """使用不同轨迹逐一匹配必需调用，调用出现顺序可以不同。"""

    used_indexes: set[int] = set()
    matched_indexes: list[int] = []
    missing_calls: list[dict[str, object]] = []

    for required_call in required_calls:
        matching_index = next(
            (
                index
                for index, trace in enumerate(tool_traces)
                if index not in used_indexes
                and _matches_call(required_call, trace)
            ),
            None,
        )
        if matching_index is None:
            missing_calls.append(required_call)
            continue
        used_indexes.add(matching_index)
        matched_indexes.append(matching_index)

    return matched_indexes, missing_calls


def score_tool_expectation(
    expectation: dict[str, object],
    tool_traces: list[dict[str, object]],
) -> dict[str, object]:
    """按照单题工具标准，检查一次 Agent 的工具调用流程。"""

    required_initial_call = expectation["required_initial_call"]
    initial_call_matches = (
        bool(tool_traces)
        and isinstance(required_initial_call, dict)
        and _matches_call(required_initial_call, tool_traces[0])
    )

    allowed_tools = set(expectation["allowed_tools"])
    unexpected_tools = list(
        dict.fromkeys(
            trace.get("name")
            for trace in tool_traces
            if trace.get("name") not in allowed_tools
        )
    )
    allowed_tools_match = not unexpected_tools

    tool_call_count = len(tool_traces)
    within_tool_call_limit = (
        tool_call_count <= expectation["max_tool_calls"]
    )
    all_calls_successful = all(
        trace.get("status") == "success" for trace in tool_traces
    )
    error_call_count = sum(
        1
        for trace in tool_traces
        if trace.get("status") == "error"
    )
    required_success_matches = (
        all_calls_successful
        if expectation.get("require_all_calls_success", False)
        else True
    )

    raw_required_calls = expectation.get("required_calls", [])
    required_calls = [
        call for call in raw_required_calls if isinstance(call, dict)
    ]
    matched_indexes, missing_required_calls = _match_required_calls(
        required_calls,
        tool_traces,
    )
    required_calls_matched = not missing_required_calls

    completion_index: int | None = None
    completion_defined = False
    completion_call = expectation.get("completion_call")
    if isinstance(completion_call, dict):
        completion_defined = True
        completion_index = next(
            (
                index
                for index, trace in enumerate(tool_traces)
                if _matches_call(completion_call, trace)
            ),
            None,
        )
    elif required_calls:
        completion_defined = True
        if required_calls_matched:
            completion_index = max(matched_indexes)

    completion_reached = (
        completion_index is not None if completion_defined else None
    )
    calls_after_completion = (
        len(tool_traces[completion_index + 1 :])
        if completion_index is not None
        else None
    )
    forbid_calls_after_completion = expectation.get(
        "forbid_calls_after_completion",
        False,
    )
    stopped_after_completion = (
        completion_reached
        and (
            not forbid_calls_after_completion
            or calls_after_completion == 0
        )
        if completion_defined
        else None
    )

    checks = [
        initial_call_matches,
        allowed_tools_match,
        within_tool_call_limit,
        required_success_matches,
        required_calls_matched,
    ]
    if completion_defined:
        checks.extend(
            [
                bool(completion_reached),
                bool(stopped_after_completion),
            ]
        )
    tool_flow_passed = all(checks)

    return {
        "initial_call_matches": initial_call_matches,
        "allowed_tools_match": allowed_tools_match,
        "unexpected_tools": unexpected_tools,
        "tool_call_count": tool_call_count,
        "within_tool_call_limit": within_tool_call_limit,
        "all_calls_successful": all_calls_successful,
        "error_call_count": error_call_count,
        "required_calls_matched": required_calls_matched,
        "missing_required_calls": missing_required_calls,
        "completion_reached": completion_reached,
        "calls_after_completion": calls_after_completion,
        "stopped_after_completion": stopped_after_completion,
        "tool_flow_passed": tool_flow_passed,
    }
