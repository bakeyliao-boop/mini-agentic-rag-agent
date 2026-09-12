"""Offline boundaries for the fixed five-case contribution calibration runner.

Every source, request, manifest and local count report is synthetic and created
under tmp_path. These tests never read the planning worktree or call a model.
"""

import asyncio
import hashlib
import json
import socket
from collections import Counter
from decimal import Decimal
from pathlib import Path

import httpx
import pytest

from evaluation import tool_contribution_calibration as calibration


RESERVATION = Decimal("0.040384")
CASE_IDS = [f"CAL-{number:02d}" for number in range(1, 6)]
RULES_FILE = "评估器统一说明草案-v2.md"
COUNTS_FILE = "本地Token计数结果-v1.json"
SYSTEM = "仅评价单例中实际取得的工具返回，提供可回查的判断依据。"


def _bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: object) -> None:
    path.write_bytes(_bytes(value))


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _request_path(material: Path, case_id: str = "CAL-01") -> Path:
    return material / "request-draft-v1" / f"{case_id}.request.example.json"


def _refresh_hashes(material: Path) -> None:
    """Rebind hashes after intentional fixture mutation, without fixing content."""
    cases_path = material / "cases.json"
    cases = _read_json(cases_path)
    manifest_path = material / "request-draft-v1" / "manifest.json"
    manifest = _read_json(manifest_path)
    manifest["source_cases_sha256"] = _sha(cases_path.read_bytes())
    manifest["source_cases_material_revision"] = cases["material_revision"]
    manifest["source_rules_file_sha256"] = _sha(
        (material / RULES_FILE).read_bytes()
    )
    manifest["source_document_sha256"] = cases["source"]["sha256"]
    for entry in manifest["cases"]:
        request_bytes = _request_path(material, entry["case_id"]).read_bytes()
        request = json.loads(request_bytes)
        entry["request_file_sha256"] = _sha(request_bytes)
        entry["request_file_utf8_bytes"] = len(request_bytes)
        entry["input_object_sha256"] = _sha(
            request["messages"][1]["content"].encode("utf-8")
        )
    by_case = {entry["case_id"]: entry for entry in manifest["cases"]}
    for slot in manifest["proposed_order_not_execution_records"]:
        slot["request_sha256"] = by_case[slot["case_id"]]["request_file_sha256"]
    _write_json(manifest_path, manifest)
    report_path = material / COUNTS_FILE
    report = _read_json(report_path)
    report["source_cases_sha256"] = _sha(cases_path.read_bytes())
    report["source_request_manifest_sha256"] = _sha(manifest_path.read_bytes())
    report["source_cases_material_revision"] = cases["material_revision"]
    for entry in report["cases"]:
        entry["request_sha256"] = by_case[entry["case_id"]]["request_file_sha256"]
    _write_json(report_path, report)


@pytest.fixture
def material(tmp_path: Path) -> Path:
    material_dir = tmp_path / "synthetic-material"
    requests_dir = material_dir / "request-draft-v1"
    requests_dir.mkdir(parents=True)
    source = material_dir / "synthetic-source.md"
    source.write_text("提交后一个月内缴费。\n", encoding="utf-8")
    source_sha = _sha(source.read_bytes())
    (material_dir / RULES_FILE).write_text(
        "# 离线规则夹具\n\n## 评价规则正文\n\n"
        + SYSTEM
        + "\n\n## 组织侧使用边界\n\n此段不得发送。\n",
        encoding="utf-8",
    )
    cases = []
    entries = []
    for number, case_id in enumerate(CASE_IDS, start=1):
        case_input = {
            "task_question": f"合成问题{number}：何时缴费？",
            "task_instruction": "仅根据实际返回回答。",
            "task_reference": {
                "necessary_facts": [
                    {
                        "fact_id": "F1",
                        "meaning": "提交后一个月内缴费。",
                        "reference_excerpt": "提交后一个月内缴费。",
                        "source_id": "SYNTHETIC",
                    }
                ]
            },
            "history_prefix_complete": True,
            "prior_events": [],
            "current_event": {
                "event_id": f"synthetic-event-{number}",
                "tool_name": "read",
                "execution_status": "success",
                "returned_content": {
                    "path": str(source),
                    "source_sha256": source_sha,
                    "returned_lines": [{"line": 1, "text": "提交后一个月内缴费。"}],
                },
                "citable_read_evidence_registered": True,
            },
        }
        cases.append(
            {
                "case_id": case_id,
                "input": case_input,
                "proposed_expected_judgment": {
                    "contribution": f"ORGANIZER_ONLY_GOLD_{number}"
                },
                "reviewer_explanation": "组织侧独立参照，不能进入请求。",
                "human_label_reviewed": True,
                "actual_evaluator_result": None,
            }
        )
        user_content = json.dumps(case_input, ensure_ascii=False, separators=(",", ":"))
        request = {
            "model": "deepseek-flash",
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": user_content},
            ],
            "thinking": {"type": "disabled"},
            "temperature": 0,
            "max_tokens": 2048,
            "stream": False,
            "response_format": {"type": "text"},
        }
        _write_json(_request_path(material_dir, case_id), request)
        entries.append(
            {
                "case_id": case_id,
                "request_file": f"{case_id}.request.example.json",
                "planned_repetitions": 3,
            }
        )
    _write_json(
        material_dir / "cases.json",
        {
            "schema": "planning-evaluator-calibration-v1",
            "material_revision": 7,
            "model": "deepseek-flash",
            "repetitions": 3,
            "planned_request_count": 15,
            "approved_case_ids": CASE_IDS,
            "source": {"local_path": str(source), "sha256": source_sha},
            "cases": cases,
        },
    )
    order = [
        {
            "ordinal": (repetition - 1) * 5 + number,
            "case_id": case_id,
            "repetition": repetition,
            "request_file": f"{case_id}.request.example.json",
        }
        for repetition in range(1, 4)
        for number, case_id in enumerate(CASE_IDS, start=1)
    ]
    _write_json(
        requests_dir / "manifest.json",
        {
            "execution_authorized": True,
            "parameter_acceptance": "accepted",
            "execution_batch_id": "synthetic-offline-batch",
            "execution_output_dir": "<fixture-output-directory>",
            "execution_token_report_sha256": "<fixture-token-report-sha256>",
            "planned_model": "deepseek-flash",
            "approved_case_count": 5,
            "approved_repetitions": 3,
            "approved_request_count": 15,
            "approved_budget": {"currency": "CNY", "max_total": 1},
            "proposed_client_controls": {
                "concurrency": 1,
                "max_retries": 0,
                "timeout_seconds": 120,
            },
            "input_token_reservation_per_request": 12000,
            "proposed_max_output_tokens": 2048,
            "system_content_sha256": _sha(SYSTEM.encode("utf-8")),
            "cases": entries,
            "proposed_order_not_execution_records": order,
        },
    )
    _write_json(
        material_dir / COUNTS_FILE,
        {
            "kind": "local_token_count_not_api_usage",
            # The version declaration follows the approved encoding; counts below
            # are synthetic test inputs, not observations from that tokenizer.
            "model_repository_commit": "dba1be0a40aa45a94ad051997016db3960a90277",
            "download_provenance": {
                "commit": "dba1be0a40aa45a94ad051997016db3960a90277"
            },
            "tokenizers_version": "0.22.2",
            "api_requests_sent": 0,
            "actual_api_usage": None,
            "api_input_count_verified": False,
            "cases": [
                {
                    "case_id": case_id,
                    "encoded_prompt_tokens": 100,
                    "input_reservation": 12000,
                    "planned_repetitions": 3,
                }
                for case_id in CASE_IDS
            ],
            "cost_basis": {
                "currency": "CNY",
                "input_peak_uncached_per_million": "2",
                "output_peak_per_million": "8",
                "accepted_total_budget_cny": "1",
            },
        },
    )
    _refresh_hashes(material_dir)
    return material_dir


def _response(
    *, usage=True, content="离线假回复，仅用于传输边界验证。", finish_reason="stop"
):
    payload = {
        "id": "offline-response-id",
        "model": "deepseek-flash",
        "choices": [
            {"finish_reason": finish_reason, "message": {"content": content}}
        ],
    }
    if usage:
        payload["usage"] = {"prompt_tokens": 100, "completion_tokens": 20}
    return calibration.RawResponse(
        status_code=200,
        body=_bytes(payload),
        headers={"x-request-id": "offline-request-id"},
    )


def _run(material: Path, output: Path, sender=None, *, execute=True):
    if execute:
        manifest_path = material / "request-draft-v1" / "manifest.json"
        manifest = _read_json(manifest_path)
        changed = False
        if manifest.get("execution_batch_id") == "synthetic-offline-batch":
            manifest["execution_batch_id"] = output.name
            changed = True
        if manifest.get("execution_output_dir") == "<fixture-output-directory>":
            manifest["execution_output_dir"] = str(output.resolve())
            changed = True
        if manifest.get("execution_token_report_sha256") == "<fixture-token-report-sha256>":
            manifest["execution_token_report_sha256"] = _sha(
                (material / COUNTS_FILE).read_bytes()
            )
            changed = True
        if changed:
            _write_json(manifest_path, manifest)
    return asyncio.run(
        calibration.run_batch(material, output, execute=execute, sender=sender)
    )


def _files_snapshot(directory: Path):
    return {
        path.relative_to(directory).as_posix(): path.read_bytes()
        for path in directory.rglob("*")
        if path.is_file()
    }


def test_default_prepare_does_not_send_or_load_connection(
    material, tmp_path, monkeypatch
):
    before = _files_snapshot(material)
    manifest_path = material / "request-draft-v1" / "manifest.json"
    manifest = _read_json(manifest_path)
    manifest["execution_authorized"] = False
    manifest["parameter_acceptance"] = "proposed_not_yet_confirmed"
    _write_json(manifest_path, manifest)
    before = _files_snapshot(material)

    async def forbidden_sender(body):
        pytest.fail("prepare must not send")

    def forbidden_connection(*args, **kwargs):
        pytest.fail("prepare must not load secrets or establish a connection")

    monkeypatch.setattr(calibration, "_load_connection", forbidden_connection)
    monkeypatch.setattr(socket, "create_connection", forbidden_connection)
    output = tmp_path / "prepare-only"
    summary = _run(material, output, forbidden_sender, execute=False)
    assert summary["status"] == "prepared"
    assert summary["planned_count"] == 15
    assert summary["attempted_count"] == summary["response_count"] == 0
    assert summary["authorization_errors"]
    assert not list((output / "calls").glob("*.response.raw"))
    assert _files_snapshot(material) == before


@pytest.mark.parametrize(
    "contamination",
    ["gold", "whole_case", "other_case", "history", "tools", "changed_system"],
)
def test_rehashed_request_contamination_is_rejected_before_send(
    material, tmp_path, contamination
):
    request_path = _request_path(material)
    request = _read_json(request_path)
    if contamination == "gold":
        user_input = json.loads(request["messages"][1]["content"])
        user_input["proposed_expected_judgment"] = {"contribution": "LEAKED_GOLD"}
        request["messages"][1]["content"] = json.dumps(user_input)
    elif contamination == "whole_case":
        request["messages"][1]["content"] = json.dumps(
            _read_json(material / "cases.json")["cases"][0]
        )
    elif contamination == "other_case":
        request["messages"][1]["content"] = _read_json(
            _request_path(material, "CAL-02")
        )["messages"][1]["content"]
    elif contamination == "history":
        request["messages"].append(
            {"role": "assistant", "content": "HISTORICAL_EVALUATOR_RESPONSE"}
        )
    elif contamination == "tools":
        request["tools"] = [{"type": "function", "function": {"name": "read"}}]
    else:
        request["messages"][0]["content"] += "\n组织侧金标：充分。"
    _write_json(request_path, request)
    _refresh_hashes(material)
    calls = []

    async def sender(body):
        calls.append(body)
        return _response()

    with pytest.raises(calibration.CalibrationError):
        _run(material, tmp_path / contamination, sender)
    assert calls == []


@pytest.mark.parametrize("field", ["proposed_expected_judgment", "source_line"])
def test_rehashed_source_input_cannot_smuggle_organizer_fields(
    material, tmp_path, field
):
    cases_path = material / "cases.json"
    cases = _read_json(cases_path)
    target = cases["cases"][0]["input"]["task_reference"]
    target[field] = "ORGANIZER_ONLY"
    _write_json(cases_path, cases)
    request_path = _request_path(material)
    request = _read_json(request_path)
    request["messages"][1]["content"] = json.dumps(cases["cases"][0]["input"])
    _write_json(request_path, request)
    _refresh_hashes(material)
    calls = []

    async def sender(body):
        calls.append(body)
        return _response()

    with pytest.raises(calibration.CalibrationError):
        _run(material, tmp_path / field, sender)
    assert not calls


@pytest.mark.parametrize("changed_file", ["request", "rules", "cases", "source"])
def test_changed_material_bytes_are_rejected(material, tmp_path, changed_file):
    paths = {
        "request": _request_path(material),
        "rules": material / RULES_FILE,
        "cases": material / "cases.json",
        "source": material / "synthetic-source.md",
    }
    path = paths[changed_file]
    path.write_bytes(path.read_bytes() + b"\n")
    calls = []

    async def sender(body):
        calls.append(body)
        return _response()

    with pytest.raises(calibration.CalibrationError):
        _run(material, tmp_path / changed_file, sender)
    assert not calls


@pytest.mark.parametrize("missing_approval", ["execution", "parameters", "human"])
def test_execute_requires_review_and_batch_authorization(
    material, tmp_path, missing_approval
):
    manifest_path = material / "request-draft-v1" / "manifest.json"
    manifest = _read_json(manifest_path)
    if missing_approval == "execution":
        manifest["execution_authorized"] = False
    elif missing_approval == "parameters":
        manifest["parameter_acceptance"] = "proposed_not_yet_confirmed"
    else:
        cases_path = material / "cases.json"
        cases = _read_json(cases_path)
        cases["cases"][1]["human_label_reviewed"] = False
        _write_json(cases_path, cases)
    _write_json(manifest_path, manifest)
    _refresh_hashes(material)
    calls = []

    async def sender(body):
        calls.append(body)
        return _response()

    with pytest.raises(calibration.CalibrationError):
        _run(material, tmp_path / missing_approval, sender)
    assert not calls


def test_fifteen_slots_send_exact_bytes_serially_and_persist_before_send(
    material, tmp_path
):
    output = tmp_path / "fifteen"
    expected = {
        _request_path(material, case_id).read_bytes() for case_id in CASE_IDS
    }
    calls = []
    in_flight = 0

    async def sender(body):
        nonlocal in_flight
        in_flight += 1
        assert in_flight == 1
        calls.append(body)
        ordinal = len(calls)
        assert (output / "calls" / f"{ordinal:03d}.request.json").read_bytes() == body
        journal = (output / "journal.jsonl").read_text(encoding="utf-8")
        assert "0.040384" in journal
        assert body in expected
        assert b"ORGANIZER_ONLY_GOLD" not in body
        await asyncio.sleep(0)
        in_flight -= 1
        return _response()

    summary = _run(material, output, sender)
    assert len(calls) == 15
    assert set(Counter(calls).values()) == {3}
    assert summary["planned_count"] == summary["attempted_count"] == 15
    assert summary["response_count"] == 15
    assert summary["status"] == "completed_pending_review"
    assert summary["review_status"] == "pending_human_review"
    assert "passed" not in summary
    assert Decimal(summary["cost_computed_cny"]) == Decimal("0.0054")
    assert Decimal(summary["cost_reserved_unknown_cny"]) == 0
    assert len(list((output / "calls").glob("*.response.raw"))) == 15
    assert _read_json(output / "summary.json") == summary


def test_sixteenth_planned_slot_is_rejected(material, tmp_path):
    manifest_path = material / "request-draft-v1" / "manifest.json"
    manifest = _read_json(manifest_path)
    extra = dict(manifest["proposed_order_not_execution_records"][-1])
    extra["ordinal"] = 16
    manifest["proposed_order_not_execution_records"].append(extra)
    _write_json(manifest_path, manifest)
    calls = []

    async def sender(body):
        calls.append(body)
        return _response()

    with pytest.raises(calibration.CalibrationError):
        _run(material, tmp_path / "sixteen", sender)
    assert not calls


def test_insufficient_budget_prevents_a_request(material, tmp_path):
    manifest_path = material / "request-draft-v1" / "manifest.json"
    manifest = _read_json(manifest_path)
    manifest["approved_budget"]["max_total"] = "0.040383"
    _write_json(manifest_path, manifest)
    calls = []

    async def sender(body):
        calls.append(body)
        return _response()

    summary = _run(material, tmp_path / "under-reservation", sender)
    assert calls == []
    assert summary["status"] == "halted"
    assert summary["attempted_count"] == 0
    assert summary["halt_reason"]


def test_existing_batch_is_never_overwritten_or_resumed(material, tmp_path):
    output = tmp_path / "existing"
    _run(material, output, execute=False)
    before = _files_snapshot(output)
    calls = []

    async def sender(body):
        calls.append(body)
        return _response()

    with pytest.raises(FileExistsError):
        _run(material, output, sender)
    assert _files_snapshot(output) == before
    assert not calls


@pytest.mark.parametrize("failure", ["timeout", "missing_usage", "invalid_json"])
def test_uncertain_attempt_is_not_retried_and_keeps_full_reservation(
    material, tmp_path, failure
):
    output = tmp_path / failure
    calls = []
    raw = b"{this is the complete received body, but invalid JSON"

    async def sender(body):
        calls.append(body)
        if failure == "timeout":
            raise TimeoutError("synthetic overall deadline expired")
        if failure == "missing_usage":
            return _response(usage=False)
        return calibration.RawResponse(status_code=200, body=raw)

    summary = _run(material, output, sender)
    assert len(calls) == 1
    assert summary["attempted_count"] == 1
    assert summary["status"] == "halted"
    assert Decimal(summary["cost_reserved_unknown_cny"]) == RESERVATION
    result = _read_json(output / "calls" / "001.result.json")
    assert result["accounting_status"] == "unknown_reserved"
    if failure != "timeout":
        assert summary["response_count"] == 1
        saved_raw = (output / "calls" / "001.response.raw").read_bytes()
        assert saved_raw == (raw if failure == "invalid_json" else _response(usage=False).body)
    else:
        assert summary["response_count"] == 0
        assert "timeout" in json.dumps(result).lower()
    assert not (output / "calls" / "002.request.json").exists()


def test_raw_response_is_saved_before_optional_interpretation(
    material, tmp_path, monkeypatch
):
    output = tmp_path / "raw-first"
    raw = b"invalid JSON but recoverable original response"
    writes = []
    original = calibration._write_new

    def record_write(path, data):
        result = original(path, data)
        writes.append(Path(path).name)
        return result

    monkeypatch.setattr(calibration, "_write_new", record_write)

    async def sender(body):
        return calibration.RawResponse(status_code=200, body=raw)

    summary = _run(material, output, sender)
    assert summary["status"] == "halted"
    assert (output / "calls" / "001.response.raw").read_bytes() == raw
    assert writes.index("001.response.raw") < writes.index("001.result.json")


def test_truncated_free_text_retains_content_and_finish_reason(material, tmp_path):
    output = tmp_path / "truncated"
    text = "当前支持充分；但实际返回只有标题。后续说明尚未完成"
    calls = []

    async def sender(body):
        calls.append(body)
        return _response(content=text, finish_reason="length")

    summary = _run(material, output, sender)
    assert len(calls) == 1
    assert summary["status"] == "halted"
    result = _read_json(output / "calls" / "001.result.json")
    assert result["content"] == text
    assert result["finish_reason"] == "length"
    assert "passed" not in result
    assert (output / "calls" / "001.response.raw").read_bytes() == _response(
        content=text, finish_reason="length"
    ).body


@pytest.mark.parametrize("failed_suffix", [".request.json", ".response.raw"])
def test_persistence_failure_prevents_following_sends(
    material, tmp_path, monkeypatch, failed_suffix
):
    output = tmp_path / ("save-request" if failed_suffix == ".request.json" else "save-raw")
    original = calibration._write_new
    calls = []

    def fail_selected_write(path, data):
        if str(path).endswith(failed_suffix):
            raise OSError("synthetic disk failure")
        return original(path, data)

    monkeypatch.setattr(calibration, "_write_new", fail_selected_write)

    async def sender(body):
        calls.append(body)
        return _response()

    try:
        summary = _run(material, output, sender)
    except OSError:
        pass
    else:
        assert summary["status"] == "halted"
    assert len(calls) == (0 if failed_suffix == ".request.json" else 1)
    assert output.is_dir()
    before = _files_snapshot(output)
    with pytest.raises(FileExistsError):
        _run(material, output, sender)
    assert _files_snapshot(output) == before
    assert len(calls) == (0 if failed_suffix == ".request.json" else 1)


def test_httpx_sender_sends_bytes_and_does_not_follow_redirects():
    calls = []
    body = b'{"model":"deepseek-flash","unchanged":"request bytes"}'

    async def handler(request):
        calls.append(request)
        return httpx.Response(
            307,
            headers={"location": "https://must-not-follow.invalid/chat/completions"},
            content=b"redirect response",
        )

    async def exercise():
        sender = calibration.HttpxSender(
            "https://api.deepseek.com",
            "synthetic-placeholder",
            transport=httpx.MockTransport(handler),
        )
        try:
            return await sender(body)
        finally:
            await sender.aclose()

    response = asyncio.run(exercise())
    assert len(calls) == 1
    assert calls[0].content == body
    assert response.status_code == 307
    assert response.body == b"redirect response"
    assert "authorization" not in {name.lower() for name in response.headers}


def test_whole_deadline_cancels_slow_chunked_response(material, tmp_path):
    manifest_path = material / "request-draft-v1" / "manifest.json"
    manifest = _read_json(manifest_path)
    manifest["proposed_client_controls"]["timeout_seconds"] = 0.3
    manifest["execution_batch_id"] = "whole-deadline"
    manifest["execution_output_dir"] = str((tmp_path / "whole-deadline").resolve())
    manifest["execution_token_report_sha256"] = _sha(
        (material / COUNTS_FILE).read_bytes()
    )
    _write_json(manifest_path, manifest)
    calls = []
    chunks_seen = []
    cancelled = []
    closed = []

    class SlowStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            chunks_seen.append(0)
            yield b" "
            try:
                # The first chunk is immediate; only the whole-request deadline
                # can end this deliberately unfinished response.
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancelled.append(True)
                raise

        async def aclose(self):
            closed.append(True)

    async def handler(request):
        calls.append(request)
        return httpx.Response(200, stream=SlowStream())

    async def exercise():
        sender = calibration.HttpxSender(
            "https://api.deepseek.com",
            "synthetic-placeholder",
            transport=httpx.MockTransport(handler),
        )
        try:
            return await calibration.run_batch(
                material, tmp_path / "whole-deadline", execute=True, sender=sender
            )
        finally:
            await sender.aclose()

    summary = asyncio.run(exercise())
    assert len(calls) == 1
    assert chunks_seen == [0]
    assert cancelled
    assert closed
    assert summary["status"] == "halted"
    assert summary["attempted_count"] == 1
    assert Decimal(summary["cost_reserved_unknown_cny"]) == RESERVATION


def test_computed_cost_is_counted_before_next_reservation(material, tmp_path):
    manifest_path = material / "request-draft-v1" / "manifest.json"
    manifest = _read_json(manifest_path)
    manifest["approved_budget"]["max_total"] = "0.0405"
    _write_json(manifest_path, manifest)
    calls = []

    async def sender(body):
        calls.append(body)
        return _response()

    summary = _run(material, tmp_path / "settled-cost-budget", sender)
    assert len(calls) == 1
    assert summary["attempted_count"] == 1
    assert summary["response_count"] == 1
    assert summary["status"] == "halted"
    assert Decimal(summary["cost_computed_cny"]) == Decimal("0.00036")
    assert Decimal(summary["cost_reserved_unknown_cny"]) == 0
    assert summary["halt_reason"]


def test_material_change_after_first_response_halts_following_requests(
    material, tmp_path
):
    calls = []

    async def sender(body):
        calls.append(body)
        source = material / "synthetic-source.md"
        source.write_bytes(source.read_bytes() + b"changed during the batch\n")
        return _response()

    summary = _run(material, tmp_path / "mid-batch-drift", sender)
    assert len(calls) == 1
    assert summary["attempted_count"] == summary["response_count"] == 1
    assert summary["status"] == "halted"
    assert summary["halt_reason"]
    assert (tmp_path / "mid-batch-drift" / "calls" / "001.response.raw").exists()


def test_local_input_count_above_reservation_is_rejected(material, tmp_path):
    report_path = material / COUNTS_FILE
    report = _read_json(report_path)
    report["cases"][0]["encoded_prompt_tokens"] = 12001
    _write_json(report_path, report)
    calls = []

    async def sender(body):
        calls.append(body)
        return _response()

    with pytest.raises(calibration.CalibrationError):
        _run(material, tmp_path / "input-over-bound", sender)
    assert calls == []


@pytest.mark.parametrize("batch_id", ["another-approved-batch", None, "../invalid"])
def test_missing_invalid_or_mismatched_batch_id_is_rejected(
    material, tmp_path, batch_id
):
    manifest_path = material / "request-draft-v1" / "manifest.json"
    manifest = _read_json(manifest_path)
    manifest["execution_batch_id"] = batch_id
    _write_json(manifest_path, manifest)
    calls = []

    async def sender(body):
        calls.append(body)
        return _response()

    with pytest.raises(calibration.CalibrationError):
        _run(material, tmp_path / "requested-batch", sender)
    assert calls == []


def test_same_batch_id_cannot_be_replayed_in_another_parent(material, tmp_path):
    calls = []

    async def sender(body):
        calls.append(body)
        return _response()

    approved_output = tmp_path / "approved-parent" / "fixed-batch"
    summary = _run(material, approved_output, sender)
    assert summary["attempted_count"] == len(calls) == 15
    original = _files_snapshot(approved_output)
    other_output = tmp_path / "other-parent" / "fixed-batch"
    with pytest.raises(calibration.CalibrationError):
        _run(material, other_output, sender)
    assert len(calls) == 15
    assert not other_output.exists()
    assert _files_snapshot(approved_output) == original


@pytest.mark.parametrize(
    "field,value",
    [
        ("execution_output_dir", None),
        ("execution_output_dir", "relative-output-directory"),
        ("execution_token_report_sha256", None),
        ("execution_token_report_sha256", "0" * 64),
    ],
)
def test_execution_requires_frozen_output_and_token_report(
    material, tmp_path, field, value
):
    manifest_path = material / "request-draft-v1" / "manifest.json"
    manifest = _read_json(manifest_path)
    manifest[field] = value
    _write_json(manifest_path, manifest)
    calls = []

    async def sender(body):
        calls.append(body)
        return _response()

    with pytest.raises(calibration.CalibrationError):
        _run(material, tmp_path / "frozen-authorization", sender)
    assert calls == []


def test_wrong_fixed_tokenizer_commit_is_rejected(material, tmp_path):
    report_path = material / COUNTS_FILE
    report = _read_json(report_path)
    report["model_repository_commit"] = "a" * 40
    report["download_provenance"]["commit"] = "a" * 40
    _write_json(report_path, report)
    calls = []

    async def sender(body):
        calls.append(body)
        return _response()

    with pytest.raises(calibration.CalibrationError):
        _run(material, tmp_path / "wrong-tokenizer-commit", sender)
    assert calls == []


def test_unc_source_is_refused_before_resolve_or_filesystem_probe(
    material, tmp_path, monkeypatch
):
    cases_path = material / "cases.json"
    cases = _read_json(cases_path)
    cases["source"]["local_path"] = r"\\untrusted.invalid\share\source.md"
    _write_json(cases_path, cases)
    _refresh_hashes(material)
    probes = []
    calls = []

    for method_name in ("resolve", "is_file", "stat"):
        original = getattr(Path, method_name)

        def guarded_method(
            path, *args, _original=original, _method=method_name, **kwargs
        ):
            if str(path).replace("/", "\\").startswith("\\\\"):
                probes.append(_method)
                pytest.fail(f"UNC source reached Path.{_method} before rejection")
            return _original(path, *args, **kwargs)

        monkeypatch.setattr(Path, method_name, guarded_method)

    async def sender(body):
        calls.append(body)
        return _response()

    with pytest.raises(calibration.CalibrationError):
        _run(material, tmp_path / "unc-source", sender)
    assert probes == []
    assert calls == []
