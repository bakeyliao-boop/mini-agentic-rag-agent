"""Bounded tool-contribution calibration; prepare-only unless explicitly authorized.

Run with --help for the local entry point. No configuration or credentials are
loaded on import or in prepare mode. This module never grades semantic answers.
"""

from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import hashlib
import json
import os
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx


MODEL = 'deepseek-flash'
CASE_IDS = tuple(f'CAL-{i:02d}' for i in range(1, 6))
REQUEST_LIMIT = 15
INPUT_RESERVATION = 12000
OUTPUT_LIMIT = 2048
INPUT_PRICE = Decimal('2')
OUTPUT_PRICE = Decimal('8')
RESERVATION = Decimal('0.040384')
TOKENIZER_COMMIT = 'dba1be0a40aa45a94ad051997016db3960a90277'
TOKENIZERS_VERSION = '0.22.2'


class CalibrationError(ValueError):
    """The supplied material, scope, or authorization cannot be used."""


@dataclass(frozen=True)
class RawResponse:
    status_code: int
    body: bytes
    headers: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Bundle:
    manifest: dict[str, Any]
    requests: dict[str, bytes]
    slots: tuple[dict[str, Any], ...]
    source_hashes: dict[Path, str]
    budget: Decimal
    timeout_seconds: float
    authorization_errors: tuple[str, ...]


Sender = Callable[[bytes], Awaitable[RawResponse]]


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2) + '\n').encode('utf-8')


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CalibrationError(message)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_bytes())
    except (ValueError, OSError) as exc:
        raise CalibrationError(f'Cannot read JSON: {path.name}') from exc
    _require(isinstance(value, dict), f'Expected object: {path.name}')
    return value


def _no_reference_locations(value: Any) -> None:
    if isinstance(value, dict):
        _require('source_line' not in value, 'Gold source_line in task_reference')
        for item in value.values():
            _no_reference_locations(item)
    elif isinstance(value, list):
        for item in value:
            _no_reference_locations(item)


def _no_review_metadata(value: Any) -> None:
    if isinstance(value, dict):
        forbidden = {'case_id', 'proposed_expected_judgment', 'reviewer_explanation',
                     'human_review', 'human_label_reviewed', 'actual_evaluator_result'}
        _require(not (forbidden & set(value)), 'Review metadata leaked into nested input')
        for item in value.values():
            _no_review_metadata(item)
    elif isinstance(value, list):
        for item in value:
            _no_review_metadata(item)


def _within(folder: Path, name: Any) -> Path:
    _require(isinstance(name, str) and bool(name), 'Missing request file name')
    path = (folder / name).resolve()
    _require(path.is_relative_to(folder.resolve()), 'Request path escapes packet')
    return path


def load_bundle(material_dir: Path) -> Bundle:
    """Validate actual bytes and projections, without network or credentials."""
    material_dir = material_dir.resolve()
    packet = material_dir / 'request-draft-v1'
    manifest_path = packet / 'manifest.json'
    cases_path = material_dir / 'cases.json'
    rules_path = material_dir / '评估器统一说明草案-v2.md'
    counts_path = material_dir / '本地Token计数结果-v1.json'
    manifest = _read_json(manifest_path)
    cases = _read_json(cases_path)
    counts = _read_json(counts_path)
    rules_raw = rules_path.read_bytes()
    rules_text = rules_raw.decode('utf-8-sig')
    _require(rules_text.count('## 评价规则正文') == 1, 'Ambiguous rules body')
    _require(rules_text.count('## 组织侧使用边界') == 1, 'Missing rules boundary')
    rules = rules_text.split('## 评价规则正文', 1)[1].split('## 组织侧使用边界', 1)[0].strip()
    _require(_sha(cases_path.read_bytes()) == manifest.get('source_cases_sha256'), 'Cases hash changed')
    _require(_sha(rules_raw) == manifest.get('source_rules_file_sha256'), 'Rules file hash changed')
    _require(_sha(rules.encode()) == manifest.get('system_content_sha256'), 'Rules body hash changed')
    _require(cases.get('material_revision') == manifest.get('source_cases_material_revision'), 'Cases revision changed')
    _require(counts.get('source_cases_sha256') == manifest.get('source_cases_sha256'), 'Token report cases mismatch')
    _require(counts.get('model_repository_commit') == TOKENIZER_COMMIT
             and counts.get('download_provenance', {}).get('commit') == TOKENIZER_COMMIT
             and counts.get('tokenizers_version') == TOKENIZERS_VERSION,
             'Tokenizer evidence version mismatch')
    _require(manifest.get('planned_model') == MODEL, 'Model outside approved scope')
    _require(manifest.get('approved_case_count') == 5 and manifest.get('approved_repetitions') == 3,
             'Scope must be five cases, three repetitions')
    _require(manifest.get('approved_request_count') == REQUEST_LIMIT, 'Request count must be 15')
    try:
        budget = Decimal(str(manifest['approved_budget']['max_total']))
        controls = manifest['proposed_client_controls']
        timeout = float(controls['timeout_seconds'])
    except (KeyError, TypeError, ValueError, InvalidOperation) as exc:
        raise CalibrationError('Invalid budget or client controls') from exc
    _require(manifest['approved_budget'].get('currency') == 'CNY', 'Budget currency must be CNY')
    _require(budget.is_finite() and Decimal('0') < budget <= Decimal('1'), 'Budget outside approved ceiling')
    _require(controls.get('concurrency') == 1 and controls.get('max_retries') == 0, 'Require serial, zero-retry execution')
    _require(0 < timeout <= 120, 'Overall timeout outside approved ceiling')
    _require(manifest.get('input_token_reservation_per_request') == INPUT_RESERVATION,
             'Input reservation changed')
    _require(manifest.get('proposed_max_output_tokens') == OUTPUT_LIMIT, 'Output limit changed')

    case_list = cases.get('cases', [])
    _require(isinstance(case_list, list), 'Missing cases')
    _require(Counter(c.get('case_id') for c in case_list) == Counter(CASE_IDS), 'Invalid case IDs')
    case_map = {c['case_id']: c for c in case_list}
    entries = manifest.get('cases', [])
    _require(Counter(e.get('case_id') for e in entries) == Counter(CASE_IDS), 'Invalid packet case IDs')
    count_list = counts.get('cases', [])
    _require(Counter(c.get('case_id') for c in count_list) == Counter(CASE_IDS), 'Invalid token case IDs')
    count_map = {c['case_id']: c for c in count_list}
    source = cases.get('source', {})
    source_name = source.get('local_path', '')
    _require(isinstance(source_name, str) and bool(source_name), 'Missing source path')
    _require(not source_name.replace(chr(92), '/').startswith('//'), 'Network source is forbidden')
    source_candidate = Path(source_name)
    _require(source_candidate.is_absolute()
             and source_candidate.drive.lower() == material_dir.drive.lower(),
             'Source must be on the reviewed local material drive')
    source_path = source_candidate.resolve()
    _require(not str(source_path).replace(chr(92), '/').startswith('//'), 'Network source is forbidden')
    _require(source_path.suffix.lower() == '.md' and source_path.is_file(), 'Expected frozen Markdown source')
    _require(_sha(source_path.read_bytes()) == source.get('sha256') == manifest.get('source_document_sha256'),
             'Frozen document hash changed')
    paths = [manifest_path, cases_path, rules_path, counts_path, source_path]
    requests: dict[str, bytes] = {}
    file_by_case: dict[str, str] = {}
    for entry in entries:
        cid = entry['case_id']
        path = _within(packet, entry.get('request_file'))
        body = path.read_bytes()
        _require(_sha(body) == entry.get('request_file_sha256'), f'{cid}: request hash changed')
        request = _read_json(path)
        expected_controls = {
            'model': MODEL, 'thinking': {'type': 'disabled'}, 'temperature': 0,
            'max_tokens': OUTPUT_LIMIT, 'stream': False, 'response_format': {'type': 'text'},
        }
        _require(set(request) == set(expected_controls) | {'messages'}, f'{cid}: unexpected request fields')
        _require(all(request[k] == v for k, v in expected_controls.items()), f'{cid}: request parameters changed')
        messages = request.get('messages')
        _require(isinstance(messages, list) and len(messages) == 2, f'{cid}: must have exactly two messages')
        _require(messages[0] == {'role': 'system', 'content': rules}, f'{cid}: wrong system message')
        _require(isinstance(messages[1], dict) and set(messages[1]) == {'role', 'content'}
                 and messages[1]['role'] == 'user' and isinstance(messages[1]['content'], str),
                 f'{cid}: wrong user message')
        try:
            input_value = json.loads(messages[1]['content'])
        except ValueError as exc:
            raise CalibrationError(f'{cid}: invalid input JSON') from exc
        original_input = case_map[cid].get('input')
        _require(isinstance(original_input, dict) and input_value == original_input, f'{cid}: input projection changed')
        _require(not ({'case_id', 'proposed_expected_judgment', 'reviewer_explanation',
                       'human_review', 'human_label_reviewed', 'actual_evaluator_result'} & set(input_value)),
                 f'{cid}: review metadata leaked into input')
        _no_review_metadata(input_value)
        _no_reference_locations(input_value.get('task_reference'))
        _require(_sha(messages[1]['content'].encode()) == entry.get('input_object_sha256'), f'{cid}: input hash changed')
        count = count_map[cid]
        _require(count.get('request_sha256') == _sha(body), f'{cid}: tokenizer evidence request mismatch')
        tokens = count.get('encoded_prompt_tokens')
        _require(type(tokens) is int and 0 < tokens <= INPUT_RESERVATION, f'{cid}: input reservation exceeded')
        requests[cid] = body
        file_by_case[cid] = entry['request_file']
        paths.append(path)
    slots = manifest.get('proposed_order_not_execution_records', [])
    _require(isinstance(slots, list) and len(slots) == REQUEST_LIMIT, 'Exactly 15 planned slots required')
    _require(Counter((s.get('case_id'), s.get('repetition')) for s in slots)
             == Counter((cid, rep) for cid in CASE_IDS for rep in range(1, 4)), 'Invalid repetitions')
    for ordinal, slot in enumerate(slots, 1):
        cid = slot['case_id']
        _require(slot.get('ordinal') == ordinal, 'Invalid slot order')
        _require(slot.get('request_file') == file_by_case[cid]
                 and slot.get('request_sha256') == _sha(requests[cid]), 'Slot request mismatch')
    errors = []
    if manifest.get('execution_authorized') is not True:
        errors.append('Execution has not been authorized')
    if manifest.get('parameter_acceptance') != 'accepted':
        errors.append('Parameters have not been accepted')
    batch_id = manifest.get('execution_batch_id')
    if not isinstance(batch_id, str) or not batch_id or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_' for c in batch_id):
        errors.append('Missing approved execution_batch_id')
    approved_output = manifest.get('execution_output_dir')
    if not isinstance(approved_output, str) or not Path(approved_output).is_absolute():
        errors.append('Missing approved absolute execution_output_dir')
    if manifest.get('execution_token_report_sha256') != _sha(counts_path.read_bytes()):
        errors.append('Token report has not been frozen for execution')
    if not all(c.get('human_label_reviewed') is True for c in case_list):
        errors.append('Whole-case human labels are not all signed off')
    return Bundle(manifest, requests, tuple(slots), {p: _sha(p.read_bytes()) for p in paths},
                  budget, timeout, tuple(errors))


def _assert_unchanged(bundle: Bundle) -> None:
    for path, expected in bundle.source_hashes.items():
        _require(_sha(path.read_bytes()) == expected, f'Material changed during batch: {path.name}')


def _write_new(path: Path, data: bytes) -> None:
    with path.open('xb') as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def _journal(folder: Path, event: dict[str, Any]) -> None:
    event = {'at': datetime.now(timezone.utc).isoformat(), **event}
    with (folder / 'journal.jsonl').open('ab') as stream:
        stream.write(json.dumps(event, ensure_ascii=False).encode() + b'\n')
        stream.flush()
        os.fsync(stream.fileno())


def _cost(prompt: int, completion: int) -> Decimal:
    return (Decimal(prompt) * INPUT_PRICE + Decimal(completion) * OUTPUT_PRICE) / Decimal(1_000_000)


def _inspect_response(response: RawResponse) -> dict[str, Any]:
    result: dict[str, Any] = {
        'http_status': response.status_code, 'technical_status': 'ok',
        'accounting_status': 'unknown_reserved', 'content': None, 'finish_reason': None,
        'usage': None, 'cost_computed_cny': None, 'review_status': 'pending_human_review',
    }
    try:
        payload = json.loads(response.body)
        _require(isinstance(payload, dict), 'Invalid response object')
        usage = payload.get('usage')
        result.update(response_id=payload.get('id'), response_model=payload.get('model'), usage=usage)
        if isinstance(usage, dict) and all(type(usage.get(k)) is int and usage[k] >= 0
                                           for k in ('prompt_tokens', 'completion_tokens')):
            result['accounting_status'] = 'computed'
            result['cost_computed_cny'] = str(_cost(usage['prompt_tokens'], usage['completion_tokens']))
            if usage['prompt_tokens'] > INPUT_RESERVATION or usage['completion_tokens'] > OUTPUT_LIMIT:
                result['technical_status'] = 'usage_exceeded_reservation'
        choice = payload['choices'][0]
        result['finish_reason'] = choice.get('finish_reason')
        result['content'] = choice['message'].get('content')
        if not 200 <= response.status_code < 300:
            result['technical_status'] = 'http_error'
        elif result['technical_status'] != 'ok':
            pass
        elif result['finish_reason'] == 'length':
            result['technical_status'] = 'output_truncated'
        elif result['finish_reason'] != 'stop':
            result['technical_status'] = 'unexpected_finish_reason'
        elif not isinstance(result['content'], str) or not result['content'].strip():
            result['technical_status'] = 'missing_evaluation_text'
        elif payload.get('model') != MODEL:
            result['technical_status'] = 'model_requires_review'
        elif choice['message'].get('reasoning_content'):
            result['technical_status'] = 'unexpected_thinking_content'
        elif result['accounting_status'] == 'unknown_reserved':
            result['technical_status'] = 'missing_usage'
    except (ValueError, KeyError, IndexError, TypeError, AttributeError):
        result['technical_status'] = 'invalid_api_response'
    return result


class HttpxSender:
    """One HTTP POST, no redirects/proxy discovery/retries; caller owns deadline."""

    def __init__(self, base_url: str, api_key: str, *, transport: httpx.AsyncBaseTransport | None = None):
        url = urlsplit(base_url)
        _require(url.scheme == 'https' and url.hostname == 'api.deepseek.com'
                 and not url.username and not url.password and not url.query and not url.fragment
                 and url.port in (None, 443) and url.path.rstrip('/') in ('', '/v1'),
                 'Only the reviewed official DeepSeek endpoint is allowed')
        _require(bool(api_key.strip()), 'Missing API credential')
        self._secret = api_key
        self._url = base_url.rstrip('/') + '/chat/completions'
        self._client = httpx.AsyncClient(
            transport=transport if transport is not None else httpx.AsyncHTTPTransport(retries=0),
            follow_redirects=False, trust_env=False, timeout=None,
        )

    async def __call__(self, body: bytes) -> RawResponse:
        response = await self._client.post(self._url, content=body, headers={
            'Authorization': f'Bearer {self._secret}', 'Content-Type': 'application/json',
        })
        headers = {k: v for k, v in response.headers.items() if k.lower() in ('content-type', 'x-request-id')}
        # Do not persist credentials even if a remote error echoes them.
        raw = response.content
        if self._secret.encode() in raw:
            raw = raw.replace(self._secret.encode(), b'[REDACTED]')
            headers['credential-redaction-applied'] = 'true'
        return RawResponse(response.status_code, raw, headers)

    async def aclose(self) -> None:
        await self._client.aclose()


def _load_connection() -> tuple[str, str]:
    # Deliberately lazy: prepare and offline fake transports never read .env.
    from rag_core.settings import _required_setting, load_settings_from_env
    settings = load_settings_from_env(Path(__file__).resolve().parents[1])
    return (_required_setting(settings, 'SEMANTIC_JUDGE_BASE_URL'),
            _required_setting(settings, 'SEMANTIC_JUDGE_API_KEY'))


async def run_batch(material_dir: Path, output_dir: Path, *, execute: bool = False,
                    sender: Sender | None = None) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    bundle = load_bundle(material_dir)
    if execute and bundle.authorization_errors:
        raise CalibrationError('; '.join(bundle.authorization_errors))
    if execute:
        _require(output_dir == Path(bundle.manifest['execution_output_dir']).resolve(),
                 'Output path differs from the single authorized batch directory')
        _require(output_dir.name == bundle.manifest['execution_batch_id'],
                 'Output directory must match the approved batch ID')
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / 'calls').mkdir()
    _write_new(output_dir / 'batch.json', _json_bytes({
        'mode': 'execute' if execute else 'prepare', 'model': MODEL,
        'planned_count': REQUEST_LIMIT, 'budget_cny': str(bundle.budget),
        'client_controls': bundle.manifest['proposed_client_controls'],
        'authorization_errors': list(bundle.authorization_errors),
        'sources': {str(p): h for p, h in bundle.source_hashes.items()},
        'slots': bundle.slots, 'cost_basis': 'peak uncached CNY input=2/output=8 per million',
    }))
    summary: dict[str, Any] = {
        'status': 'prepared', 'planned_count': REQUEST_LIMIT, 'attempted_count': 0,
        'response_count': 0, 'halt_reason': None, 'cost_computed_cny': '0',
        'cost_reserved_unknown_cny': '0', 'cost_kind': 'no_requests_sent',
        'review_status': 'pending_human_review',
        'authorization_errors': list(bundle.authorization_errors),
    }
    _journal(output_dir, {'event': 'batch_prepared'})
    if not execute:
        _write_new(output_dir / 'summary.json', _json_bytes(summary))
        return summary
    owned_sender = None
    computed, reserved = Decimal('0'), Decimal('0')
    try:
        if sender is None:
            base_url, key = _load_connection()
            owned_sender = HttpxSender(base_url, key)
            sender = owned_sender
        for slot in bundle.slots:
            try:
                _assert_unchanged(bundle)
            except (CalibrationError, OSError) as exc:
                summary.update(status='halted', halt_reason='material_changed')
                _journal(output_dir, {'event': 'preflight_refused', 'reason': type(exc).__name__})
                break
            if summary['attempted_count'] >= REQUEST_LIMIT or computed + reserved + RESERVATION > bundle.budget:
                summary.update(status='halted', halt_reason='budget_or_request_limit')
                _journal(output_dir, {'event': 'preflight_refused', 'reason': summary['halt_reason']})
                break
            ordinal = slot['ordinal']
            prefix = output_dir / 'calls' / f'{ordinal:03d}'
            body = bundle.requests[slot['case_id']]
            _write_new(prefix.with_suffix('.request.json'), body)
            reserved += RESERVATION
            _journal(output_dir, {'event': 'attempt_started', 'ordinal': ordinal,
                                 'case_id': slot['case_id'], 'repetition': slot['repetition'],
                                 'request_sha256': _sha(body), 'reserved_cny': str(RESERVATION),
                                 'delivery_confirmation': 'unknown_until_response'})
            summary['attempted_count'] += 1
            try:
                async with asyncio.timeout(bundle.timeout_seconds):
                    response = await sender(body)
            except Exception as exc:
                result = {'technical_status': 'transport_error', 'error_type': type(exc).__name__,
                          'accounting_status': 'unknown_reserved', 'usage': None,
                          'cost_computed_cny': None, 'review_status': 'pending_human_review'}
                _write_new(prefix.with_suffix('.error.json'), _json_bytes(result))
            else:
                summary['response_count'] += 1
                _write_new(prefix.with_suffix('.response.raw'), response.body)
                _write_new(prefix.with_suffix('.response-meta.json'), _json_bytes({
                    'status_code': response.status_code, 'headers': response.headers,
                }))
                result = _inspect_response(response)
                if result['accounting_status'] == 'computed':
                    computed += Decimal(result['cost_computed_cny'])
                    reserved -= RESERVATION
            _write_new(prefix.with_suffix('.result.json'), _json_bytes(result))
            _journal(output_dir, {'event': 'attempt_finished', 'ordinal': ordinal,
                                 'technical_status': result['technical_status'],
                                 'accounting_status': result['accounting_status']})
            if result['technical_status'] != 'ok':
                summary.update(status='halted', halt_reason=result['technical_status'])
                break
        else:
            try:
                _assert_unchanged(bundle)
            except (CalibrationError, OSError):
                summary.update(status='halted', halt_reason='material_changed')
            else:
                summary['status'] = 'completed_pending_review'
    finally:
        # No semantic pass/fail is inferred here; raw evidence remains authoritative.
        summary.update(cost_computed_cny=str(computed), cost_reserved_unknown_cny=str(reserved),
                       cost_kind='computed_with_unknown_reservations' if reserved else 'computed_from_usage')
        if owned_sender is not None:
            await owned_sender.aclose()
    _write_new(output_dir / 'summary.json', _json_bytes(summary))
    _journal(output_dir, {'event': 'batch_finished', 'status': summary['status']})
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--materials', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path, help='New batch directory; never overwritten')
    parser.add_argument('--execute', action='store_true', help='Requires explicit accepted material authorization')
    args = parser.parse_args(argv)
    try:
        result = asyncio.run(run_batch(args.materials, args.output, execute=args.execute))
    except (CalibrationError, OSError) as exc:
        print(f'Calibration refused: {type(exc).__name__}: {exc}')
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 1 if result['status'] == 'halted' else 0


if __name__ == '__main__':
    raise SystemExit(main())
