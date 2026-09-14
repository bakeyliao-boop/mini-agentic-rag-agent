"""Bounded transport and index preparation for the supervised one-question trial."""

from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
import math
from pathlib import Path
import os
import shutil
from threading import RLock
import time

import httpx


class TrialHalted(RuntimeError):
    pass


def write_new(path, data):
    with Path(path).open('xb') as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2) + '\n').encode()


def tree_hashes(root):
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(Path(root).rglob('*')) if p.is_file()}


def copy_checked_index(source, destination):
    """Refuse a live-changing source; never open original Chroma for writing."""
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if destination.exists() or destination.is_relative_to(source):
        raise TrialHalted('Index destination must be new and outside source')
    before = tree_hashes(source)
    if 'chroma.sqlite3' not in before:
        raise TrialHalted('Missing source index')
    shutil.copytree(source, destination)
    if before != tree_hashes(source) or before != tree_hashes(destination):
        raise TrialHalted('Source changed or index copy differs; do not execute')
    return before


def compare_vectors(old, new):
    """Tolerance fixed before the compatibility request; no post-result tuning."""
    if len(old) != 1024 or len(new) != 1024:
        return {'compatible': False, 'reason': 'dimension_mismatch'}
    if not all(math.isfinite(x) for x in [*old, *new]):
        return {'compatible': False, 'reason': 'nonfinite_vector'}
    norm = math.sqrt(sum(x*x for x in old) * sum(x*x for x in new))
    cosine = sum(x*y for x, y in zip(old, new)) / norm if norm else 0
    delta = max(abs(x-y) for x, y in zip(old, new))
    return {'compatible': cosine >= .99999 and delta <= .00001,
            'cosine': cosine, 'max_abs_difference': delta,
            'minimum_cosine': .99999, 'maximum_abs_difference': .00001}


class BudgetTransport(httpx.BaseTransport):
    """Capture actual HTTP bodies, reserving cost durably before any send.

    HTTP clients/SDKs must also use zero retries and no redirects. Only this
    synchronous transport is supplied to the synchronous experimental runtime.
    """

    LIMITS = {'chat': 8, 'embedding': 7}
    RESERVE = {'chat': Decimal('1.2098304'), 'embedding': Decimal('.004096')}

    def __init__(self, directory, delegate=None, *, deadline_seconds=600):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.delegate = delegate if delegate is not None else httpx.HTTPTransport(retries=0)
        self.lock = RLock()
        self.counts = {'chat': 0, 'embedding': 0}
        self.reserved = Decimal(0)
        self.computed = Decimal(0)
        self.halted = False
        self.deadline = time.monotonic() + deadline_seconds

    def _journal(self, event, **data):
        with (self.directory/'journal.jsonl').open('a', encoding='utf-8') as f:
            f.write(json.dumps({'at': datetime.now(timezone.utc).isoformat(),
                                'event': event, **data}) + '\n')
            f.flush()
            os.fsync(f.fileno())

    def _validate(self, request, body):
        url = request.url
        if (request.method != 'POST' or url.scheme != 'https'
                or url.host != 'dashscope.aliyuncs.com' or url.port not in (None, 443)
                or url.query or url.userinfo):
            raise TrialHalted('Endpoint outside approved scope')
        if url.path == '/compatible-mode/v1/chat/completions':
            kind = 'chat'
            if (body.get('model') != 'qwen3.7-flash' or body.get('temperature') != 0
                    or body.get('enable_thinking') is not False or body.get('stream') is not False
                    or body.get('max_tokens', body.get('max_completion_tokens')) != 2048
                    or body.get('n', 1) != 1):
                raise TrialHalted('Chat parameters differ from approved scope')
            if any(k in body for k in ('enable_search', 'web_search', 'modalities', 'audio')):
                raise TrialHalted('Extra model capabilities are not approved')
        elif url.path == '/compatible-mode/v1/embeddings':
            kind = 'embedding'
            inputs = body.get('input')
            if (body.get('model') != 'text-embedding-v4' or body.get('dimensions') != 1024
                    or not isinstance(inputs, list) or len(inputs) != 1
                    or not isinstance(inputs[0], str) or len(inputs[0].encode()) > 8192):
                raise TrialHalted('Embedding scope or conservative byte cap exceeded')
        else:
            raise TrialHalted('Endpoint path outside approved scope')
        return kind

    def handle_request(self, request):
        raw_request = request.read()
        body = json.loads(raw_request)
        with self.lock:
            kind = self._validate(request, body)
            reserve = self.RESERVE[kind]
            if (self.halted or time.monotonic() >= self.deadline
                    or self.counts[kind] >= self.LIMITS[kind]
                    or self.computed + self.reserved + reserve > 10):
                self.halted = True
                raise TrialHalted('Trial cost, request count, or deadline reached')
            ordinal = sum(self.counts.values()) + 1
            self.counts[kind] += 1
            self.reserved += reserve
            prefix = self.directory / f'{ordinal:03d}'
            # No authorization headers or serialized credentials are persisted.
            try:
                write_new(prefix.with_suffix('.request.json'), raw_request)
                self._journal('attempt_started', ordinal=ordinal, kind=kind,
                              reserve_cny=str(reserve), request_sha256=hashlib.sha256(raw_request).hexdigest())
            except BaseException:
                self.halted = True
                raise
        response = None
        try:
            request.extensions['timeout'] = {k: min(120, max(.001, self.deadline-time.monotonic()))
                                             for k in ('connect', 'read', 'write', 'pool')}
            response = self.delegate.handle_request(request)
            raw_response = response.read()
            write_new(prefix.with_suffix('.response.raw'), raw_response)
            write_new(prefix.with_suffix('.meta.json'), json_bytes({'http_status': response.status_code}))
            if response.status_code != 200:
                raise TrialHalted('Non-success response; no automatic retry')
            payload = json.loads(raw_response)
            usage = payload.get('usage')
            if not isinstance(usage, dict):
                raise TrialHalted('Missing usage; reservation remains unknown')
            inputs = usage.get('prompt_tokens', usage.get('total_tokens') if kind == 'embedding' else None)
            outputs = usage.get('completion_tokens', 0 if kind == 'embedding' else None)
            if any(type(x) is not int or x < 0 for x in (inputs, outputs)):
                raise TrialHalted('Invalid usage; reservation remains unknown')
            if kind == 'chat':
                input_rate, output_rate = ((Decimal('.2'), Decimal('.8')) if inputs <= 32000 else
                                          (Decimal('.6'), Decimal('2.4')) if inputs <= 256000 else
                                          (Decimal('1.2'), Decimal('4.8')))
                if inputs > 1000000 or outputs > 2048:
                    raise TrialHalted('Usage outside reserved bound')
                finishes = [x.get('finish_reason') for x in payload.get('choices', [])]
                if not finishes or any(x not in ('stop', 'tool_calls') for x in finishes):
                    raise TrialHalted('Incomplete model output; do not continue')
            else:
                input_rate, output_rate = Decimal('.5'), Decimal(0)
                if inputs > 8192:
                    raise TrialHalted('Embedding usage outside bound')
            cost = (inputs*input_rate + outputs*output_rate)/1000000
            with self.lock:
                self._journal('attempt_completed', ordinal=ordinal, kind=kind,
                              usage=usage, cost_cny=str(cost), cost_kind='uncached_rate_estimate_not_invoice')
                self.reserved -= reserve
                self.computed += cost
            return httpx.Response(response.status_code, headers=response.headers,
                                  content=raw_response, request=request)
        except BaseException as error:
            with self.lock:
                self.halted = True
                self._journal('attempt_interrupted', ordinal=ordinal,
                              error_type=type(error).__name__, unknown_reserve_cny=str(reserve))
            raise
        finally:
            if response is not None:
                response.close()

    def close(self):
        self.delegate.close()

    def summary(self):
        return {'counts': self.counts, 'computed_uncached_cny': str(self.computed),
                'unknown_reserved_cny': str(self.reserved), 'halted': self.halted,
                'budget_cny': '10', 'invoice_verified': False}
