"""Optional local observation; never supplies evaluation references to the agent."""

from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
from threading import RLock
import warnings

from langchain_core.callbacks import BaseCallbackHandler


def _safe(value):
    if hasattr(value, 'model_dump'):
        return _safe(value.model_dump(mode='json'))
    if isinstance(value, dict):
        return {str(k): ('[REDACTED]' if re.search(
            r'authorization|api.?key|password|secret|access.?token|headers', str(k), re.I
        ) else _safe(v)) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(v) for v in value]
    if isinstance(value, str):
        return re.sub(r'(?i)Bearer\s+\S+|\bsk-[A-Za-z0-9_-]{16,}', '[REDACTED]', value)
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return {'unserializable_type': type(value).__name__}


class RunRecorder(BaseCallbackHandler):
    """Flush each observation. A persistence error invalidates the whole record."""

    run_inline = True
    raise_error = False

    def __init__(self, directory: Path):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.events = self.directory / 'events.jsonl'
        self._lock = RLock()
        self.failed = False
        self.sequence = 0
        self.pending = set()
        self.proposers = {}
        self.emit('recording_started', layer='langchain_callback_not_http_wire',
                  redaction='credential keys and recognizable credential strings')
        self._status('in_progress')

    def _status(self, state):
        try:
            with (self.directory / 'status.json').open('w', encoding='utf-8') as f:
                json.dump({'state': state, 'recording_failed': self.failed,
                           'pending_observations': sorted(self.pending),
                           'history_prefix_complete': state == 'completed' and not self.failed
                           and not self.pending}, f)
                f.flush()
                os.fsync(f.fileno())
        except Exception:
            self._fail()

    def _fail(self):
        self.failed = True
        warnings.warn('Run recording failed; this trace is incomplete.', RuntimeWarning)

    def emit(self, event, **data):
        with self._lock:
            try:
                self.sequence += 1
                item = {'sequence': self.sequence, 'at': datetime.now(timezone.utc).isoformat(),
                        'event': event, 'data': _safe(data)}
                with self.events.open('a', encoding='utf-8', newline='\n') as f:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
                    f.flush()
                    os.fsync(f.fileno())
            except Exception:
                self._fail()

    def on_chat_model_start(self, serialized, messages, *, run_id, **kwargs):
        with self._lock:
            self.pending.add(str(run_id))
            params = kwargs.get('invocation_params', {})
            allowed = {'model', 'model_name', 'temperature', 'max_tokens', 'max_completion_tokens',
                       'tools', 'tool_choice', 'response_format', 'stop', 'thinking', 'extra_body'}
            self.emit('model_input', model_call_id=str(run_id), messages=messages,
                      parameters={k: v for k, v in params.items() if k in allowed},
                      layer='langchain_pre_normalization_callback',
                      service_received='unknown')

    def on_llm_end(self, response, *, run_id, **kwargs):
        with self._lock:
            messages = [g.message for group in response.generations for g in group
                        if hasattr(g, 'message')]
            for message in messages:
                for call in getattr(message, 'tool_calls', []):
                    self.proposers[call['id']] = str(run_id)
            self.emit('model_output', model_call_id=str(run_id), messages=messages)
            self.pending.discard(str(run_id))

    def on_llm_error(self, error, *, run_id, **kwargs):
        self.emit('model_error', model_call_id=str(run_id), error_type=type(error).__name__)
        with self._lock:
            self.pending.discard(str(run_id))

    def on_tool_start(self, serialized, input_str, *, run_id, inputs=None, **kwargs):
        with self._lock:
            self.pending.add(str(run_id))
            call_id = kwargs.get('tool_call_id')
            self.emit('tool_started', tool_run_id=str(run_id), tool_call_id=call_id,
                      model_call_id=self.proposers.get(call_id), name=serialized.get('name'),
                      arguments=inputs if inputs is not None else input_str)

    def on_tool_end(self, output, *, run_id, **kwargs):
        with self._lock:
            self.emit('tool_returned', tool_run_id=str(run_id), output=output)
            self.pending.discard(str(run_id))

    def on_tool_error(self, error, *, run_id, **kwargs):
        self.emit('tool_error', tool_run_id=str(run_id), error_type=type(error).__name__)
        with self._lock:
            self.pending.discard(str(run_id))


@contextmanager
def recording_session(directory, runtime, question, thread_id):
    if directory is None:
        yield None
        return
    recorder = RunRecorder(directory)
    try:
        from rag_core.agentic.prompts import KNOWLEDGE_AGENT_SYSTEM_PROMPT
        sources = [{
            'path': str(p.relative_to(runtime.knowledge_root)),
            'sha256': hashlib.sha256(p.read_bytes()).hexdigest(),
        } for p in sorted(runtime.knowledge_root.rglob('*.md'))]
        recorder.emit('run_started', thread_id=thread_id, question=question, sources=sources,
                      system_prompt_sha256=hashlib.sha256(
                          KNOWLEDGE_AGENT_SYSTEM_PROMPT.encode()).hexdigest())
        yield recorder
    except BaseException as error:
        recorder.emit('run_interrupted', error_type=type(error).__name__, stop_reason='unknown')
        recorder._status('interrupted')
        raise
    else:
        recorder.emit('run_completed', stop_reason='unknown')
        recorder._status('completed')
