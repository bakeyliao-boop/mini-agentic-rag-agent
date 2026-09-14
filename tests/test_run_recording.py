import json
from pathlib import Path
from uuid import uuid4

import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from rag_core.agentic.recording import RunRecorder
from rag_core.agentic.runner import AgenticRuntime, run_agentic_question, stream_agentic_question


class Model(FakeMessagesListChatModel):
    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        return self.bind(tools=[t if isinstance(t, dict) else {'name': t.name}
                                for t in tools])


def answer():
    return AIMessage(content='', tool_calls=[{
        'name': 'GroundedAnswer', 'args': {'answer_type': 'conversation',
        'answer': 'done', 'evidence_ids': []}, 'id': 'answer-1', 'type': 'tool_call'}])


def runtime(tmp_path, responses):
    root = tmp_path / 'corpus'
    root.mkdir(exist_ok=True)
    (root / 'guide.md').write_text('all materials are returned\n', encoding='utf-8')
    return AgenticRuntime(root, object(), Model(responses=responses), [])


def read_call(call_id='read-1'):
    return AIMessage(content='', tool_calls=[{'name': 'read', 'args': {
        'path': '/guide.md', 'start_line': 1, 'limit': 1}, 'id': call_id, 'type': 'tool_call'}])


def events(directory):
    return [json.loads(line) for line in (directory / 'events.jsonl').read_text().splitlines()]


def test_real_graph_records_return_before_next_input_and_preserves_registration(tmp_path):
    d = tmp_path / 'trace'
    result = run_agentic_question(runtime(tmp_path, [read_call(), answer()]), 'question', 't', recording_dir=d)
    rows = events(d)
    returned = next(x for x in rows if x['event'] == 'tool_returned')
    output = returned['data']['output']
    body = json.loads(output['content'])
    assert body['lines'][0]['text'] == 'all materials are returned'
    assert body['lines'][0]['evidence_id']
    next_input = next(x for x in rows if x['event'] == 'model_input' and x['sequence'] > returned['sequence'])
    tool_message = next(m for m in next_input['data']['messages'][0] if m['type'] == 'tool')
    assert tool_message['content'] == output['content']
    started = next(x for x in rows if x['event'] == 'tool_started')
    assert started['data']['tool_call_id'] == 'read-1'
    assert started['data']['model_call_id']
    assert next_input['data']['parameters']['tools']
    assert result['answer'] == 'done'
    assert json.loads((d / 'status.json').read_text())['history_prefix_complete']


def test_recording_off_preserves_result_shape_and_calls(tmp_path):
    off = run_agentic_question(runtime(tmp_path, [read_call(), answer()]), 'q', 't')
    on = run_agentic_question(runtime(tmp_path, [read_call(), answer()]), 'q', 't', recording_dir=tmp_path/'trace')
    assert off == on


def test_stream_records_without_replacing_ui_payload(tmp_path):
    d = tmp_path/'trace'
    output = list(stream_agentic_question(runtime(tmp_path, [read_call(), answer()]), 'q', 't', recording_dir=d))
    assert output[-1]['event'] == 'completed'
    assert any(x['event'] == 'tool_returned' for x in events(d))
    assert events(d)[-1]['event'] == 'run_completed'


def test_parallel_calls_keep_distinct_association(tmp_path):
    call = read_call(); call.tool_calls.append(read_call('read-2').tool_calls[0])
    d = tmp_path/'trace'
    run_agentic_question(runtime(tmp_path, [call, answer()]), 'q', 't', recording_dir=d)
    rows = events(d)
    starts = [x['data'] for x in rows if x['event'] == 'tool_started']
    ends = [x['data'] for x in rows if x['event'] == 'tool_returned']
    assert {x['tool_call_id'] for x in starts} == {'read-1', 'read-2'}
    assert len({x['model_call_id'] for x in starts}) == 1
    assert {x['tool_run_id'] for x in starts} == {x['tool_run_id'] for x in ends}


def test_result_survives_no_next_model_and_missing_result_stays_unknown(tmp_path):
    d = tmp_path/'trace'; recorder = RunRecorder(d); tool_id = uuid4()
    recorder.on_tool_start({'name': 'read'}, '{}', run_id=tool_id, tool_call_id='r')
    recorder.on_tool_end(ToolMessage(content='useful', tool_call_id='r'), run_id=tool_id)
    assert events(d)[-1]['event'] == 'tool_returned'
    other_id = uuid4()
    recorder.on_tool_start({'name': 'read'}, '{}', run_id=other_id, tool_call_id='missing')
    recorder._status('interrupted')
    state = json.loads((d/'status.json').read_text())
    assert not state['history_prefix_complete']
    assert state['pending_observations'] == [str(other_id)]
    assert not any(x['event'] == 'tool_error' for x in events(d))


def test_explicit_error_and_stream_close_are_observed(tmp_path):
    recorder = RunRecorder(tmp_path/'error'); run = uuid4()
    recorder.on_tool_start({'name': 'read'}, '{}', run_id=run)
    recorder.on_tool_error(ValueError('private details'), run_id=run)
    assert events(tmp_path/'error')[-1]['data']['error_type'] == 'ValueError'
    d = tmp_path/'stream'
    stream = stream_agentic_question(runtime(tmp_path, [answer()]), 'q', 't', recording_dir=d)
    next(stream); stream.close()
    assert json.loads((d/'status.json').read_text())['state'] == 'interrupted'


def test_redaction_and_missing_usage(tmp_path):
    d = tmp_path/'trace'; rec = RunRecorder(d); run = uuid4()
    rec.on_chat_model_start({'api_key': 'hidden-secret'}, [[HumanMessage(content='q')]],
                            run_id=run, invocation_params={'api_key': 'hidden-secret',
                            'headers': {'Authorization': 'Bearer hidden-secret'},
                            'temperature': 0})
    msg = AIMessage(content='ok', response_metadata={'headers': {'Authorization': 'secret'}})
    rec.on_llm_end(LLMResult(generations=[[ChatGeneration(message=msg)]]), run_id=run)
    text = (d/'events.jsonl').read_text()
    assert 'hidden-secret' not in text and 'Authorization' not in text
    assert events(d)[-1]['data']['messages'][0]['usage_metadata'] is None


def test_write_failure_does_not_change_execution_but_invalidates_trace(tmp_path, monkeypatch):
    d = tmp_path/'trace'; rec = RunRecorder(d)
    original = Path.open
    def broken(path, *args, **kwargs):
        if path == rec.events:
            raise OSError('disk full')
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'open', broken)
    with pytest.warns(RuntimeWarning):
        rec.on_tool_end('returned', run_id=uuid4())
    rec._status('completed')
    state = json.loads((d/'status.json').read_text())
    assert state['recording_failed'] and not state['history_prefix_complete']


def test_refuses_overwriting_prior_run(tmp_path):
    RunRecorder(tmp_path/'trace')
    with pytest.raises(FileExistsError):
        RunRecorder(tmp_path/'trace')


def test_actual_model_failure_after_read_preserves_return_and_marks_interruption(tmp_path):
    class FailingModel(Model):
        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            if any(isinstance(m, ToolMessage) for m in messages):
                raise RuntimeError('offline simulated model failure')
            return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)

    base = runtime(tmp_path, [read_call()])
    rt = AgenticRuntime(base.knowledge_root, base.vector_store,
                        FailingModel(responses=[read_call()]), base.path_snapshot)
    d = tmp_path/'trace'
    with pytest.raises(RuntimeError, match='offline simulated'):
        run_agentic_question(rt, 'q', 't', recording_dir=d)
    rows = events(d)
    returned = next(x for x in rows if x['event'] == 'tool_returned')
    error = next(x for x in rows if x['event'] == 'model_error')
    assert returned['sequence'] < error['sequence']
    assert 'all materials are returned' in returned['data']['output']['content']
    state = json.loads((d/'status.json').read_text())
    assert state['state'] == 'interrupted' and not state['history_prefix_complete']
