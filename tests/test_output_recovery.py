import json
from pathlib import Path

import httpx
import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from rag_core.agentic.evidence import EvidenceRegistry
from rag_core.agentic.output_recovery import OutputProtocolError, recover_output_once
from rag_core.agentic.runner import finalize_agent_result
from rag_core.models import GroundedAnswer


def setup(tmp_path):
    (tmp_path/'guide.md').write_text('the relevant fact\n', encoding='utf-8')
    registry = EvidenceRegistry('test')
    evidence = registry.register_read_page({'path': '/guide.md', 'lines': [
        {'line': 1, 'text': 'the relevant fact'}], 'next_line': None})[0]
    state = {'messages': [HumanMessage(content='question'),
        AIMessage(content='the relevant fact', response_metadata={'finish_reason': 'stop'})]}
    return state, registry, evidence.evidence_id


class SubmissionModel:
    def __init__(self, reply):
        self.reply = reply
        self.calls = 0

    def bind_tools(self, tools, *, tool_choice):
        assert tools == [GroundedAnswer] and tool_choice == 'GroundedAnswer'
        return self

    def invoke(self, messages, config=None):
        self.calls += 1
        assert '不进行新的检索' in messages[-1].content
        if isinstance(self.reply, Exception): raise self.reply
        return self.reply


def submission(id):
    return AIMessage(content='', tool_calls=[{'name': 'GroundedAnswer', 'id': 'submit',
        'type': 'tool_call', 'args': {'answer_type': 'knowledge', 'answer': 'the relevant fact',
                                   'evidence_ids': [id]}}],
        usage_metadata={'input_tokens': 10, 'output_tokens': 2, 'total_tokens': 12})


def test_valid_submission_preserves_history_and_validates_citation(tmp_path):
    state, registry, id = setup(tmp_path)
    model = SubmissionModel(submission(id))
    result = recover_output_once(state, model, registry, tmp_path)
    assert result['messages'][:-1] == state['messages']
    assert 'structured_response' not in state
    delivered = finalize_agent_result(result, registry, tmp_path)
    assert delivered['citations'][0]['quote'] == 'the relevant fact'
    assert model.calls == 1


@pytest.mark.parametrize('kind', ['plain_again', 'bad_id', 'wrong_tool', 'bad_schema', 'error', 'truncated'])
def test_failed_submission_is_explicit_and_never_loops(tmp_path, kind):
    state, registry, id = setup(tmp_path)
    reply = submission(id)
    if kind == 'plain_again': reply = AIMessage(content='still plain')
    elif kind == 'bad_id': reply = submission('invented-id')
    elif kind == 'wrong_tool': reply.tool_calls[0]['name'] = 'read'
    elif kind == 'bad_schema': reply.tool_calls[0]['args'] = {'answer': 'missing answer type'}
    elif kind == 'error': reply = RuntimeError('offline error')
    else: reply.response_metadata['finish_reason'] = 'length'
    model = SubmissionModel(reply)
    with pytest.raises(OutputProtocolError): recover_output_once(state, model, registry, tmp_path)
    assert model.calls == 1


@pytest.mark.parametrize('kind', ['valid', 'tool_stop', 'budget_stop'])
def test_valid_or_control_stopped_run_never_gets_format_retry(tmp_path, kind):
    state, registry, id = setup(tmp_path)
    if kind == 'valid': state['structured_response'] = GroundedAnswer(answer_type='knowledge', answer='fact', evidence_ids=[id])
    elif kind == 'tool_stop': state['messages'].append(ToolMessage(content='none', tool_call_id='s'))
    else: state['messages'].append(AIMessage(content='tool limit reached'))
    model = SubmissionModel(RuntimeError('must not invoke'))
    assert recover_output_once(state, model, registry, tmp_path) is state
    assert model.calls == 0


def test_plain_normal_completion_is_not_reported_as_missing_evidence(tmp_path):
    state, registry, _ = setup(tmp_path)
    with pytest.raises(OutputProtocolError, match='missing_structured_response'):
        finalize_agent_result(state, registry, tmp_path)


def test_changed_source_cannot_be_used_by_recovery(tmp_path):
    state, registry, id = setup(tmp_path)
    (tmp_path/'guide.md').write_text('changed source\n')
    with pytest.raises(OutputProtocolError, match='invalid_recovery_evidence'):
        recover_output_once(state, SubmissionModel(submission(id)), registry, tmp_path)


def test_unknown_sdk_retry_policy_refuses_recovery_before_invocation(tmp_path):
    state, registry, id = setup(tmp_path)
    model = SubmissionModel(submission(id)); model.max_retries = None
    with pytest.raises(OutputProtocolError, match='zero_transport_retries'):
        recover_output_once(state, model, registry, tmp_path)
    assert model.calls == 0


@pytest.mark.parametrize('streaming', [False, True])
def test_saved_dev01_failure_plus_synthetic_submission_through_real_sdk(tmp_path, streaming):
    """Four real saved responses; fifth is synthetic, NOT evidence of Qwen recovery."""
    from langchain_openai import ChatOpenAI
    from rag_core.agentic.runner import AgenticRuntime, run_agentic_question, stream_agentic_question
    from rag_core.knowledge.store import build_knowledge_path_snapshot
    root = Path(__file__).resolve().parents[1]
    saved = root/'evaluation/results/harness-dev01/dev01-20260914-02/trial-http'
    assert saved.is_dir()
    requests = []
    def send(request):
        body = json.loads(request.content); requests.append(body)
        n = len(requests)
        if n <= 4:
            return httpx.Response(200, content=(saved/f'{n:03d}.response.raw').read_bytes(),
                                  headers={'content-type': 'application/json'})
        assert n == 5
        assert [t['function']['name'] for t in body['tools']] == ['GroundedAnswer']
        assert body['tool_choice'] == {'type': 'function', 'function': {'name': 'GroundedAnswer'}}
        tool = next(m for m in reversed(body['messages']) if m['role'] == 'tool')
        evidence = next(x for x in json.loads(tool['content'])['lines'] if x['line'] == 489)
        reply = {'id': 'synthetic-format-submission', 'object': 'chat.completion', 'created': 0,
            'model': 'qwen3.7-flash', 'usage': {'prompt_tokens': 10, 'completion_tokens': 5, 'total_tokens': 15},
            'choices': [{'index': 0, 'finish_reason': 'tool_calls', 'message': {'role': 'assistant',
            'content': None, 'tool_calls': [{'id': 'synthetic-submit', 'type': 'function', 'function': {
                'name': 'GroundedAnswer', 'arguments': json.dumps({'answer_type': 'knowledge',
                'answer': '提交办理请求一个月内缴费。', 'evidence_ids': [evidence['evidence_id']]})}}]}}]}
        return httpx.Response(200, json=reply)
    with httpx.Client(transport=httpx.MockTransport(send), trust_env=False) as client:
        model = ChatOpenAI(model='qwen3.7-flash', temperature=0, max_tokens=2048,
            extra_body={'enable_thinking': False}, api_key='offline-only', max_retries=0, http_client=client)
        corpus = root/'evaluation/fixtures/multi_chunk_guide_001/corpus'
        runtime = AgenticRuntime(corpus, object(), model, build_knowledge_path_snapshot(corpus))
        question = json.loads((saved/'001.request.json').read_bytes())['messages'][1]['content']
        if streaming:
            result = list(stream_agentic_question(runtime, question, 'recovery-test', recover_output=True))[-1]['data']
        else:
            result = run_agentic_question(runtime, question, 'recovery-test', recover_output=True)
    assert len(requests) == 5
    assert result['answer_type'] == 'knowledge' and result['citations'][0]['start_line'] == 489
    assert [t['name'] for t in result['tool_traces']] == ['glob', 'grep', 'read']
    assert result['token_usage']['total_tokens'] == 13180 + 15
    assert result['output_recovery']['status'] == 'validated'
