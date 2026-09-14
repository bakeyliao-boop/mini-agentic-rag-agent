import json
from decimal import Decimal

import httpx
import pytest

from evaluation.harness_trial import BudgetTransport, TrialHalted, compare_vectors, copy_checked_index


def request(kind='chat', **changes):
    body = ({'model': 'qwen3.7-flash', 'temperature': 0, 'enable_thinking': False,
             'stream': False, 'max_completion_tokens': 2048, 'messages': []} if kind == 'chat'
            else {'model': 'text-embedding-v4', 'dimensions': 1024, 'input': ['text']})
    body.update(changes)
    path = 'chat/completions' if kind == 'chat' else 'embeddings'
    return httpx.Request('POST', 'https://dashscope.aliyuncs.com/compatible-mode/v1/'+path,
                         json=body, headers={'Authorization': 'Bearer never-log-this'})


def ok(req):
    return httpx.Response(200, json={'usage': {'prompt_tokens': 100, 'completion_tokens': 20},
                                   'choices': [{'finish_reason': 'stop'}]})


@pytest.mark.parametrize('kind,limit', [('chat', 8), ('embedding', 7)])
def test_exact_request_cap_and_body_only_archive(tmp_path, kind, limit):
    calls = []
    def send(req):
        calls.append(req)
        return ok(req)
    t = BudgetTransport(tmp_path/'http', httpx.MockTransport(send))
    for _ in range(limit):
        t.handle_request(request(kind))
    with pytest.raises(TrialHalted):
        t.handle_request(request(kind))
    assert len(calls) == limit
    assert all('never-log-this' not in p.read_text() for p in t.directory.iterdir())
    assert t.reserved == 0 and t.computed > 0


@pytest.mark.parametrize('mutation', [{'model': 'other'}, {'stream': True}, {'max_completion_tokens': 4096},
                                     {'enable_thinking': True}, {'temperature': 1}, {'n': 2}])
def test_wrong_parameters_cannot_send(tmp_path, mutation):
    calls = []
    t = BudgetTransport(tmp_path/'http', httpx.MockTransport(lambda r: calls.append(r)))
    with pytest.raises(TrialHalted):
        t.handle_request(request(**mutation))
    assert not calls


@pytest.mark.parametrize('failure', ['http', 'usage', 'timeout', 'truncation'])
def test_failure_keeps_reservation_and_blocks_further_requests(tmp_path, failure):
    calls = []
    def send(req):
        calls.append(req)
        if failure == 'timeout':
            raise httpx.ReadTimeout('offline simulated')
        if failure == 'http':
            return httpx.Response(429, json={'error': 'rate limit'})
        if failure == 'usage':
            return httpx.Response(200, json={})
        return httpx.Response(200, json={'usage': {'prompt_tokens': 1, 'completion_tokens': 2048},
                                        'choices': [{'finish_reason': 'length'}]})
    t = BudgetTransport(tmp_path/'http', httpx.MockTransport(send))
    with pytest.raises((TrialHalted, httpx.ReadTimeout)):
        t.handle_request(request())
    with pytest.raises(TrialHalted):
        t.handle_request(request())
    assert len(calls) == 1 and t.reserved == t.RESERVE['chat']


def test_budget_deadline_and_endpoint_refuse_before_send(tmp_path):
    calls = []
    t = BudgetTransport(tmp_path/'http', httpx.MockTransport(lambda r: calls.append(r)))
    bad = request(); bad.url = httpx.URL('https://example.com/compatible-mode/v1/chat/completions')
    with pytest.raises(TrialHalted): t.handle_request(bad)
    t.computed = Decimal('9')
    with pytest.raises(TrialHalted): t.handle_request(request())
    t2 = BudgetTransport(tmp_path/'expired', httpx.MockTransport(lambda r: calls.append(r)), deadline_seconds=-1)
    with pytest.raises(TrialHalted): t2.handle_request(request())
    assert not calls


def test_disk_error_blocks_send_and_later_calls(tmp_path, monkeypatch):
    calls = []
    t = BudgetTransport(tmp_path/'http', httpx.MockTransport(lambda r: calls.append(r)))
    monkeypatch.setattr('evaluation.harness_trial.write_new', lambda *args: (_ for _ in ()).throw(OSError()))
    with pytest.raises(OSError): t.handle_request(request())
    with pytest.raises(TrialHalted): t.handle_request(request())
    assert not calls


def test_embedding_conservative_input_and_dimension_caps(tmp_path):
    t = BudgetTransport(tmp_path/'http', httpx.MockTransport(ok))
    for change in [{'input': ['x', 'y']}, {'input': ['中'*3000]}, {'dimensions': 2048}]:
        with pytest.raises(TrialHalted): t.handle_request(request('embedding', **change))
    assert t.counts['embedding'] == 0


def test_vector_tolerance_and_copy_immutability(tmp_path):
    vector = [.01]*1024
    assert compare_vectors(vector, vector)['compatible']
    assert not compare_vectors(vector, [.02]*1024)['compatible']
    assert not compare_vectors(vector, vector[:-1])['compatible']
    source = tmp_path/'source'; source.mkdir(); (source/'chroma.sqlite3').write_bytes(b'fixture')
    before = copy_checked_index(source, tmp_path/'copy')
    assert (source/'chroma.sqlite3').read_bytes() == b'fixture'
    assert before
    with pytest.raises(TrialHalted): copy_checked_index(source, tmp_path/'copy')


def test_real_sdks_route_through_guard_without_network_or_retry(tmp_path):
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    calls = []
    def send(req):
        calls.append(json.loads(req.content))
        if req.url.path.endswith('embeddings'):
            return httpx.Response(200, json={'object': 'list', 'data': [
                {'object': 'embedding', 'index': 0, 'embedding': [.01]*1024}],
                'model': 'text-embedding-v4', 'usage': {'prompt_tokens': 1, 'total_tokens': 1}})
        return httpx.Response(200, json={'id': 'offline', 'object': 'chat.completion',
            'created': 0, 'model': 'qwen3.7-flash', 'choices': [{'index': 0,
            'message': {'role': 'assistant', 'content': 'ok'}, 'finish_reason': 'stop'}],
            'usage': {'prompt_tokens': 3, 'completion_tokens': 1, 'total_tokens': 4}})
    guard = BudgetTransport(tmp_path/'http', httpx.MockTransport(send))
    with httpx.Client(transport=guard, trust_env=False, follow_redirects=False) as client:
        common = dict(api_key='offline-test-value', base_url='https://dashscope.aliyuncs.com/compatible-mode/v1',
                      http_client=client, max_retries=0, timeout=120)
        model = ChatOpenAI(model='qwen3.7-flash', temperature=0, max_tokens=2048,
                           extra_body={'enable_thinking': False}, streaming=False, **common)
        embed = OpenAIEmbeddings(model='text-embedding-v4', dimensions=1024,
                                 check_embedding_ctx_length=False, chunk_size=1, **common)
        assert len(embed.embed_query('text')) == 1024
        assert model.invoke('question').content == 'ok'
    assert len(calls) == 2 and guard.reserved == 0


def test_compressed_response_is_not_decoded_twice(tmp_path):
    import gzip
    payload = {'usage': {'prompt_tokens': 99, 'total_tokens': 99},
               'data': [{'index': 0, 'embedding': [.01]*1024}]}
    raw = json.dumps(payload).encode()
    def send(req):
        return httpx.Response(200, content=gzip.compress(raw), headers={
            'content-encoding': 'gzip', 'content-type': 'application/json'})
    t = BudgetTransport(tmp_path/'http', httpx.MockTransport(send))
    result = t.handle_request(request('embedding'))
    assert result.json() == payload
    assert 'content-encoding' not in result.headers
    assert (t.directory/'001.response.raw').read_bytes() == raw
    assert not t.halted and t.reserved == 0
    assert t.computed == Decimal('.0000495')


def test_index_build_phase_cannot_call_chat_or_exceed_its_budget(tmp_path):
    calls = []
    def send(req):
        calls.append(req)
        return httpx.Response(200, json={'usage': {'prompt_tokens': 8192, 'total_tokens': 8192}})
    t = BudgetTransport(tmp_path/'build', httpx.MockTransport(send), budget_cny='.36',
                        limits={'chat': 0, 'embedding': 86})
    for _ in range(86):
        t.handle_request(request('embedding'))
    assert t.computed == Decimal('.352256')
    with pytest.raises(TrialHalted): t.handle_request(request('embedding'))
    assert len(calls) == 86
    other = BudgetTransport(tmp_path/'small', httpx.MockTransport(send), budget_cny='.004',
                            limits={'chat': 0, 'embedding': 86})
    with pytest.raises(TrialHalted): other.handle_request(request('embedding'))
    with pytest.raises(TrialHalted): other.handle_request(request())
    assert len(calls) == 86
