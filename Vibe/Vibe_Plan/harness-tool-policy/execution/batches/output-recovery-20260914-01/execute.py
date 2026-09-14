"""One approved recovery HTTP request, followed by the actual local recovery validator."""
from pathlib import Path
import sys,json,hashlib,datetime
r=Path.cwd();sys.path.insert(0,str(r));p=Path(__file__).resolve().parent
from evaluation.harness_trial import BudgetTransport,write_new,json_bytes
from rag_core.agentic.output_recovery import recover_output_once
from rag_core.agentic.evidence import EvidenceRegistry
from rag_core.agentic.runner import finalize_agent_result
from rag_core.agentic.recording import RunRecorder
from langchain_core.messages import convert_to_messages
import httpx

def main():
    frozen=json.loads((p/'frozen.json').read_text())
    for path,sha in frozen['files'].items():assert hashlib.sha256((r/path).read_bytes()).hexdigest()==sha
    approval=json.loads((p/'approval.json').read_text());assert approval['execution_authorized']
    request=json.loads((p/'request.json').read_text(encoding='utf-8'))
    original=r/'evaluation/results/harness-dev01/dev01-20260914-02'
    old_req=json.loads((original/'trial-http/004.request.json').read_bytes())
    old_resp=json.loads((original/'trial-http/004.response.raw').read_bytes())
    state={'messages':convert_to_messages(old_req['messages']+[old_resp['choices'][0]['message']])}
    state['messages'][-1].response_metadata['finish_reason']='stop'
    read=json.loads(old_req['messages'][-1]['content'])
    ids=[x['evidence_id'] for x in read['lines'] if 'evidence_id' in x]
    registry=EvidenceRegistry(ids[0].rsplit(':evidence-',1)[0])
    registered=registry.register_read_page(read);assert [x.evidence_id for x in registered]==ids
    corpus=r/'evaluation/fixtures/multi_chunk_guide_001/corpus'
    output=r/approval['output_dir'];output.mkdir(parents=True,exist_ok=False)
    from rag_core.settings import load_settings_from_env,_required_setting
    settings=load_settings_from_env(Path('/mnt/c/Users/hyf/Desktop/mini-agentic-rag-agent'))
    base=_required_setting(settings,'DASHSCOPE_BASE_URL').rstrip('/');key=_required_setting(settings,'DASHSCOPE_API_KEY');del settings
    if base!='https://dashscope.aliyuncs.com/compatible-mode/v1':
        write_new(output/'summary.json',json_bytes({'status':'preflight_refused','api_calls':0}));return 2
    guard=BudgetTransport(output/'http',budget_cny='1.30',limits={'chat':1,'embedding':0},deadline_seconds=120)
    recorder=RunRecorder(output/'recovery-trace');summary={'status':'started'}
    try:
        with httpx.Client(transport=guard,trust_env=False,follow_redirects=False,timeout=120) as client:
            class PreparedModel:
                max_retries=0
                def bind_tools(self, tools, *, tool_choice):
                    assert [x.__name__ for x in tools]==['GroundedAnswer'] and tool_choice=='GroundedAnswer'
                    return self
                def invoke(self,messages,config=None):
                    # Verify helper-generated context is exactly the approved message content.
                    canonical=lambda seq: [(m.type,m.content,getattr(m,'tool_calls',None),getattr(m,'tool_call_id',None),m.name) for m in seq]
                    assert canonical(messages)==canonical(convert_to_messages(request['messages']))
                    response=client.post(base+'/chat/completions',json=request,headers={'Authorization':'Bearer '+key})
                    payload=response.json();message=convert_to_messages([payload['choices'][0]['message']])[0]
                    message.response_metadata={'finish_reason':payload['choices'][0]['finish_reason'],'model_name':payload['model']}
                    u=payload['usage'];message.usage_metadata={'input_tokens':u['prompt_tokens'],'output_tokens':u['completion_tokens'],'total_tokens':u['total_tokens']}
                    return message
            recovered=recover_output_once(state,PreparedModel(),registry,corpus,recorder=recorder)
            delivered=finalize_agent_result(recovered,registry,corpus)
            write_new(output/'structured-answer.json',json_bytes(recovered['structured_response'].model_dump()))
            write_new(output/'answer.json',json_bytes(delivered))
            recorder._status('completed');summary['status']='completed_pending_review'
    except BaseException as error:
        summary.update(status='failed',error_type=type(error).__name__,reason=getattr(error,'reason',None));recorder._status('interrupted')
    finally:
        summary.update(guard.summary());write_new(output/'summary.json',json_bytes(summary));guard.close()
    print(json.dumps(summary),flush=True)
    return 0 if summary['status']=='completed_pending_review' else 2
if __name__=='__main__':raise SystemExit(main())