"""One explicitly authorized DEV-01 run. Do not rerun or overwrite its output."""
from pathlib import Path
import hashlib,json,os,sqlite3,struct,sys,time
os.environ['ANONYMIZED_TELEMETRY']='False'
root=Path.cwd();sys.path.insert(0,str(root))
packet=Path(__file__).resolve().parent
from evaluation.harness_trial import BudgetTransport, TrialHalted, compare_vectors, write_new, json_bytes, tree_hashes
import httpx

def main():
    approved=json.loads((packet/'approval.json').read_text())
    assert approved['execution_authorized'] and approved['budget_cny']==10
    frozen=json.loads((packet/'frozen.json').read_text())
    for name,sha in frozen['files'].items():
        assert hashlib.sha256((root/name).read_bytes()).hexdigest()==sha
    index=root/approved['copied_index'];assert tree_hashes(index)==frozen['copied_index_files']
    original=Path('/mnt/c/Users/hyf/Desktop/mini-agentic-rag-agent/data/pilots/multi-chunk-guide-001')
    assert tree_hashes(original)==frozen['original_index_files']
    output=root/approved['output_dir'];output.mkdir(parents=True,exist_ok=False)
    # Only these two approved settings are used. Nothing from the environment is logged.
    from rag_core.settings import load_settings_from_env,_required_setting
    settings=load_settings_from_env(Path('/mnt/c/Users/hyf/Desktop/mini-agentic-rag-agent'))
    base_url=_required_setting(settings,'DASHSCOPE_BASE_URL').rstrip('/')
    api_key=_required_setting(settings,'DASHSCOPE_API_KEY');del settings
    if base_url!='https://dashscope.aliyuncs.com/compatible-mode/v1':
        write_new(output/'summary.json',json_bytes({'status':'preflight_refused','reason':'endpoint_outside_scope','api_calls':0}))
        return 2
    guard=BudgetTransport(output/'http')
    summary={'status':'started','start_monotonic':time.monotonic()}
    try:
        with httpx.Client(transport=guard,trust_env=False,follow_redirects=False,timeout=120) as client:
            from langchain_openai import OpenAIEmbeddings,ChatOpenAI
            common=dict(api_key=api_key,base_url=base_url,http_client=client,max_retries=0,timeout=120)
            embedding=OpenAIEmbeddings(model='text-embedding-v4',dimensions=1024,chunk_size=1,
                check_embedding_ctx_length=False,**common)
            db=sqlite3.connect((index/'chroma.sqlite3').as_uri()+'?mode=ro',uri=True)
            blob,metadata=db.execute('SELECT vector,metadata FROM embeddings_queue ORDER BY seq_id LIMIT 1').fetchone();db.close()
            metadata=json.loads(metadata);text=metadata['chroma:document'];old=list(struct.unpack('<1024f',blob))
            new=embedding.embed_query(text)
            compatibility=compare_vectors(old,new)
            compatibility.update(sample_selection='first stored queue row by seq_id',
                sample_text_sha256=hashlib.sha256(text.encode()).hexdigest(),
                scope='one sample numerical compatibility, not full provenance proof')
            write_new(output/'vector-compatibility.json',json_bytes(compatibility))
            if not compatibility['compatible']:
                raise TrialHalted('Index sample incompatible; no question run')
            from langchain_chroma import Chroma
            from chromadb.config import Settings
            from rag_core.agentic.runner import AgenticRuntime,run_agentic_question
            from rag_core.knowledge.store import build_knowledge_path_snapshot
            store=Chroma(collection_name='knowledge_chunks',persist_directory=str(index),embedding_function=embedding,
                         client_settings=Settings(anonymized_telemetry=False))
            assert store._collection.count()==86
            corpus=root/'evaluation/fixtures/multi_chunk_guide_001/corpus'
            model=ChatOpenAI(model='qwen3.7-flash',temperature=0,max_tokens=2048,
                extra_body={'enable_thinking':False},streaming=False,**common)
            runtime=AgenticRuntime(corpus,store,model,build_knowledge_path_snapshot(corpus))
            result=run_agentic_question(runtime,(packet/'question.txt').read_text(encoding='utf-8').strip(),
                'harness-dev01-20260914-01',recording_dir=output/'trace')
            write_new(output/'answer.json',json_bytes(result))
            summary['status']='completed_pending_review'
    except BaseException as error:
        summary.update(status='interrupted',error_type=type(error).__name__)
    finally:
        summary.update(guard.summary());summary['elapsed_seconds']=time.monotonic()-summary.pop('start_monotonic')
        summary['original_index_unchanged']=tree_hashes(original)==frozen['original_index_files']
        write_new(output/'summary.json',json_bytes(summary))
        guard.close()
    print(json.dumps(summary),flush=True)
    return 0 if summary['status']=='completed_pending_review' else 2

if __name__=='__main__':raise SystemExit(main())