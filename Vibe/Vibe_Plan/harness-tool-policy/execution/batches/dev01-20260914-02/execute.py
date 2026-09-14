"""Authorized fresh D01 index build followed by one unchanged DEV-01 run."""
from pathlib import Path
import hashlib,json,os,sys,time
os.environ['ANONYMIZED_TELEMETRY']='False'
root=Path.cwd();sys.path.insert(0,str(root));packet=Path(__file__).resolve().parent
from evaluation.harness_trial import BudgetTransport,TrialHalted,write_new,json_bytes,tree_hashes
import httpx

def main():
    approved=json.loads((packet/'approval.json').read_text());frozen=json.loads((packet/'frozen.json').read_text())
    assert approved['execution_authorized'] and approved['build_budget_cny']==.36
    for name,sha in frozen['files'].items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==sha
    original=Path('/mnt/c/Users/hyf/Desktop/mini-agentic-rag-agent/data/pilots/multi-chunk-guide-001')
    assert tree_hashes(original)==frozen['original_index_files']
    index=root/approved['index_dir'];assert not index.exists()
    output=root/approved['output_dir'];output.mkdir(parents=True,exist_ok=False)
    from rag_core.settings import load_settings_from_env,_required_setting
    settings=load_settings_from_env(Path('/mnt/c/Users/hyf/Desktop/mini-agentic-rag-agent'))
    base=_required_setting(settings,'DASHSCOPE_BASE_URL').rstrip('/')
    key=_required_setting(settings,'DASHSCOPE_API_KEY');del settings
    if base!='https://dashscope.aliyuncs.com/compatible-mode/v1':
        write_new(output/'summary.json',json_bytes({'status':'preflight_refused','api_calls':0}));return 2
    build=BudgetTransport(output/'build-http',budget_cny='.36',limits={'chat':0,'embedding':86})
    trial=None;summary={'status':'started'};started=time.monotonic()
    try:
        from langchain_openai import OpenAIEmbeddings,ChatOpenAI
        from langchain_chroma import Chroma
        from chromadb.config import Settings
        from rag_core.knowledge.indexer import collect_knowledge_chunks,chunk_to_document
        from rag_core.knowledge.store import build_knowledge_path_snapshot
        from rag_core.agentic.runner import AgenticRuntime,run_agentic_question
        corpus=root/'evaluation/fixtures/multi_chunk_guide_001/corpus'
        chunks=collect_knowledge_chunks(corpus)
        assert len(chunks)==86 and all(len(c.text.encode())<=8192 for c in chunks)
        assert [hashlib.sha256(c.text.encode()).hexdigest() for c in chunks]==frozen['chunk_text_hashes']
        with httpx.Client(transport=build,trust_env=False,follow_redirects=False,timeout=120) as client:
            embedding=OpenAIEmbeddings(model='text-embedding-v4',dimensions=1024,chunk_size=1,
                check_embedding_ctx_length=False,api_key=key,base_url=base,http_client=client,max_retries=0,timeout=120)
            store=Chroma(collection_name='knowledge_chunks',persist_directory=str(index),embedding_function=embedding,
                collection_metadata={'hnsw:space':'cosine'},client_settings=Settings(anonymized_telemetry=False))
            store.add_documents([chunk_to_document(c) for c in chunks],ids=[c.chunk_id for c in chunks])
            stored=store.get(include=['documents','metadatas'])
            expected={c.chunk_id:c for c in chunks}
            assert len(stored['ids'])==86 and set(stored['ids'])==set(expected)
            for id,text,meta in zip(stored['ids'],stored['documents'],stored['metadatas']):
                c=expected[id];assert text==c.text and meta['path']==c.path and meta['start_line']==c.start_line and meta['end_line']==c.end_line
            write_new(output/'index-build.json',json_bytes({'status':'validated','model':'text-embedding-v4','dimensions':1024,
                'count':86,'all_stored_text_and_positions_verified':True,'source_manifest':frozen['corpus_files'],
                'http_request_count':build.counts['embedding'],'usage':build.summary(),'snapshot_files':tree_hashes(index)}))
            print('Fresh D01 index built and 86 entries verified; starting DEV-01 once.',flush=True)
        trial=BudgetTransport(output/'trial-http',budget_cny='10',limits={'chat':8,'embedding':6})
        with httpx.Client(transport=trial,trust_env=False,follow_redirects=False,timeout=120) as client:
            common=dict(api_key=key,base_url=base,http_client=client,max_retries=0,timeout=120)
            embedding=OpenAIEmbeddings(model='text-embedding-v4',dimensions=1024,chunk_size=1,check_embedding_ctx_length=False,**common)
            store=Chroma(collection_name='knowledge_chunks',persist_directory=str(index),embedding_function=embedding,
                client_settings=Settings(anonymized_telemetry=False))
            model=ChatOpenAI(model='qwen3.7-flash',temperature=0,max_tokens=2048,extra_body={'enable_thinking':False},streaming=False,**common)
            runtime=AgenticRuntime(corpus,store,model,build_knowledge_path_snapshot(corpus))
            result=run_agentic_question(runtime,(packet/'question.txt').read_text(encoding='utf-8').strip(),
                'harness-dev01-20260914-02',recording_dir=output/'trace')
            write_new(output/'answer.json',json_bytes(result));summary['status']='completed_pending_review'
    except BaseException as error:
        summary.update(status='interrupted',error_type=type(error).__name__)
    finally:
        summary.update(build=build.summary(),trial=trial.summary() if trial else None,
            elapsed_seconds=time.monotonic()-started,original_index_unchanged=tree_hashes(original)==frozen['original_index_files'])
        write_new(output/'summary.json',json_bytes(summary));build.close()
        if trial:trial.close()
    print(json.dumps(summary),flush=True)
    return 0 if summary['status']=='completed_pending_review' else 2
if __name__=='__main__':raise SystemExit(main())