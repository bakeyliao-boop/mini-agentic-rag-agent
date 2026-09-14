"""One explicitly enabled formatting submission, without a knowledge-tool loop."""

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from rag_core.agentic.evidence import validate_answer_evidence, validate_evidence_sources
from rag_core.agentic.prompts import KNOWLEDGE_AGENT_SYSTEM_PROMPT
from rag_core.models import GroundedAnswer


class OutputProtocolError(RuntimeError):
    """Output delivery failed; this does not mean the source evidence is absent."""

    def __init__(self, reason):
        self.reason = reason
        super().__init__(f'模型答案提交格式或引用校验失败（{reason}），未交付答案。')


FORMAT_SUBMISSION = (
    '上一条回复未按要求提交结构化答案。请仅依据本轮已有对话与工具返回，'
    '调用 GroundedAnswer 提交答案。知识性回答只能引用本轮 read 实际返回的 evidence_id。'
    '不进行新的检索，不补造事实或证据编号；已有依据不足时如实提交 insufficient。'
    '不要返回普通文字。'
)


def is_unstructured_completion(state):
    if isinstance(state.get('structured_response'), GroundedAnswer):
        return False
    messages = state.get('messages', [])
    if not messages:
        return False
    last = messages[-1]
    # Budget middleware synthetic messages and tool-triggered stops are not
    # evidence of a normal provider completion and must not bypass stop rules.
    return (isinstance(last, AIMessage) and not last.tool_calls
            and not last.invalid_tool_calls and bool(last.content)
            and last.response_metadata.get('finish_reason') == 'stop')


def recover_output_once(state, model, registry, knowledge_root, *, recorder=None):
    """One logical invoke. The experiment's SDK/transport retains zero retries."""
    if not is_unstructured_completion(state):
        return state
    # ChatOpenAI owns an SDK client whose retries cannot be disabled by bind().
    # Require an explicitly zero-retry runtime rather than silently multiplying
    # the single authorized submission into several HTTP attempts.
    if getattr(model, 'max_retries', 0) != 0:
        raise OutputProtocolError('recovery_requires_zero_transport_retries')
    if recorder is not None:
        recorder.emit('output_recovery_started', reason='missing_structured_response', maximum_attempts=1)
    messages = list(state['messages'])
    if not any(isinstance(m, SystemMessage) for m in messages):
        messages.insert(0, SystemMessage(content=KNOWLEDGE_AGENT_SYSTEM_PROMPT))
    messages.append(HumanMessage(content=FORMAT_SUBMISSION))
    try:
        reply = model.bind_tools([GroundedAnswer], tool_choice='GroundedAnswer').invoke(
            messages, config={'callbacks': [recorder]} if recorder is not None else None)
        if recorder is not None:
            recorder.emit('output_recovery_returned', reply=reply)
        if (not isinstance(reply, AIMessage) or reply.invalid_tool_calls
                or len(reply.tool_calls) != 1 or reply.tool_calls[0]['name'] != 'GroundedAnswer'
                or reply.response_metadata.get('finish_reason') in ('length', 'content_filter')):
            raise OutputProtocolError('invalid_recovery_format')
        try:
            answer = GroundedAnswer.model_validate(reply.tool_calls[0]['args'])
        except ValueError as error:
            raise OutputProtocolError('invalid_recovery_schema') from error
        try:
            evidence = validate_answer_evidence(answer, registry)
            validate_evidence_sources(evidence, knowledge_root)
        except ValueError as error:
            raise OutputProtocolError('invalid_recovery_evidence') from error
    except Exception as error:
        if recorder is not None:
            recorder.emit('output_recovery_failed', error_type=type(error).__name__,
                          reason=getattr(error, 'reason', 'recovery_request_failed'))
        if isinstance(error, OutputProtocolError):
            raise
        raise OutputProtocolError('recovery_request_failed') from error
    if recorder is not None:
        recorder.emit('output_recovery_completed', structured_response=answer)
    return {**state, 'messages': [*state['messages'], reply], 'structured_response': answer,
            'output_recovery': {'attempted': True, 'status': 'validated'}}
