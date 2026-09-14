# DEV-01结构化收口失败原因核查

日期：2026-09-14。只读核查业务代码与已安装依赖、检索官方协议，并用保存响应做离线回放；未修改业务代码或调用真实模型。

## 已确认的因果链

1. 最后一次实际HTTP请求004.request.json携带六个tools（包括GroundedAnswer）、tool_choice=required、enable_thinking=false、stream=false。read L489正文及已登记的evidence_id就在该请求的最后一条tool消息中。
2. 原始HTTP响应004.response.raw为200，finish_reason=stop，message只含普通content及role，没有tool_calls。正文正确说明“提交办理请求一个月内”，但没有通过GroundedAnswer提交答案和证据编号。问题已经出现在保存的服务端响应中，不是后续SDK把已有工具调用丢失。
3. LangChain 1.3.11 factory.py L1196—1201只在ToolStrategy且output.tool_calls非空时进入结构化工具处理。完全没有调用时走L1270，只返回messages，不生成structured_response，也不进入“结构化参数校验失败”的纠正流程。
4. 同文件L1865—1868的model_to_tools边在最后AIMessage没有工具调用时直接结束循环，即使本次要求结构化输出。
5. 本项目rag_core/agentic/runner.py的finalize_agent_result发现structured_response不是GroundedAnswer，直接返回insufficient及空引用。该分支先于正文证据校验，因此这次不是evidence_id校验失败，也不是程序认定原文不足。

## 参数如何到达HTTP

build_knowledge_agent传response_format=GroundedAnswer。factory.py L1387—1391在结构化工具存在时使用tool_choice=any；langchain_openai/chat_models/base.py将any转换为required。实际HTTP记录证明转换完成，不是参数仅出现在计划或回调中。

[百炼Function Calling官方文档](https://help.aliyun.com/zh/model-studio/qwen-function-calling)将Qwen3.7-Flash列为支持模型，并说明非思考模式下required用于强制返回工具调用；思考模式下不支持这一设置。本次enable_thinking=false，因此不能归因于已开启思考模式造成冲突。本次响应与上述required语义不一致。

required要求至少一个工具，并不单独指定必须调用GroundedAnswer；现有提示词另有完成取证后提交GroundedAnswer的要求。无论它应选哪项工具，本次返回完全没有tool_calls这一事实已确认。

## 离线复现

structured-output-offline-replay.json记录离线结果：真实SDK和当前Agent图使用MockTransport依次读取保存的4个HTTP响应，未加载.env或发外部请求；重现glob→grep→read、structured_response=null、最终insufficient和空引用。

这个回放证明对已观察到的响应，本地框架和项目收口行为可确定复现。不是再次运行Qwen，不证明模型错误出现频率，也不解释服务端内部为何未遵循required。

## 原因边界

已确定：API此次返回未遵从工具输出要求；框架把无工具调用当正常结束；项目将输出格式缺失错误映射为“证据不足”。

尚未确定：是模型本次未遵循要求、供应商在该模型/对话阶段的参数执行问题，还是其他服务端处理因素。仅凭客户端请求与响应无法进一步区分，不能宣称Qwen3.7普遍不支持required或LangChain发送了错误参数。若继续验证需另行批准最小兼容性对照，或由供应商依request id检查服务端；本轮未发送外部工单。

## 建议修复方向（尚未实施）

先让缺结构化结果被明确识别为输出协议失败，不再显示为业务证据不足。若考虑一次受限格式补交，应只使用本轮实际已有上下文和证据ID，不重新取证、不注入金标；失败时保留真实错误，新增调用成本及恢复上限需单独确认。这是收口层修复，超出首轮DeepSeek只能修改位置策略的权限。

原定位流程本次已有正面证据，不应把收口失败归因于位置提示词不够好。候选设计与付费复测尚未进行。
## 2026-09-14 · 公开资料补充核查

找到高度对应的LangChain上游报告：[issue #36349](https://github.com/langchain-ai/langchain/issues/36349)。报告区分“工具调用参数错误”与“根本没有工具调用”：后者可能静默返回且没有structured_response；所指出的output.tool_calls条件与本地1.3.11及离线回放一致。它是公开复现证据，不据此推定任意新版本已修复或建议直接升级。

百炼[强制工具调用说明](https://help.aliyun.com/zh/model-studio/qwen-function-calling)还说明：如果希望模型普通文字总结工具结果，应去掉tool_choice，否则仍会返回工具调用。这支持“required不应因前一条是工具返回就自动失效”，不支持把本次普通正文响应当作正常required行为。mini-agent用GroundedAnswer作为最终提交通道，不能照普通聊天例子直接删required解决收口。

另找到vLLM上游[issue #54808](https://github.com/vllm-project/vllm/issues/54808)，报告Qwen解析器路径可能丢失required/指定工具的约束。但它针对自部署vLLM及其他Qwen模型，不是本次百炼qwen3.7-flash服务的证据；不能认定百炼采用相同实现或具有同一个缺陷。

本轮未定位到明确针对百炼qwen3.7-flash、非思考、非流式、工具结果后忽略required的官方缺陷公告。已证实的仍是本次HTTP响应与约定不一致、LangChain结束及本地误报链路。进一步定位上游需要获准兼容性复测或服务端排查；未进行新模型调用、外部工单或代码修改。