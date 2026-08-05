# 待解决与延后生效说明

本文记录以下类型的问题：

- 代码已经创建，但尚未接入完整运行流程。
- 当前看起来与已有功能重复，只有后续阶段完成后才产生实际作用。
- 为避免误以为功能已经完整生效，需要明确当前边界和完成条件。

## P-001：EvidenceRegistry 当前与 read 返回内容重复

### 当前疑问

`read` 已经返回 `path + lines`，为什么还要把相同内容登记为 `Evidence`？

### 当前状态

`EvidenceRegistry` 已接入 Agent 的 `read` 工具：每次运行会创建独立 Registry，`read` 返回的非空行会增加 `evidence_id`。目前它仍未参与最终回答校验。

因此现阶段它的实际作用是：

- 为 `read` 原文增加本轮 `run_id` 和 `evidence_id`。
- 将原文整理为固定的 `Evidence` 数据结构。
- 在服务端内存中保存一份本轮证据记录。

当前 Agent 仍然可以只读取 `read` 结果后直接生成答案，Registry 尚未形成强制约束。

### 后续何时生效

阶段 6 完成以下链路后才真正生效：

```text
read 原文
  -> EvidenceRegistry 注册
  -> 模型输出 GroundedAnswer + evidence_ids
  -> 服务端检查 evidence_id 是否属于本轮 Registry
  -> 重新核对 path、行号和 quote
  -> 生成 citation 或降级为 insufficient
```

### 完成条件

- [x] `read` 工具自动把非空原文注册到本轮 Registry。
- 最终知识回答必须提交本轮有效的 `evidence_id`。
- 伪造、过期或其他轮次的 evidence ID 会被拒绝。
- 没有有效证据的知识回答被强制降级为 `insufficient`。

状态：已接入 `read`，待接入最终回答校验。

## P-002：InMemorySaver 目前不能跨命令行进程保存会话

### 当前疑问

已经配置 `InMemorySaver`，为什么重复运行 `python -m app.agent_runner` 时，相同 `thread_id` 不一定保留上一条命令的消息？

### 当前状态

每次执行命令都会创建一个新的 Python 进程、Agent 和 `InMemorySaver`。进程退出后，内存中的会话也会消失。

当前会话记忆只在同一个 Agent 实例和同一个 Python 进程内生效。

### 后续何时生效

FastAPI 服务持续运行后，同一服务进程中的相同 `thread_id` 可以复用内存状态。需要跨进程或重启保存时，再替换为持久化 checkpointer。

状态：第一版有意保留的限制。

## P-003：系统提示词目前不是服务端强制规则

### 当前疑问

提示词已经规定“search 只能定位，知识回答必须 read”，为什么模型仍有可能直接回答？

### 当前状态

系统提示词只是在引导模型决策。`create_agent` 没有强制第一步调用工具，也没有在最终回答阶段检查是否存在 `read` 证据。

### 后续何时生效

阶段 6 的证据闸门接入后，服务端会验证最终知识回答使用的 evidence ID。即使模型没有遵守提示词，没有有效证据的回答也不能正常通过。

状态：待证据闸门强制执行。

## P-004：思考模式与强制结构化输出冲突

### 当前问题

`create_agent(response_format=GroundedAnswer)` 会通过强制工具调用生成结构化结果，
但 `qwen3.6-flash` 默认开启思考模式。百炼不允许思考模式使用
`tool_choice="required"` 或指定工具对象，因此请求返回 400。

### 当前处理

模型工厂已明确设置：

```python
extra_body={"enable_thinking": False}
```

传统 RAG 和 Agent 共用该模型工厂，因此两者都会关闭思考模式，保持评测配置一致。

### 后续事项

- 重新生成传统 RAG 基线结果，旧结果不能直接用于正式对比。
- 观察非思考模式下的工具选择、回答质量、延迟和 Token 数据。
- 如果复杂问题效果不足，再评估“思考模型回答 + 非思考模型格式化”的两阶段方案。

状态：已临时解决接口冲突，待重新评测效果。
