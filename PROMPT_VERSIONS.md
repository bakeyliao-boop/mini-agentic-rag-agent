# Knowledge Agent Prompt 版本记录

每次修改 Agent 决策规则时新增版本，不直接覆盖历史 Prompt。运行中的版本由
`app/prompts.py` 内的 `KNOWLEDGE_AGENT_PROMPT_VERSION` 指定。

## Prompt-V1.0

- 建立 `ls/search/read` 三工具职责。
- 要求知识回答在最终回答前使用 `read`。

## Prompt-V1.1

- 增加目录范围规则。
- `ls` 找到具体目录后，要求将该目录作为 `search.path`。

## Prompt-V1.2

- `read` 已获得充分证据后立即停止搜索。
- 最终知识回答必须提交 `GroundedAnswer` 和 `evidence_id`。
- 禁止相似关键词重复搜索；证据不足时及时输出 `insufficient`。

真实冒烟结果：知识库内问题恢复为 `search -> read -> GroundedAnswer`，知识库外
问题在工具调用达到上限时由服务端降级为 `insufficient`。
