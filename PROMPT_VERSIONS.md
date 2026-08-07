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

### 调整内容

- `read` 已获得充分证据后立即停止搜索。
- 最终知识回答必须提交 `GroundedAnswer` 和 `evidence_id`。
- 禁止相似关键词重复搜索；证据不足时及时输出 `insufficient`。

### 真实结果

- 普通知识库内问题恢复为 `search -> read -> GroundedAnswer`。
- 知识库外问题可以降级为 `insufficient`，但仍可能耗尽工具调用次数。
- 完整结果见 [Prompt-V1.2 原始结果](evaluation/results/agentic-baseline-qwen3.6-flash-thinking-off-prompt-v1.2.json)。

### 已发现问题

- `directory-001`：Agent 在 `ls("/课程资源")` 已返回“初中、小学”后，仍提交
  `insufficient`。V1.2 没有明确规定纯目录题可以直接依据 `ls` 结果提交
  `answer_type=directory`。
- `ambiguity-001`：问题中的目录名只是知识题范围，但 Agent 将它误判成目录遍历任务；
  先猜测 `/自动控制系统`，失败后从根目录选择“初中”分支逐层 `ls`，最终触发工具调用上限。
- “处理目录或范围问题时逐层 `ls`”的规则覆盖面过大，没有区分纯目录清单与目录范围内的
  文件内容、属性问题。

## Prompt-V1.3

### 实验假设

- 明确区分纯目录题与“目录只是范围”的知识题，可以修复 `directory-001` 和
  `ambiguity-001`，无需先提高工具调用上限。
- 在暂不增加 `glob` 工具的第一轮实验中，可用根路径 `search` 定位未知完整路径，再用
  `read` 取证。

### 调整内容

- 纯目录题仅指直接子项、文件/目录清单和路径存在性问题；`ls` 返回所需子项后立即提交
  `answer_type=directory`，不再调用 `search/read`。
- 目录名只用于限定范围，而用户询问文件内容、属性或资源类型时，仍按知识题处理。
- 完整虚拟路径未知时禁止猜路径和从根目录选择单一分支逐层试探；先从 `/` 使用
  `search`，并只使用工具返回的完整路径。
- 工具报告路径不存在后必须重新发现路径，不能继续拼接错误路径。
- 保留 V1.2 的充分证据停止规则和重复搜索限制。

### 状态

- 已运行新基线验证。
- 原始结果见 [Prompt-V1.3 原始结果](evaluation/results/agentic-baseline-qwen3.6-flash-thinking-off-prompt-v1.3.json)。
- 评分结果见 [Prompt-V1.3 评分](evaluation/results/agentic-baseline-qwen3.6-flash-thinking-off-prompt-v1.3-score.json)。

### 真实结果

| 指标 | Prompt-V1.2 | Prompt-V1.3 | 变化 |
|---|---:|---:|---:|
| 回答类型准确率 | 80% | 100% | +20 个百分点 |
| 答案点覆盖率 | 48% | 64% | +16 个百分点 |
| 目录题准确率 | 0% | 100% | 已修复 |
| 引用覆盖率 | 87.5% | 100% | +12.5 个百分点 |
| `read` 合规率 | 87.5% | 100% | +12.5 个百分点 |
| 工具调用总数 | 32 | 29 | -3 |
| 平均延迟 | 3.94 秒 | 4.66 秒 | +18% |
| 总 Token | 85,755 | 105,888 | +23.5% |

目标题验证：

- `directory-001` 从 `insufficient` 修复为 `directory`，正确回答直接子目录是“初中、小学”。
  但首次调用仍猜测了不存在的 `/课程资源目录`，工具调用次数维持 3 次，说明“禁止猜路径”
  没有完全生效。
- `ambiguity-001` 从 `insufficient` 修复为 `knowledge`，执行轨迹从错误分支的 7 次 `ls`
  改为一次失败 `ls`、一次根路径 `search` 和两次 `read`，工具调用从 7 次降至 4 次；两个资源类型
  和两条引用都正确。但首次调用仍猜测了 `/自动控制系统`。

### 遗留问题

- V1.3 Prompt 更长，使多数普通知识题即使工具次数不变，也增加约 600–1000 Token。
- `out-of-scope-001` 连续进行了 6 次相似 `search`，第 7 次触发工具限制。虽然最终正确拒答，
  Token 从 20,489 增至 34,867，延迟从 4.4 秒增至 11.3 秒，是本轮成本回退的主要来源。
- V1.2 已存在“禁止相似关键词重复搜索”的文字规则，但 V1.3 真实运行仍未稳定遵守；后续应评估
  使用代码限制 `search` 次数，而不是继续只增加 Prompt 文字。

### 实验结论

- “区分纯目录题与目录范围内的知识题”假设成立，修复了两道目标题并提升了整体正确性。
- “禁止猜测路径”和“禁止重复搜索”仅靠 Prompt 不能稳定生效。
- V1.3 暂时作为正确性更高的基线保留；下一轮优先治理重复搜索和输入 Token，而不是提高全局工具上限。
