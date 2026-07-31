# Agentic RAG 从零复现计划

## 1. 项目目标

从一个小型、可测试的本地 Markdown 知识库出发，逐步复现 Agentic RAG 最有价值的能力：

```text
发现知识空间（ls）
  -> 语义定位候选（search）
  -> 读取并验证原文（read）
  -> 注册本轮证据
  -> 输出结构化回答
  -> 服务端校验证据
```

本项目不是 `yunzhi-rag-server` 的完整复制品。第一版只复现足以体现 Agentic RAG 与传统 RAG 差异的核心链路。

## 2. 第一版固定边界

### 保留

- Python 3.12
- FastAPI + Pydantic
- LangChain `create_agent` / LangGraph
- 一个 OpenAI-compatible LLM
- 本地 Markdown 知识库
- Chroma 本地向量检索
- `ls`、`search`、`read` 三个工具
- 传统 RAG 对照版本
- 每轮独立的 `EvidenceRegistry`
- `GroundedAnswer` 结构化输出
- `path + line range + exact quote` 引用验证
- pytest

### 暂不实现

- MySQL、PostgreSQL、Redis、COS
- MinerU、Doc2PDF、Office、图片和音视频解析
- BM25、Hybrid、RRF、二阶段 rerank
- 多租户、鉴权、受众过滤
- canonical/custom mount
- PostgreSQL checkpoint
- SSE、AG-UI、CopilotKit、Next.js
- Langfuse 和复杂评测 Trace
- PDF bbox、PPT slide、音视频时间定位
- 异步训练 Worker、消息队列和容器编排

## 3. 必须保留的核心原则

1. Source Markdown 是面向 Agent 的唯一事实源。
2. `search` 只返回候选，不能直接成为知识回答的证据。
3. 知识事实在最终回答前必须通过 `read` 读取原文。
4. evidence ID 只能由服务端根据本轮 `read` 结果生成。
5. 最终回答必须经过服务端证据校验。
6. 证据不足时输出 `insufficient`，不能用模型常识补齐。
7. Agent 的每次工具调用都要可观察，便于与传统 RAG 对比。

## 4. 三个工具的第一版契约

### `ls`

职责：列出一个虚拟目录的直接子项，不递归读取正文。

```python
ls(path: str = "/") -> {
    "path": str,
    "entries": [
        {"path": str, "type": "directory | file"}
    ]
}
```

约束：

- 所有虚拟路径使用 POSIX 风格，以 `/` 开头。
- 不暴露本机绝对路径。
- 必须阻止 `..` 等路径穿越。
- 目录类问题可以使用 `ls` 回答，但 `ls` 不证明正文事实。

### `search`

职责：在指定虚拟目录范围内做语义检索，返回待读取的候选。

```python
search(query: str, path: str = "/", limit: int = 5) -> {
    "hits": [
        {
            "path": str,
            "start_line": int,
            "end_line": int,
            "score": float,
            "preview": str
        }
    ],
    "usage": "candidate_only"
}
```

约束：

- 支持按 `path` 前缀缩小检索范围。
- 每个 chunk 必须保存 `path/start_line/end_line`。
- preview 只帮助 Agent 选择下一步，不能注册 evidence。
- 第一版最多返回 5 个候选。

### `read`

职责：按行读取 Source Markdown，并为可引用原文生成服务端 evidence ID。

```python
read(path: str, start_line: int = 1, limit: int = 80) -> {
    "path": str,
    "lines": [
        {
            "line": int,
            "text": str,
            "evidence_id": str
        }
    ],
    "next_line": int | None
}
```

约束：

- 行号从 1 开始并保持稳定。
- 第一版每个非空 Markdown 行注册一条 evidence。
- 单次最多读取 80 行，并设置最大字符数。
- evidence 绑定本轮运行、文件路径、行号和原文。

## 5. 数据模型草案

```python
class Chunk:
    chunk_id: str
    path: str
    start_line: int
    end_line: int
    text: str


class Evidence:
    evidence_id: str
    path: str
    start_line: int
    end_line: int
    quote: str


class Citation:
    path: str
    start_line: int
    end_line: int
    quote: str


class GroundedAnswer:
    answer_type: Literal[
        "knowledge",
        "directory",
        "conversation",
        "insufficient",
    ]
    answer: str
    evidence_ids: list[str]
```

## 6. 里程碑与验收标准

### 阶段 0：初始化与语料准备

- [x] 创建 Python 3.12 项目和虚拟环境
- [x] 固定依赖版本，不使用 `latest`
- [x] 建立 `.env.example`，不提交真实密钥
- [ ] 从现有文档中选择 15～30 个 Markdown 文件
- [ ] 设计包含相似标题、重复概念和目录层次的知识库
- [ ] 准备至少 10 个固定评测问题

验收：

- [x] 新环境可以一次安装成功
- [x] 知识库无需外部解析服务即可读取
- [ ] 评测问题覆盖精确事实、目录、歧义、多文档和知识库外问题

当前状态（2026-07-31）：本地 Python 3.12、`uv`、`.venv` 和锁定依赖已验证；
`knowledge/` 中没有 Markdown 语料，`evaluation/` 及固定评测问题尚未同步到当前工作区。

### 阶段 1：虚拟路径与 Source Markdown

- [x] 定义知识库真实根目录和虚拟根目录 `/`
- [x] 实现虚拟路径规范化
- [x] 实现安全的虚拟路径到本地文件映射
- [x] 阻止绝对路径和 `..` 路径穿越
- [x] 实现稳定的 1-based Markdown 行号
- [x] 定义 `Chunk`、`Evidence`、`Citation`、`GroundedAnswer`

验收：

- [x] 正常路径可以稳定解析
- [x] 恶意路径不能逃出知识库根目录
- [x] 同一文件重复读取时行号完全一致

### 阶段 2：实现 `ls` 与 `read`

- [x] `ls` 只返回直接子项
- [x] `ls` 区分 file/directory
- [x] `ls` 返回稳定排序
- [x] `read` 支持 `start_line + limit`
- [x] `read` 支持下一页游标 `next_line`
- [ ] `read` 限制最大行数和最大字符数
- [x] 为 `ls` 和 `read` 编写单元测试

验收：

- [x] 可以从 `/` 逐层找到目标文件
- [ ] 可以分页读完一个长 Markdown
- [ ] 不存在的文件、目录和越界行号有稳定错误

当前状态（2026-07-31）：`ls`、基础 Markdown 行读取和分页读取已在本地完成；
最大行数、最大字符数、参数合法性和越界错误仍未完成。

### 阶段 3：切块、索引与 `search`

- [ ] 按段落/行切块，同时保留行号范围
- [ ] 第一版 chunk 约 500～800 字符并带少量重叠
- [ ] 使用 Chroma 建立本地索引
- [ ] 索引保存虚拟路径和行号 metadata
- [ ] 实现全库 search
- [ ] 实现 path-scoped search
- [ ] search 结果明确标记为 `candidate_only`
- [ ] 为索引与 search 编写单元测试

验收：

- [ ] 每个 search hit 都能通过 path/line 被 read 复现
- [ ] 指定 `/数学/小学/` 时不会命中 `/数学/高中/`
- [ ] 重建索引后，相同语料的结果结构保持稳定

### 阶段 4：传统 RAG 基线

- [ ] 实现固定流程：问题 -> search top-k -> Prompt -> 回答
- [ ] 固定模型、温度、top-k 和语料版本
- [ ] 记录检索结果、答案、延迟和 token
- [ ] 用固定问题集跑一份 baseline 结果

验收：

- [ ] 每个答案都能追溯到当时塞入 Prompt 的 chunk
- [ ] 保存传统 RAG 的错误回答、错误引用和拒答表现

### 阶段 5：三工具 Agent

- [ ] 使用 LangChain `create_agent`
- [ ] 注册 `ls/search/read`
- [ ] 写入“search 只能定位，知识回答必须 read”的系统规则
- [ ] 限制最大工具调用次数为 6
- [ ] 限制 search top-k 和 read 总字符数
- [ ] 输出每次工具调用轨迹
- [ ] 暂时使用内存会话状态

验收：

- [ ] Agent 可以完成 `search -> read -> answer`
- [ ] Agent 可以完成 `ls -> scoped search -> read -> answer`
- [ ] 第一次检索不充分时可以继续搜索或读取
- [ ] 不出现无限工具循环

### 阶段 6：证据闸门

- [ ] 每轮 Agent 创建独立 `EvidenceRegistry`
- [ ] 只有 `read` 可以注册正文 evidence
- [ ] 最终输出使用 `GroundedAnswer`
- [ ] 校验 evidence ID 是否属于本轮 Registry
- [ ] 根据 evidence 构造 citation
- [ ] 重新校验 path、line range 和 quote
- [ ] 无有效 evidence 的 knowledge 回答强制降级
- [ ] 为伪造 ID、过期引用和原文变更编写测试

验收：

- [ ] 模型编造 evidence ID 时回答被拒绝
- [ ] 修改原文后旧 citation 验证失败
- [ ] 知识库外问题稳定返回 `insufficient`
- [ ] 有效 citation 能精确定位到 Markdown 行

### 阶段 7：传统 RAG 与 Agentic RAG 对照评测

- [ ] 两个版本使用相同语料、模型和 embedding
- [ ] 比较答案正确率
- [ ] 比较引用有效率
- [ ] 比较目录范围命中率
- [ ] 比较知识库外问题的拒答率
- [ ] 比较延迟、token 和调用次数
- [ ] 记录 Agent 从错误候选中恢复的案例
- [ ] 形成一份 A/B 结果报告

验收：

- [ ] 可以明确说明哪些问题 Agentic RAG 更好
- [ ] 可以明确说明哪些简单问题传统 RAG 更快、更便宜
- [ ] 结论来自固定问题集，而不是单次演示

### 阶段 8：FastAPI 接口

- [ ] `GET /health`
- [ ] `POST /chat/traditional`
- [ ] `POST /chat/agentic`
- [ ] 返回回答、citation 和工具轨迹
- [ ] 定义统一错误响应
- [ ] 为主要接口编写 TestClient 测试

验收：

- [ ] 无前端也能通过 OpenAPI/Swagger 完整体验两个版本
- [ ] 同一问题可以方便地比较传统与 Agentic 结果

## 7. 第一批 A/B 问题类型

1. 精确事实：需要找到并读取一段原文。
2. 目录问题：需要通过 `ls` 理解知识空间。
3. 范围歧义：小学和高中存在相同关键词或同名文件。
4. 多文档问题：需要读取两个文件后综合回答。
5. 长上下文问题：search preview 不足，必须 read 前后文。
6. 知识库外问题：正确行为是 `insufficient`。
7. 错误候选恢复：第一条 search hit 不是最终证据。

## 8. 建议项目结构

```text
agentic-rag-from-scratch/
├── AGENTS.md
├── PLAN.md
├── pyproject.toml
├── .env.example
├── app/
│   ├── models.py
│   ├── knowledge_store.py
│   ├── indexer.py
│   ├── tools.py
│   ├── evidence.py
│   ├── traditional_rag.py
│   ├── agent.py
│   └── main.py
├── knowledge/
├── evaluation/
│   ├── questions.json
│   └── results/
└── tests/
    ├── test_paths.py
    ├── test_ls.py
    ├── test_read.py
    ├── test_search.py
    ├── test_grounding.py
    └── test_api.py
```

上述文件除 `PLAN.md` 外，等进入对应阶段后再逐个创建。

## 9. 工作方式

- 每次只推进一个里程碑或一个可验证子项。
- 实现前先写清输入、输出和失败行为。
- 重要逻辑先写失败测试，再实现。
- 不提前加入当前阶段不需要的抽象。
- 每个阶段结束后更新本文件的勾选项和进度记录。
- 每次开发、修复、重构、依赖调整后，都要按当前本地代码和测试结果同步本文件。
- 每次 `pull`、`merge`、`rebase` 或 `cherry-pick` 后，都要检查新提交并同步勾选项、进度记录和“下一步”。
- 只有当前工作区已实现且验收证据成立的项目才能勾选；远程尚未合并的实现不能计入本地完成状态。
- 若测试未运行或验收资产缺失，必须在进度记录中写明，不能仅依据提交信息标记完成。
- 新发现的需求先记录，不直接插入当前阶段。

## 10. 进度记录

| 日期 | 阶段 | 状态 | 结果与决定 |
|---|---|---|---|
| 待开始 | 阶段 0 | 未开始 | 第一项：初始化 Python 3.12 项目并固定依赖 |
| 2026-07-23 | 阶段 0 | 进行中 | 已完成 Python 3.12 项目初始化、固定依赖、生成 `uv.lock` 并通过 WSL 安装验证；下一项：建立 `.env.example` |
| 2026-07-24 | 阶段 0 | 进行中 | 已建立 DashScope 配置模板；语料与评测问题曾在原环境准备，但当前本地工作区未同步这些非 Git 资产 |
| 2026-07-27 | 阶段 1 | 已完成 | 已完成虚拟路径规范化、安全路径映射、稳定 Markdown 行号及四个 RAG 核心数据模型 |
| 2026-07-27 | 阶段 2 | 进行中 | 已完成 `ls` 直接子项浏览、类型区分、稳定排序和相关测试；分页 `read` 尚未进入当前分支 |
| 2026-07-30 | 环境 | 已完成 | 已安装 `uv` 和 Python 3.12.13，按锁文件安装 107 个依赖，初始化本地配置与 Chroma 目录 |
| 2026-07-31 | 状态同步 | 已完成 | 当前测试 73 passed；阶段 0 回退为部分完成，阶段 1 标记完成，阶段 2 按本地实现更新 |
| 2026-07-31 | 阶段 2 | 进行中 | 从远程 `test` 快进合并 `adca968`，完成 `start_line + limit`、`next_line` 及分页读取测试；验收 75 passed |

## 11. 整体完成定义

当以下条件全部满足时，第一版完成：

- [ ] 三个工具都有清晰契约和自动化测试
- [ ] 传统 RAG 与 Agentic RAG 使用同一套语料和检索器
- [ ] Agent 能根据问题自主选择工具并进行多步取证
- [ ] knowledge 回答必须有经过服务端验证的 read evidence
- [ ] 引用可以通过 path/line/quote 回到当前 Source Markdown
- [ ] 知识库外问题可以稳定拒答
- [ ] 有一份可复现的 A/B 评测结果
- [ ] FastAPI 可以同时暴露传统和 Agentic 两个入口

## 12. 下一步

只执行阶段 0 的下一项：

> 将 15～30 个 Markdown 语料和至少 10 个固定评测问题同步到当前本地工作区，
> 并验证目录层次与 7 类评测场景。

在阶段 0 资产恢复前，不开始索引或 Agent。
