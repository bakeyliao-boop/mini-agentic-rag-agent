# mini-agentic-rag-agent

`mini-agentic-rag-agent` 是一个面向知识问答与教育资料处理场景的最小 Agentic RAG 原型，提供传统单次检索、虚拟知识空间工具、原文证据校验、流式工具轨迹，以及可重复验证的策略对照评测。

本项目用于学习和验证“什么时候需要继续定位与补证，以及额外成本是否值得”，不是生产级多租户知识平台。项目状态和后续方向见 [ROADMAP.md](./ROADMAP.md)；本文只保留项目定位、快速启动和常用开发入口。

## 核心能力

- **知识加载与向量检索**：读取 Markdown，按段落稳定切分 Chunk，保留虚拟路径和原始行号，并使用 DashScope Embedding 与本地 Chroma 完成语义检索。
- **统一知识视图**：运行时以 Source Markdown 作为内容契约，保留目录层级、正文结构和行号；不同原始资源的解析与转换属于进入该契约前的数据准备过程。
- **虚拟知识空间**：通过 `ls`、`glob`、`search`、`grep` 和 `read` 统一完成目录枚举、路径定位、语义召回、字面定位和原文分页读取。
- **传统与 Agentic 双链路**：传统 RAG 使用单次 Top-K 候选回答；Mini-Agent 可以根据中间结果继续定位、读取和补证，并受检索策略与工具预算约束。
- **证据约束回答**：`search` 和 `grep` 只提供候选位置；知识型回答需引用 `read` 登记的原文证据，再由服务端核验路径、行号和原文是否仍一致。
- **执行控制与状态**：包含内存路径快照、检索停止信号、工具调用预算、可选的先定位后读取策略，以及基于 LangGraph checkpointer 的运行状态。
- **评测与固定数据**：提供冻结任务、来源清单、对照运行器和评分器，用于比较答案覆盖、有效引用、失败率、工具调用、Token 与时延。
- **HTTP 与 Web 演示**：FastAPI 提供传统问答和 Agentic 问答入口；独立 Web 页面支持双模式切换，并通过 NDJSON 实时展示模型产生的工具调用。

证据校验只能证明“引用来自已读取且未变化的原文”，不能自动证明每个回答断言都被引用语义支持；这仍是评测和后续实现需要检查的边界。

## 项目结构

```text
app/
├── main.py                    # 通用 FastAPI 应用与生命周期
├── demo.py                    # 双模式 Web 演示应用
├── http/                      # 请求模型、路由和统一异常
└── static/pilot_demo/         # 演示页面静态资源
rag_core/
├── knowledge/                 # Markdown、路径、读取、切块和 Chroma
├── traditional/               # 传统 RAG 配置与回答流程
├── agentic/                   # Agent、工具、中间件、证据和运行器
├── models.py                  # Chunk、Evidence、Citation 等领域模型
└── settings.py                # 本地环境配置读取
evaluation/
├── cli/                       # 索引与评测运行入口
├── runners/                   # 传统与 Agentic 执行器
├── scorers/                   # 规则评分、语义评分和工具流程评分
├── demos/                     # Web 演示任务配置
├── fixtures/                  # 可提交的冻结语料与来源清单
├── pilots/                    # 独立评测规格
└── results/                   # 经确认保留的评测结果
knowledge/                     # 基础知识库 Markdown
scripts/                       # 可选转换、诊断和维护脚本
tests/                         # Pytest 与前端行为测试
data/                          # 本地 Chroma、下载源件和临时结果；不提交
```

## 环境与配置

基础要求：

- Python `>=3.12,<3.13`
- [uv](https://docs.astral.sh/uv/) 用于 Python、虚拟环境和依赖管理
- 运行前端行为测试时需要 Node.js
- DashScope 可用的 API Key 和网络连接

复制配置模板：

```bash
cp .env.example .env
```

常用配置分组：

- 模型服务：`DASHSCOPE_API_KEY`、`DASHSCOPE_BASE_URL`
- 向量检索：`EMBEDDING_MODEL`、`EMBEDDING_DIMENSIONS`、`EMBEDDING_BATCH_SIZE`
- 本地存储：`CHROMA_PERSIST_DIR` 和各评测入口约定的本地索引目录
- 语义评分：`SEMANTIC_JUDGE_MODEL`、`SEMANTIC_JUDGE_BASE_URL`、`SEMANTIC_JUDGE_API_KEY`

配置读取入口为 [rag_core/settings.py](./rag_core/settings.py)。当前实现从项目根目录的 `.env` 读取非空字符串。模板中的 `CHAT_MODEL`、`CHAT_TEMPERATURE` 和 `KNOWLEDGE_ROOT` 是预留约定，当前运行时尚未统一读取；模型和语料可能由固定配置类或任务规格选择，因此应以对应入口和规格文件为准。

任何 API Key、Token、密码或真实 `.env` 都不应提交到 Git。本地索引和临时结果写入 `data/`，同样不进入版本控制。

## 后端快速启动

安装 Python 和依赖：

```bash
uv python install 3.12
uv sync --group dev
```

在项目根目录配置 `.env` 后启动通用 API：

```bash
uv run python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

健康检查：

```bash
curl http://127.0.0.1:8000/health
```

交互式接口文档默认位于：

```text
http://127.0.0.1:8000/docs
```

`app.main` 会在应用启动时构建共享运行时和知识索引，因此知识目录、模型配置与网络必须可用。命令必须从项目根目录执行，不能进入 `app/` 后再启动。

也可以直接运行单题 Agent：

```bash
uv run python -m app.agent_runner "你的问题" --thread-id local-smoke
```

## Web 演示

演示应用使用独立的任务规格、冻结语料和本地索引。首次运行或演示语料变化后先构建索引：

```bash
uv run python -m evaluation.cli.demo_index
```

然后启动演示服务：

```bash
uv run python -m uvicorn app.demo:app --host 127.0.0.1 --port 8000 --reload
```

浏览器访问：

```text
http://127.0.0.1:8000/
```

页面提供两个模式：

- **传统 RAG**：执行一次 Top-K 检索，展示回答和候选片段；候选不是经过 `read` 登记的引用。
- **Mini-Agent**：使用相同任务、语料、索引和回答模型执行渐进式工具调用，并展示服务端验证后的引用。

Mini-Agent 通过 NDJSON 事件流发送真实的 `tool_started`、`tool_completed` 和 `completed` 事件。页面按工具名称展示执行状态、参数和返回摘要；流中断时保留已经收到的记录，不使用最终答案伪造执行过程。

演示应用当前绑定一个任务规格；题面、语料清单和索引目录由该规格定义，语料来源与哈希记录在对应 `evaluation/fixtures/` 中。索引只保存在本机 `data/`，不会随 Git 分发。单次演示结果不代表稳定性或整体方案优劣。

## API 入口

| 领域 | 代表性路径 | 说明 |
| --- | --- | --- |
| 健康检查 | `GET /health` | 返回服务基础可用状态 |
| 传统问答 | `POST /chat/traditional` | 接收问题和可选虚拟路径，返回答案、候选、时延和 Token |
| Agentic 问答 | `POST /chat/agentic` | 接收问题和线程 ID，返回结构化答案、工具轨迹、引用和 Token |
| 演示配置 | `GET /demo/pilot/config` | 返回服务端冻结任务和可用模式 |
| 演示运行 | `POST /demo/pilot/run` | 执行传统模式或非流式 Agent 模式 |
| 演示事件流 | `POST /demo/pilot/stream` | 以 NDJSON 返回 Agent 工具事件和最终结果 |

通用 API 由 `app.main:app` 提供；演示 API 与静态页面由 `app.demo:app` 提供，两者是独立应用入口。

## 开发、测试与构建

运行完整后端测试：

```bash
uv run python -m pytest tests -q
```

运行单个模块：

```bash
uv run python -m pytest tests/test_tools.py -q
```

运行前端工具轨迹测试：

```bash
node --test tests/frontend/test_trace_ui.cjs
```

测试文件命名为 `test_*.py`，并尽量与源码职责对应。修改路径、切块、索引、工具、证据或运行策略时，应同时覆盖正常路径、边界输入和失败恢复。

当前 `pyproject.toml` 设置 `package = false`，项目以应用和评测脚本形式运行，没有独立的 wheel 或 sdist 发布流程。

## 文档

主要入口：

- [README.md](./README.md)：项目定位、快速启动和常用入口
- [ROADMAP.md](./ROADMAP.md)：阶段目标、能力差距和后续路线
- [PROMPT_VERSIONS.md](./PROMPT_VERSIONS.md)：系统提示词版本、动机和实验记录
- [evaluation/demos/](./evaluation/demos)：演示任务规格
- [evaluation/fixtures/](./evaluation/fixtures)：冻结语料、来源、哈希与转换边界
- [evaluation/results/](./evaluation/results)：经确认保留的评测输出

工具契约、提示词、评测口径或启动方式变化时，应同步更新相应测试和文档。具体题目、语料规模、模型运行结果和已知失败留在 `evaluation/`，不写进项目总览。

## 数据与脚本

- `knowledge/` 保存基础知识库 Markdown，用于通用问答和回归测试。
- `evaluation/fixtures/` 保存可重复验证的冻结语料；每组语料应带来源、哈希、行数和转换边界。
- `evaluation/results/` 只保存经过确认需要共享的评测结果，不作为运行时状态。
- `data/` 保存本地 Chroma、下载源件、诊断和临时结果，默认不提交。
- `scripts/` 保存可选的数据转换、诊断和维护工具；冻结 fixture 的正常运行不应依赖本机 Office 或一次性转换环境。

修改冻结语料会改变行号、Chunk ID 和引用结果，应创建新版本或同步更新来源清单与评测标准。

## 协作约定

- 在功能分支开发，不直接修改默认分支。
- 修改范围保持聚焦，不顺手提交无关实验或本地文件。
- 暂存前显式检查文件列表，避免对脏工作树使用 `git add .`。
- 提交前运行与风险相匹配的测试，并区分工作树测试与干净提交测试。
- 对照评测应固定语料、Embedding、回答模型、题面和预算，同时报告质量、失败率、Token、时延和工具次数。
- 注释与 docstring 使用中文；提交信息使用简短、明确的中文描述。
