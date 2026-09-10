# mini-agentic-rag-agent

一个最小、可测试、基于原文证据回答的 Agentic RAG 实现。

本项目聚焦普通 RAG 与 Agentic RAG 的关键差异：普通 RAG 通常根据一次检索返回的 Top-K Chunk 回答；mini-agent 可以继续定位和读取原文，补齐跨 Chunk、跨段落或跨文件的证据。

```text
普通 RAG
Query -> Search -> Top-K Chunks -> Answer

mini-agent
Query -> 定位候选 -> 读取原文 -> 继续补证 -> 校验证据 -> Answer + Citations
```

## 核心设计目标

### 1. 深度检索与证据补全

`search` 只负责定位相关候选，不能直接作为最终知识证据。Agent 必须通过 `read` 返回 Source Markdown 原文；如果答案仍不完整，可以继续读取其他范围或其他文件。

“完整”表示覆盖当前问题所需要的证据，不表示无条件读取整篇文档。

例如，“七年级下册三角形面积计算的教学内容”可能分散在教学目标、公式推导和练习设计等不同 Chunk 中。mini-agent 的目标是继续读取并组合这些证据，而不是只使用相似度最高的一个片段。

### 2. 统一资源视图

目标架构会把不同资源经过各自的解析过程，统一投影为 Agent 可读取的 Source Markdown：

```text
文档、表格、图片、音频、视频、basic_info
                    ↓ 各自专业解析
              Source Markdown
                    ↓
          虚拟路径 + 行号 + 引用
```

统一视图不等于抹掉资源差异：文档需要保留标题和表格结构，图片依赖 OCR/VLM，音视频依赖 ASR、时间段、关键帧和视觉描述。

### 3. 稳定的 Agent 访问接口

Agent 使用虚拟路径访问知识，而不直接操作 COS URL、`resource_id`、`content_id` 或数据库结构。服务端负责把虚拟路径映射到真实存储。

```text
Agent: read("/课程资源/七年级/数学/三角形面积.md")
                           ↓
服务端: 路径解析 -> 内部资源身份 -> 权限/范围校验 -> 内容投影
```

这种设计带来的好处：

- COS 签名 URL 过期，不影响 Agent 的调用方式。
- 数据从 COS 迁移到 PostgreSQL，Agent 代码不用修改。
- 文档、图片、音视频都使用相同的 `read(path)`。
- 不把数据库字段、存储地址和签名 URL 暴露给模型。
- 工具轨迹和引用更容易被人理解。

这里的“稳定”主要指同一投影版本和连续工具调用中的稳定访问契约，并不承诺资源改名或路径规则升级后路径永远不变。

## 当前实现

### 跨平台 Pilot 运行

`multi-chunk-guide-001` 使用仓库内冻结的 Markdown 评测快照，而不是本机的原始 PDF 或 Office 文档：

```text
evaluation/fixtures/multi_chunk_guide_001/
├── SOURCE.json
└── corpus/
    └── 政务与公共服务/深圳市/知识产权公共服务/...
```

`SOURCE.json` 记录该快照的来源、哈希、行数和预期 Chunk 数。评测运行时不能修改其中的 Markdown 正文，否则证据行号和 Chunk 边界会变化。

每台机器都应自行创建本地 Chroma 索引；`data/` 不进入 Git：

```bash
# 先在 .env 填入 DASHSCOPE_API_KEY 等必需模型配置。
python -m evaluation.cli.pilot_index

# 运行 Pilot A 组。
python -m evaluation.cli.pilot
```

`PILOT_CHROMA_PERSIST_DIR` 是可选覆盖项。旧 `.env` 没有该字段时，程序默认使用：

```text
data/pilots/multi-chunk-guide-001
```

`scripts/convert_yunzhi_documents.ps1` 是 Windows + Microsoft Office 的可选数据准备工具，用于将原始文档转换为 Markdown。它不是 Pilot 运行前置条件；macOS、Linux 和 Windows 都直接使用冻结 fixture 构建索引，不需要执行该脚本。

### 面试演示页面

先完成 Pilot 索引准备，再从项目根目录启动专用演示应用：

```bash
python -m evaluation.cli.pilot_index
uvicorn app.demo:app --reload
```

浏览器访问：

```text
http://127.0.0.1:8000/
```

页面只提供两个模式：

- `传统 RAG`：对冻结问题执行单次 Top-5 检索，展示回答与候选片段；候选不作为已验证引用。
- `Mini-Agent`：使用相同语料与索引执行渐进式工具调用，展示工具轨迹和经过路径、行号、原文复核的引用。

Mini-Agent 页面通过 NDJSON 事件流实时接收 LangGraph 的真实执行事件：模型提交工具调用时显示“执行中”，对应 `ToolMessage` 返回后更新为“调用完成”或“调用失败”，最后再展示结构化回答和引用。页面不会用最终结果伪造过程动画。

工具过程以可折叠的活动列表展示：默认每次调用显示动作、目标和结果摘要，展开可查看完整路径与参数；回答返回后收起活动列表。流中断时保留已收到的记录并标为中断，切换模式可查看各自最近一次运行。前端流状态回归可使用 `node --test tests/frontend/test_trace_ui.cjs` 运行。

前端演示使用独立配置 [patent_review_comparison_001.json](evaluation/demos/patent_review_comparison_001.json)，比较“专利优先审查”和“专利快速预审”的受理条件、申请材料、办理顺序与结果。两种模式使用同一道演示题，Mini-Agent 保持 6 次知识库工具调用上限。

该演示题复用原 Pilot 的冻结指南和已有索引，无需重建 Chroma。原 `multi-chunk-guide-001` 三事项压力题、12 个答案点和结果文件仍用于历史评测，不能将它们的评分直接用于本演示题。修改 Demo 配置后需重启服务。

每次运行都使用服务端 Demo 配置中的问题，前端不能修改问题。页面记录模型上报的 Token 和服务端完整问答耗时；不包含启动、索引构建和浏览器渲染时间。
