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
