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
