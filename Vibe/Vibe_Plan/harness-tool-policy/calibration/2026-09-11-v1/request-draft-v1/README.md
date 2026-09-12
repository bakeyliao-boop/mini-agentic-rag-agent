# 待审请求正文样例 v1

日期：2026-09-12。仅规划材料，未发送，不是可执行runner配置。五份JSON对应五个案例的完整HTTP正文样例；每份拟重复使用3次，形成预定15次独立请求。不是15次真实记录，不含模拟模型回复。

先看[运行参数与请求组装草案](../运行参数与请求组装草案-v1.md)。每份messages[0].content是统一规则正文，messages[1].content解码后严格等于cases.json相应单例input。模型、预算、次数属于已讨论规划范围；具体请求参数仍待确认，不能直接调用。

| 样例 | 文件UTF-8字节数 | SHA-256 |
| --- | --- | --- |
| [CAL-01](CAL-01.request.example.json) | 9299 | `c4910ebfef04bc711db5984bf63e141a74a1cb577aad69724113eb3e32358679` |
| [CAL-02](CAL-02.request.example.json) | 9015 | `f9f88a888d53e6d91a6b8ca62e4579d58e8128056da117500af63afeaf1e16a1` |
| [CAL-03](CAL-03.request.example.json) | 10118 | `012c795a0894a0ded8409c1c0451fd12c46b58a88fd9b87df4f607a763a6e918` |
| [CAL-04](CAL-04.request.example.json) | 9989 | `ea2420fcc6afd0c7e32417f7ed957740bcedb35314b4847a78bc24a50604702f` |
| [CAL-05](CAL-05.request.example.json) | 8943 | `4a6d4f1d99e5ad8c116f26eec23da3122018f501f221ead3f643f9e38669f2ab` |

[manifest.json](manifest.json)只供组织侧核对版本、样例和预定顺序；不得整体发送给评估器。它包含case_id、请求数、预算等组织元数据，评估器请求正文没有这些字段。组织侧预期及示例答案均未复制入本目录。

静态检查已确认：5份JSON可解析；5份系统内容相同且精确来自规则正文；各user内容与对应input逐项相同；task_reference递归无金标source_line；保留实际工具返回行号；6条原文行与1条摘要子串回源一致。没有把文件存在或静态检查说成运行时隔离已验证。

未读取.env或密钥，不含认证头，不创建会话，不调用项目API，不保存猜测输出。实际usage未知／尚不存在，不能用预留数填造模型实测。


## 独立只读抽查

2026-09-12，子代理request_boundary_review核对参数草案、manifest及CAL-03请求样例，未发现实质问题：HTTP顶层thinking正确，客户端重试/超时未混入正文；system与规则正文一致，user解码后与CAL-03 input一致；样例哈希与manifest一致，预算未验证状态准确。该检查只覆盖所列材料，不能宣称已验证实际网络请求或模型行为。


## 本地计数后续证据

本轮五份请求正文及manifest字节未变。2026-09-12已完成对应官方固定版本的本地计数，见[核验报告](../本地Token计数与预算核验-v1.md)。所有本地计数低于12000预留；API真实计数／账单仍未验证。manifest的input_token_bound_verified=false不能理解为已经证明线上上界；本地与API两个层次分别看待。
