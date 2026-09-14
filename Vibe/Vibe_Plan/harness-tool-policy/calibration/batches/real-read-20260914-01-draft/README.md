# 真实read单次评价请求包（待授权）

复用人工已确认的DEV-01 read材料和固定v3规则。执行模型deepseek-flash、非思考、temperature=0、输出上限2048，串行一次，零重试，超时120秒。只评价一次调用贡献，不重新运行Agent、不生成策略。

请求与已审核request.example.json字节一致；预期只在cases.json组织侧，模型仅接收规则正文和input。字段名approved_*沿用现有入口格式，但execution_authorized=false、parameter_acceptance=pending，不能据此认为运行已获批准。

本地Token核算和预算见preparation-check.json；建议上限0.05元。需要本次执行及原项目.env中的SEMANTIC_JUDGE_BASE_URL、SEMANTIC_JUDGE_API_KEY复用授权；本轮未读取凭据或调用模型。仅当前目录是准备材料，既有源输入及人审记录保持不变。
