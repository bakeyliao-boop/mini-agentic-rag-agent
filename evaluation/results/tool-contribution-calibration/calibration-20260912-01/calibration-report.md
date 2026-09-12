# 首轮评估器校准结果：未通过基础初筛

日期：2026-09-12。批次：calibration-20260912-01。15次真实DeepSeek评估请求均正常返回、评价文字完整；但CAL-05三次均出现明确增量判断错误。因此按已接受三条条件，本轮不能通过基础初筛，暂不将此评估器反馈用于自动策略候选生成。

本次结论来自组织者逐项对账及独立子代理对15条正文的只读复核，仍供用户作最终人审。没有改金标、规则或重跑，没有新增第16次。

## 各案例结果

| 案例 | 三次观察 | 对账结果 |
| --- | --- | --- |
| CAL-01 | 充分／是／是 | 3次核心判断及依据一致。 |
| CAL-02 | 不充分／否／是，但仅标题登记 | 3次第三维范围有争议，暂不计无争议通过；不能说它把标题当成期限依据。 |
| CAL-03 | 充分／否／否 | 3次核心一致；重复确认价值的附加措辞稍显过度。 |
| CAL-04 | 充分／否／是 | 3次核心一致，事实和位置此前已有，本次补引用登记。 |
| CAL-05 | 充分／是／是 | 3次明确误判；正确应为充分／无法判断／无法判断。 |

合计：9条核心一致、3条范围待复核、3条明确增量错误。不是60%整体准确率的声明，也不把范围争议自动计错或排除。42项此前离线测试只证明执行程序边界，本次评估器模型结果另行判断。

## CAL-05：有当前证据，不等于知道本次新增

输入给定history_prefix_complete=false且prior_events为空，表示比较基线缺失。当前正文完整并已登记，故当前支持充分；但是不能证明此前没有取得同样事实或引用依据。

第1次真实回复说：

> 前序为空且历史不完整，无法证明此前已有该事实；在现有可审记录范围内，本次补充了此前尚缺的期限事实。

随后把“新增必要事实”和“增加可引用支持”都判为是。它把无法证明此前已有，变成此前缺失。这是明确推理错误，不是输出截断或解析导致。

对应全文：[第1次005](C:/Users/hyf/Desktop/mini-agent-harness-lab/evaluation/results/tool-contribution-calibration/calibration-20260912-01/outputs-readable.md:103)、[第2次010](C:/Users/hyf/Desktop/mini-agent-harness-lab/evaluation/results/tool-contribution-calibration/calibration-20260912-01/outputs-readable.md:198)、[第3次015](C:/Users/hyf/Desktop/mini-agent-harness-lab/evaluation/results/tool-contribution-calibration/calibration-20260912-01/outputs-readable.md:292)。第2次承认无法比较却仍判新增；第3次同样用无法证明已有支持来判两项是。前两次的末尾限定没有纠正前面的肯定结论。

三次请求正文完全相同，actual result.content与原始HTTP回复逐字一致，API输入计数亦相同。因此本次可以排除请求正文被换掉或本地抽取把unknown改成yes这一类原因。还不能据此确定是模型能力、规则理解或其他机制的唯一因果，也不能推出mini-agent工具策略有错。

## CAL-02：一般登记与必要事实支持的范围不同

三次均明确指出收费标题不含期限，当前依据不足，也没有新增必要事实；这些判断正确。第三维却首答是，并限定仅增加标题本身的可引用登记。例如第3次说：

> 该可引用支持只覆盖“缴纳专利文件副本证明费”，不覆盖“提交办理请求一个月内缴费”这一必要事实。

对应全文：[第1次002](C:/Users/hyf/Desktop/mini-agent-harness-lab/evaluation/results/tool-contribution-calibration/calibration-20260912-01/outputs-readable.md:22)、[第2次007](C:/Users/hyf/Desktop/mini-agent-harness-lab/evaluation/results/tool-contribution-calibration/calibration-20260912-01/outputs-readable.md:145)、[第3次012](C:/Users/hyf/Desktop/mini-agent-harness-lab/evaluation/results/tool-contribution-calibration/calibration-20260912-01/outputs-readable.md:239)。就全文语义归一，可能与no_for_required_fact相容；按统一规则所问“支持必要事实的可引用能力是否增加”，首答又发生范围偏移。

依据用户已确认的争议复核出口，保留为范围待复核，不机械按标签认定事实错误，也暂不计为无争议通过。没有把人审参照改成是或用新口径回填本批。即使这些争议最终按等价表达接受，CAL-05的明确错误仍足以使本批未通过。

## 运行、输入及费用证据

- 模型请求15次、响应15次，全部finish_reason=stop，无超时、截断、自动重试或预算中止。三轮串行完成，约52.51秒。
- 每次API返回model=deepseek-flash，system_fingerprint均为aeb56401ca74e127821c4f9126dcb669；这记录可见一致性，不独立证明不可见权重版本。
- 五类输入tokens分别1888、1831、2087、2062、1815，三轮均与本地计数相同；总输入29049、输出8803、总tokens37852。所有发前请求快照与冻结正文哈希一致，无前次评价或纠正反馈。
- API报告缓存命中输入23808 tokens、未命中5241。运行在北京时间2026-09-12（周六）15:39:32至15:40:24，属于官方非高峰时段。

按[官方价格](https://api-docs.deepseek.com/zh-cn/quick_start/pricing/)及返回用量计算：5241×1/百万＋23808×0.02/百万＋8803×4/百万＝**0.04092916元**，约0.041元。没有查询账户账单，不能称已核验实扣。

程序出于预算保守性，按高峰且全部输入未缓存计算为**0.128522元**，未结算预留为0。两者计价口径不同，都低于已接受1元上限；本地计数＋最大输出的0.304元只是运行前规划计算。

证据：[程序汇总](summary.json)、[逐次用量与费用核验](mechanical-audit.json)、[逐条语义对账](semantic-review.json)、[全部原始评价正文](outputs-readable.md)。原始HTTP响应、发送前请求、结果与日志未覆盖。

## 凭据来源与启动经过

隔离目录缺.env，首次启动在配置阶段停止，实际模型请求0。跨项目来源在自动审批拒绝后没有被绕过；随后用户明确授权从原项目.env复用SEMANTIC_JUDGE_BASE_URL和SEMANTIC_JUDGE_API_KEY，才在进程内使用。未复制.env、未展示密钥；正式批次开始前才创建输出目录。

审批与冻结材料保留在规划目录calibration/batches/calibration-20260912-01-candidate；原2026-09-11-v1材料保留历史。实际请求与响应只写正式输出目录，不把结果塞回冻结案例文件。

## 下一步建议及当前停止位置

按已确认规则，本批不进入自动策略反馈／候选生成。下一步先处理评估器的CAL-05增量判定和CAL-02维度表述；不据此修改mini-agent工具调用策略，不断言需要换更强模型。

统一规则已经写明历史缺失不等于此前没有，因此不能直接把原因说成缺少这句话。应比较模型具体错误理由与规则约束，再决定是否需要更明确的判定步骤、说明或其他评估方式。此处只提出诊断方向，没有生成新规则或新策略候选。

若修改规则、参照、模型或参数，另立批次，不与本15次合并，不使用本批剩余预算自动追加请求。当前工具事件依旧是模拟；真实的是评估器的这些评价输出。一次基础校准不能证明泛化或策略自进化已经有效。
