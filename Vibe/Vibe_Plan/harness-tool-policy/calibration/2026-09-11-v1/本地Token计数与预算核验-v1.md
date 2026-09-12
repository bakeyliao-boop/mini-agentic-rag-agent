# 本地Token计数与预算核验 v1

日期：2026-09-12。状态：本地计数完成；没有运行模型或调用推理API，未验证API实测用量及账单。本轮是用户要求推进的本地核验，不能当成15次评估实验已执行。

## 结论

五份请求在固定官方分词器和消息编码格式下分别为1888、1831、2087、2062、1815输入tokens，包含消息边界。最大2087，低于每次12000输入tokens预留；本地最小余量为9913 tokens。

每例重复3次的本地输入量共29049 tokens。若15次输出都达到拟议2048-token上限，按已查的高峰未缓存单价计算约0.303858元，即约0.30元。该值是“本地输入计数＋拟议最大输出”的计算，不是实扣，不保证线上计数完全相同。

保留原每次12000输入tokens预留及1元总预算，不因为本地结果较小就自动削减安全余量或增加请求。原保守预留费用0.60576元继续可作为规划预留；后续实际费用仍由usage／账单及来源核对。费用控制尚未实现，执行权限仍未授权。

## 可回查结果

| 案例 | 规则正文 | 单例input内容 | 消息边界增量 | 单次本地输入tokens | 3次本地输入tokens |
| --- | --- | --- | --- | --- | --- |
| CAL-01 | 1306 | 577 | 5 | **1888** | 5664 |
| CAL-02 | 1306 | 520 | 5 | **1831** | 5493 |
| CAL-03 | 1306 | 776 | 5 | **2087** | 6261 |
| CAL-04 | 1306 | 751 | 5 | **2062** | 6186 |
| CAL-05 | 1306 | 504 | 5 | **1815** | 5445 |

完整数据及每份请求、编码文本的哈希见[计数结果JSON](本地Token计数结果-v1.json)。这些是token计数结果，actual_api_usage及actual_debit均为空；并非评估器对CAL案例的判断输出。cases.json中的actual_evaluator_result仍为null。

## 来源、版本及实际执行范围

固定官方仓库：[DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/tree/dba1be0a40aa45a94ad051997016db3960a90277)，commit `dba1be0a40aa45a94ad051997016db3960a90277`。只下载以下五个小文件，总计约6.43MB，没有下载模型权重。

| 文件 | 字节数 | SHA-256 |
| --- | --- | --- |
| tokenizer.json | 6367257 | `c90dfa01249db1be4245780a052ede752e1361c612ac6d08e2bdada7d599476b` |
| tokenizer_config.json | 801 | `6ac8c8dc065ed118161d02dd532749ae3f52c243deac27872134fae2f50d8547` |
| encoding/encoding.py | 37316 | `502bdaec8a3fd88ebc24c4721a7038fbe42f2063c664638127056107920035c1` |
| encoding/README.md | 12120 | `a2f0fc3baea318c9cfbceca68cbfe50d37cf7da6605ace887f33148bcff7e3ae` |
| inference/generate.py | 8722 | `8668d67f7d108e32b90d50cb0d8606889ceb2219bfe95741d84e22f70768e9f0` |

文件和依赖仅放在本次临时目录：`C:/Users/hyf/AppData/Local/Temp/harness-token-audit-20260912-dba1be0`。本地库tokenizers==0.22.2从PyPI二进制包安装到临时packages目录；Python 3.12.13。没有改项目依赖、虚拟环境、代码或配置。

先检查编码文件导入及纯文本路径，再调用其encode_messages；其顶层只使用typing/copy/json/re等标准库。仅执行消息字符串编码与Rust tokenizer计数，没有加载神经网络、访问项目配置或执行官方推理脚本。下载inference/generate.py只作参考核查。

首次在受限环境读取临时库时出现权限错误，未产生计数；随后在可读执行环境完成本地计算。没有修改文件ACL，没有将该错误算作评估模型失败。

## 消息格式与BOS/EOS核对

两条文本消息、非思考chat模式、无工具时，固定版本代码实际生成：

```text
<｜begin▁of▁sentence｜><｜System｜>{system正文}<｜User｜>{user原字符串}<｜Assistant｜></think>
```

分段间不人为加入换行，user内部JSON保持请求样例中的实际字符串。模型名称、temperature等外层HTTP字段不被当作提示正文计数；API可能有额外处理，见限制部分。

编码README的Basic chat示例省略初始System标记，而本轮实际下载的encoding.py第607行添加它；第723—729行追加Assistant及非思考结束标记，第779行添加BOS。采用固定代码实际结果，并与手工拼接逐字比较。网页缓存行号可能不同，不能把缓存位置当作本地坐标。

tokenizer_config.json设置add_bos_token=false、add_eos_token=false，tokenizer.json的post_processor为ByteLevel，未配置模板式BOS/EOS追加。本轮逐例验证add_special_tokens开／关所得token IDs完全相同，BOS恰好1个、EOS为0。五个边界字符串各编码为1个token，总封装增量5；不将尚未生成的回答或EOS算入输入。

[官方生成参考](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/dba1be0a40aa45a94ad051997016db3960a90277/inference/generate.py)采用encode_messages后tokenizer.encode；本轮使用相同官方tokenizer数据的底层tokenizers库，并显式核对特殊token行为，没有调用生成函数或下载其模型权重。

## 费用重算与边界

采用[官方中文价格](https://api-docs.deepseek.com/zh-cn/quick_start/pricing/)于2026-09-12已核对的高峰未缓存单价：输入2元／百万tokens、输出8元／百万tokens，不预先扣缓存优惠。

- 本地输入计算值：29049÷1000000×2＝0.058098元。
- 拟议最大输出：15×2048÷1000000×8＝0.24576元。
- 合计：0.303858元；同量非高峰价格约0.151929元。

以上输出量是上限假设，实际输出尚不存在；模式、max_tokens及客户端行为仍需确认并在接线中验证。不能将本地计算称作实扣金额，也不能把没有响应中的usage填成本地计数。

已验证的是：这五份固定请求在官方公开编码／分词器下均低于12000输入预留。未验证的是：线上API是否使用完全相同部署版本、包装与计数，以及实际账单是否相同。[官方Token说明](https://api-docs.deepseek.com/quick_start/token_usage/)要求以实际响应usage为准。本地计数明显留有余量，但不是线上输入绝对上限的证明。

未来获准的15次请求可逐次将API prompt_tokens与本地值对照，不需另加计数探针请求。发现不同模型/模式、超出12000预留、缺少费用依据或其他无法解释的预算风险时暂停。该方式是执行前检查建议，未因此启动首条请求。

## 材料静态一致性

- 5份请求文件哈希与原manifest一致；system和user内容未改，编码函数未改变messages对象。
- cases.json修订7、规则v2、答案示例v2、请求样例、v1/v2题集和冻结D01未修改。
- 本地计数结果写入独立报告及JSON，不回填为CAL案例的实际评估结果。
- 本轮没有项目测试、模型API、索引、候选生成、Git提交推送。只完成本地编码计数及规划留痕。

关联：[参数与请求组装草案](运行参数与请求组装草案-v1.md)、[请求样例](request-draft-v1/README.md)、[试运行方案v2](最小试运行方案草案-v2.md)。


## 独立复算回执

2026-09-12，子代理verify_v41_token_encoding未导入官方encoding.py，改用手工拼接同版本消息格式后独立分词。五例1888/1831/2087/2062/1815、15次合计29049，以及请求哈希、prompt哈希、字节数、分项计数均与主线程一致。BOS每例1个、EOS为0，add_special_tokens开关所得完整ID序列一致，核心文件哈希与下载清单一致。

独立复算没有调用模型API、修改项目文件/ACL或读取.env；仍不能作为线上usage或账单核验。
