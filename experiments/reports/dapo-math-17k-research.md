# DAPO-Math-17K：训练前资料核实

核实日期：2026-08-31。仅查阅公开资料、仓库元数据和少量预览；未启动训练、连接 GPU 服务器或下载全量数据。本文区分官方发布事实、其他维护者的处理结果与本项目尚待实测项。

## 直接影响需求的结论

**“全量”应先对齐为覆盖去重后的完整题目集合，而非机械遍历发布文件的全部重复行。**官方当前文件有 1,791,700 行；官方历史 README 明确承认约 17K 唯一 prompts 被意外重复约 100 倍，并为可复现保留原状。这是发布方历史声明，不只是社区猜测；但不能据此断言有恰好 17,917 个内容唯一题。去重键不同会得到不同数量。[官方数据页](https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k)、[官方历史说明](https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k/blob/6b4e85df671ac51e3fdb81d10d0639f1e7e6b4f2/README.md)

**它可作为可验证奖励 RL 的题目来源，但不能直接当带推理过程的 SFT 数据。**发布 schema 没有参考推理轨迹；若起点需数学 SFT，应另外确定解答来源与预算。**选用 DAPO 数据不等于必须复现 DAPO 算法。**[官方预览](https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k/viewer)、[项目算法介绍](https://dapo-sia.github.io/)

## 版本、结构与规模

| 项目 | 本次核实结果 |
| --- | --- |
| 发布方 / 仓库 | BytedTsinghua-SIA / DAPO-Math-17k；DAPO 项目来自 ByteDance Seed 与 Tsinghua AIR |
| 当前 revision | `65877096c24ffa7abc4e4fa5edb95cf3413a5674` |
| 最后修改 | `2025-04-18T11:20:51Z`，以查询当日 API 为准 |
| split | `default/train`，只有一个；不是内置 train/test 切分 |
| 文件 | `data/dapo-math-17k.parquet`；网页约 299 MB，1,791,700 行 |
| 许可证标注 | 数据卡元数据 `apache-2.0`；这里只记录发布标注，不作权利链法律保证 |
| 数据卡信息量 | 当前 README 只有 YAML 元数据，正文说明已移除 |

依据：[仓库 API](https://huggingface.co/api/datasets/BytedTsinghua-SIA/DAPO-Math-17k)、[split API](https://datasets-server.huggingface.co/splits?dataset=BytedTsinghua-SIA%2FDAPO-Math-17k)、[固定版本 README](https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k/blob/65877096c24ffa7abc4e4fa5edb95cf3413a5674/README.md)、[官方项目仓库](https://github.com/BytedTsinghua-SIA/DAPO)。本次未独立扫描 Parquet 验证行数。

字段为：`data_source` 字符串、`prompt`（含 `role/content` 的消息列表）、`ability` 字符串、`reward_model`（字符串 `ground_truth/style`）、`extra_info`（字符串 `index`）。预览通常是单条 user 消息，来源 `math_dapo`，能力 `MATH`，style 为 `rule-lighteval/MATH_v2`；要求答案独占一行并以 `Answer:` 开头。`ground_truth` 为答案而非推理解答；UUID 也不能自动视作内容唯一键。[官方 schema 与预览 API](https://datasets-server.huggingface.co/first-rows?dataset=BytedTsinghua-SIA%2FDAPO-Math-17k&config=default&split=train)

预览可见负整数 `-3` 以及 `1007`、`40320` 等答案；**不能把答案限制为 AIME 的三位非负整数格式**。全量答案范围和异常比例尚未统计。[官方数据预览](https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k/viewer)

## 原题、改写、重复和质量边界

论文 §3.5 说明题源为 AoPS 和竞赛官网，经抓取、人工标注、筛选与 LLM 改写，使答案成为便于规则验证的整数。附录提供改写示例。这一“答案整数化”的变换与发布文件约百倍重复是两件事；没有证据证明重复行都是新生成的独立题。[论文 §3.5 与附录 A](https://dapo-sia.github.io/static/pdf/dapo_paper.pdf)、[官方重复说明历史版本](https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k/blob/9f6440001c15da8e7c7516fdbb3d2ce49de711de/README.md)

另一维护者 open-r1 发布的处理版声明做了 prompt 去重和 TRL 格式转换；其当前数据卡列出 `all=17,398`、`en=14,116`、`cn=3,282`。这是一手描述**该维护者处理版**的事实，不是原发布方保证，也不是本项目实测。原卡虽然标 `en`，不应据此自动丢弃中文题；处理版 `solution` 在预览中仍只是最终答案。本文不建议未经用户确认改用该版本。[open-r1 数据卡与处理逻辑入口](https://huggingface.co/datasets/open-r1/DAPO-Math-17k-Processed)

官方仓库社区讨论中，有用户自报按内容及答案去重得到 17,405 条，再排除同题答案冲突得到 17,391 条。这只是用户自己的处理报告，不能写成本项目结论或作者认可。它提示后续至少保留机械去重、同题答案冲突审计及排除台账；不需要因此推翻用户选定的数据集，但也不能承诺数据绝无错误。[社区原始报告](https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k/discussions/3)

## 评测与奖励

官方项目将 AIME 2024 用作验证；HF 的该仓库也只有名为 `train` 的 split，显示 960 行，预览有重复题。论文采用每题重复采样 32 次的评测口径。960 行不能视为 960 道独立题；本项目应先按题目去重并记录实际题数，再明确每题采样次数。[官方 AIME 数据](https://huggingface.co/datasets/BytedTsinghua-SIA/AIME-2024)、[官方项目评测说明](https://github.com/BytedTsinghua-SIA/DAPO#evaluation-on-aime-2024)

本次读取的论文、当前数据卡与项目 README 中，未找到可以直接复用的完整去污染清单、原题映射或无重叠保证。**未找到说明不等于证明存在污染**。如自行划分验证集，应先去重再按题目切分；改写近重复还需单独考虑，不能随机切重复行。[论文](https://dapo-sia.github.io/static/pdf/dapo_paper.pdf)、[官方数据卡](https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k)

官方 `eval/math_dapo.py` 的默认路径从回复末尾 300 字符中提取最后一次 `Answer:`，规范化后与 ground truth 字符串比较；正确 `+1`、错误 `-1`，另有可选 boxed 路径。它不是通用学习型奖励模型，也不是万能数学等价判定器；改动解析规则必须与评测保持一致并留记录。[官方奖励/评测代码](https://github.com/BytedTsinghua-SIA/DAPO/blob/main/eval/math_dapo.py)

## 对本项目的含义：事实与推断分开

官方 DAPO 包含 Clip-Higher、动态采样、token 级策略梯度与超长奖励处理；动态采样会过滤组内全对或全错的题。论文使用 Qwen2.5-32B，不能将其收益或吞吐直接外推到 MiniMind 64M + 单 A10。[项目算法介绍](https://dapo-sia.github.io/)、[论文 §3–4](https://dapo-sia.github.io/static/pdf/dapo_paper.pdf)

**本项目推断/建议：**先检验可解析答案率、正确率、组内奖励方差及有效更新。如果几乎全错，动态采样可能迟迟凑不齐有效组，应有硬停止条件；可按已同意的分支讨论数学 SFT，而不是让零优势流程无限空转。全量目标需要区分“题目被 rollout 覆盖”与“题目实际贡献梯度”，不能保证全错题都参与有效梯度更新。

主任务本地源码审查还发现：`train_grpo.py` 默认 `loss_type=cispo`，奖励使用通用 `LMForRewardModel` 及附加规则；`RLAIFDataset` 要求 `conversations` 并返回空答案。故不能只把 `data_path` 换为上述 Parquet 就称为数学 RL 或 DAPO 复现。此项由主任务同日只读审查提供，本文不实施修改。[训练入口](<D:/CODE/Personal devs/minimind/trainer/train_grpo.py>)、[数据适配](<D:/CODE/Personal devs/minimind/dataset/lm_dataset.py>)

**待小规模实测后才可决定：**去重后的真实题数、语言和 token 长度分布、截断比例、每题采样数、生成及更新耗时、有效组比例、峰值显存、检查点和评测开销。全量时间不能仅凭 17K 名称或 A10 型号给出；先分别记录 rollout 覆盖速度与有效更新速度，再按确认后的覆盖口径估算区间。数学 SFT 或教师生成若触发，应另计预算。本文没有任何训练耗时或效果实测结果。
