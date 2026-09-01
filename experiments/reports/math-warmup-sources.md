# 数学 warm-up 数据来源核实

核实日期：2026-08-31。范围仅为免费公开数据的一手来源调查；本报告不代表已执行下载、去污染或训练。

## 可执行选择

**用 GSM8K main 的原始 train 解答做小规模数学 SFT；必要时补充已有 DAPO 训练题的答案格式监督。**不需要付费教师。GSM8K 是人工编写的多步骤小学应用题，难度适合先检查小模型能否学习解答与答案格式；是否能帮助 MiniMind 在 DAPO 竞赛题上产生非零奖励必须实测，不能从数据介绍推定。[官方仓库](https://github.com/openai/grade-school-math)

建议本轮仅选最多 1,024 条 GSM8K train 与最多 512 条已有 DAPO 有效训练题。两个来源分别标记；固定种子、题目哈希、原始行号和最终入选清单。样本上限属于项目方案，不是数据集作者建议。不启动全量 SFT。

## GSM8K：固定来源、规模与许可

| 项目 | 核实结果 |
| --- | --- |
| 原始仓库 | openai/grade-school-math |
| 固定 Git revision | `3101c7d5072418e28b9008a6636bde82a006892c` |
| 原始训练文件 | `grade_school_math/data/train.jsonl`，网页显示 3.97 MB |
| 推荐固定下载 | [train.jsonl 原始地址](https://raw.githubusercontent.com/openai/grade-school-math/3101c7d5072418e28b9008a6636bde82a006892c/grade_school_math/data/train.jsonl) |
| HF 官方备选 | openai/gsm8k，revision `740312add88f781978c0658806c59bc2815b9866`，config `main` |
| HF 训练文件 | `main/train-00000-of-00001.parquet`，网页约 2.31 MB |
| split / 数量 | `train` 7,473；`test` 1,319。没有独立 validation split；卡片正文旧表把 test 写作 validation，应以配置元数据为准 |
| 字段 | `question`、`answer`，均为字符串；英文 |
| 许可 | MIT；原仓库 LICENSE 标注 Copyright (c) 2021 OpenAI |

版本依据：[GitHub commit](https://github.com/openai/grade-school-math/commit/3101c7d5072418e28b9008a6636bde82a006892c)、[固定文件](https://github.com/openai/grade-school-math/blob/3101c7d5072418e28b9008a6636bde82a006892c/grade_school_math/data/train.jsonl)、[HF commit](https://huggingface.co/datasets/openai/gsm8k/commit/740312add88f781978c0658806c59bc2815b9866)。数量/字段依据：[官方 HF 数据卡元数据](https://huggingface.co/datasets/openai/gsm8k/raw/main/README.md)。体积依据：[HF 文件目录](https://huggingface.co/datasets/openai/gsm8k/tree/main/main)。许可依据：[原始 LICENSE](https://github.com/openai/grade-school-math/blob/master/LICENSE)。随数据派生材料保留来源与许可；这里仅记录官方许可文本，不作独立权利链保证。

可用备选固定下载：[HF train Parquet](https://huggingface.co/datasets/openai/gsm8k/resolve/740312add88f781978c0658806c59bc2815b9866/main/train-00000-of-00001.parquet)。下载后仍需记录实际文件 SHA-256、字节数、行数；本报告未下载文件，不提供虚构校验和。

## 解答转换

原始 `answer` 包含自然语言逐步解答、`<<...>>` 计算标记和末行 `#### 数值`。官方说明允许删除计算标记；本项目未部署计算器，不应声称执行了这些标记里的代码。不用 `socratic` 变体，也不用 `example_model_solutions.jsonl`；后者是对测试题生成的模型解答，不能混入训练。[官方格式与变体说明](https://github.com/openai/grade-school-math#dataset-details)

本项目方案：以原始解答为依据，仅移除 `<<...>>` 标记，保留其外部计算结果和自然语言步骤，把 `####` 最终答案规范化成独立末行 `Answer: 整数`。验证最终数值可解析后再纳入；不能把非整数悄悄截断或四舍五入。保留源答案供审计；异常记录单独排除并计数。不要伪造新的推理步骤。仅 assistant 答案 token 参与 SFT loss，按现有模板构造 user 问题。转换细节是项目决策。

## DAPO 仅答案监督的边界

继续使用项目已锁定 revision `65877096c24ffa7abc4e4fa5edb95cf3413a5674` 的已有去重训练数据，不重新下载官方全量文件。官方预览中的 `reward_model.ground_truth` 是最终答案，未提供推理过程；例如存在负整数和大于三位数的答案。原提示要求独立末行 `Answer:`。数据卡标注 Apache-2.0。[固定数据卡](https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k/blob/65877096c24ffa7abc4e4fa5edb95cf3413a5674/README.md)、[官方字段与样本](https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k)

项目可将少量训练题和 `Answer: ground_truth` 配对作为**答案格式监督/答案记忆 warm-up**。这不提供该题推理轨迹，也不是自蒸馏。不把同题后续命中当成未见题泛化；DAPO warm-up 题与本轮诊断题按规范化题目哈希隔离，而不只按可能变化的 ID 隔离。若原 user 提示要求逐步推理而 assistant 只有答案，应明确这是受限的格式实验，不能据此宣称完成推理 SFT。

## AIME 与诊断隔离

项目方案：AIME 2024 固定评测题不参与 SFT、RL 或训练样本筛选；GSM8K 只取原始 train，保留 test。新增 warm-up 样本继续对 AIME 做规范化文本去重与近重复检查，保留检查日志。GSM8K 原始提交日期早于 AIME 2024，[提交记录](https://github.com/openai/grade-school-math/commit/3101c7d5072418e28b9008a6636bde82a006892c)可证明版本早晚，但不能证明完全无语义重合，也不能排除基础权重原训练污染。

DAPO 诊断题的隔离仅表示本轮未用于 SFT，不把它称作最终独立测试集。若后续经批准进行 DAPO 全量训练，这些题可回到训练覆盖范围；AIME 仍保持隔离。小样本应同时报告可解析率、正确率、组内奖励方差、非零优势组数、有效更新数；仅格式正确或某个常见答案偶然命中都不足以证明学会数学。结论应区分“优化流程已验证”和“全量训练具有足够真实奖励信号”。