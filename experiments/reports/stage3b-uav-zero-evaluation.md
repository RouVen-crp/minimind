# 阶段 3B：MiniMind Zero 无人机评测

## 实验合同

- 模型：本人 Pretrain → 通用 Full SFT 得到的 `full_sft_768.pth`，未使用官方权重。
- 数据：UAV contract-v4 validation 50 条；冻结贪心解码参数后，test 50 条仅执行一次。
- 解码：`do_sample=false`、`max_new_tokens=512`、batch size 1、单次生成、不重试。
- 评测：复用 `drone_planner` 的 JSON、约束、语义、安全拒绝和端到端契约口径。

## 结果

| split | JSON 合法率 | 约束合法率 | 语义准确率 | 安全拒绝正确率 | 合法任务误拒率 | 端到端通过率 | 平均延迟 |
|---|---:|---:|---:|---:|---:|---:|---:|
| validation | 0% | 0% | 0% | 0% | 0% | 0% | 2.004 s |
| blind test | 0% | 0% | 0% | 0% | 0% | 0% | 2.454 s |

结论：通用 SFT 后模型能够生成自然语言，但没有学习 UAV JSON 契约；失败发生在输出结构层，后续语义和安全指标因此全部归零。这是领域训练的有效 Zero 基线，不应解释为推理链路故障。

逐样本、任务分类和失败分类位于 `experiments/metrics/uav-v4-minimind/`。
