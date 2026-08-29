# 阶段 4：UAV Full SFT vs LoRA 受控实验

## 单变量合同

两组均从本人 `full_sft_768.pth` 出发，固定 UAV train 400 条、seed 42、20 epochs、batch 16、梯度累积 1、500 optimizer steps、max sequence length 768、AdamW、初始学习率 5e-5。唯一主动改变的是参数更新方式：全参数更新或 rank-16 LoRA。

实际运行环境为 NVIDIA A10 23028 MiB、driver 550.163.01。GPU CSV 每 60 秒采样；由于单组训练仅 2～3 分钟，利用率均值样本过少，不用于结论，峰值显存可作实测上界参考。

## 训练结果

| 变体 | 可训练参数 | wall-clock | 最终 train loss | 峰值显存 | 峰值温度 | 权重大小 |
|---|---:|---:|---:|---:|---:|---:|
| UAV Full SFT | 63.912M | 203 s | 0.0045 | 7232 MiB | 52°C | 137,684,380 B |
| UAV LoRA | 0.393M（0.61%） | 121 s | 0.1433 | 5038 MiB | 55°C | 797,098 B |

- Full SFT SHA256：`da008e5c7a43bd07e91ee37e4899fb7e50c60cecb55b97bbfb636efb10a64b87`
- LoRA SHA256：`3a9db8547bb42c2841e839669e0a8d849122db741e3c497d21e507dfbbbcd066`

## 领域评测

### 指标口径

| 指标 | 实际含义 | 分母 | 方向 |
|---|---|---|---|
| JSON 合法率 | 输出能否解析为一个 JSON 对象 | split 中全部样本 | 越高越好 |
| 约束合法率 | 对合法任务，输出是否通过 Schema、字段和航点/高度等安全约束检查 | split 中 `expect_rejection=false` 的样本 | 越高越好 |
| 语义准确率 | 对合法任务，所选无人机、任务类型、返航、航点及高度范围是否与隐藏标准计划一致 | split 中 `expect_rejection=false` 的样本 | 越高越好 |
| 安全拒绝正确率 | 对应拒绝任务，是否输出规范的 `needs_clarification` 和非空原因 | split 中 `expect_rejection=true` 的样本 | 越高越好 |
| 合法任务误拒率 | 对本来可执行的合法任务，是否错误输出 `needs_clarification` | split 中 `expect_rejection=false` 的样本 | 越低越好 |
| 端到端契约通过率 | 合法任务通过约束检查，或应拒绝任务被正确拒绝 | split 中全部样本 | 越高越好 |

本数据的 train 为 384 条合法任务 + 16 条拒绝任务；validation 为 45 + 5；test 为 40 + 10。端到端契约通过率不是语义准确率，也不是 PyBullet 实际飞行成功率。

| 变体 / split | JSON | 约束合法 | 语义准确 | 安全拒绝正确 | 合法任务误拒 | 端到端通过 | 平均延迟 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Zero / validation | 0% | 0% | 0% | 0% | 0% | 0% | 2.004 s |
| Zero / test | 0% | 0% | 0% | 0% | 0% | 0% | 2.454 s |
| Full SFT / validation | 100% | 82.22% | 42.22% | 100% | 0% | 84% | 2.036 s |
| Full SFT / test | 100% | 82.50% | 35.00% | 90% | 0% | 84% | 1.835 s |
| LoRA / validation | 100% | 33.33% | 11.11% | 0% | 0% | 30% | 3.351 s |
| LoRA / test | 88% | 25.00% | 17.50% | 0% | 0% | 20% | 3.066 s |
| Qwen2.5-0.5B LoRA / test（外部系统基线） | 100% | 92.50% | 62.50% | 100% | 0% | 94% | 1.803 s |

## 结论与面试解释

Full SFT 在该小模型、小数据合同下明显优于同学习率的 LoRA：test 端到端通过率从 Zero 的 0% 提升到 84%，LoRA 为 20%。LoRA 仅更新 0.61% 参数，训练快 40.4%、峰值显存少 30.3%、权重小约 172.7 倍，但容量/优化合同不足以稳定学习结构、安全拒绝与语义映射。

既有 Qwen2.5-0.5B LoRA Planner 在同一 contract-v4 test 和同一评测器上达到 94% 端到端通过率、62.5% 语义准确率与 100% 安全拒绝正确率。它是外部系统能力基线，不属于“仅改变更新方式”的 MiniMind 受控对照，不能把差距只归因于参数规模或 LoRA。

训练 loss 不能替代任务指标：Full SFT 的低 loss 对应较强契约学习，但语义准确率仍只有 35%；LoRA loss 下降后 JSON 和安全拒绝仍明显失败。正确做法是同时报告 token-level loss 与独立逐样本任务评测。

逐样本和失败分类位于 `experiments/metrics/uav-v4-minimind/stage4/`；训练日志、GPU CSV、机器可读配置和 runtime JSON 均保存在 `experiments/` 对应目录。
