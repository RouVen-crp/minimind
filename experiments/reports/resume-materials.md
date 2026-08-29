# 中文简历素材（仅实测数字）

## 推荐项目标题

MiniMind 64M 从头训练与无人机领域微调评测

## 三条版

- 基于 MiniMind 复现 64M Dense Causal LM 训练链路，从随机初始化完成 1 epoch Pretrain 与个人 checkpoint 上的 Full SFT；通用验证 PPL 从 7.62 降至 5.18，并用 SHA256、runtime、GPU CSV 和固定样例建立可审计证据。
- 构建 400/50/50 UAV contract-v4 无泄漏数据与统一评测适配，覆盖 JSON、约束、语义、安全拒绝和端到端指标；领域 Full SFT 将 blind test 端到端通过率从 0% 提升至 84%，JSON 合法率 100%、安全拒绝正确率 90%。
- 设计 Full SFT vs rank-16 LoRA 单变量实验：固定 base/data/seed/batch/500 steps/LR，LoRA 仅训练 0.393M（0.61%）参数，较全参训练快 40.4%、峰值显存低 30.3%、权重小 172.7 倍；结合 84% vs 20% 端到端结果分析参数效率与能力边界。

## 一条精简版

- 复现 MiniMind 64M Pretrain→SFT 主链并完成 UAV 领域迁移；建立无泄漏逐样本评测与 Full SFT/LoRA 受控实验，Full SFT blind-test 端到端通过率 84%，LoRA 以 0.61% 可训练参数实现 20%。

## 不应写

- 不写“自主研发 MiniMind 架构”；模型结构来自上游。
- 不写“完整 epoch 仅 6267 秒”；该数字只是 step 9000 后的远程恢复段。
- 不写“LoRA 普遍不如 Full SFT”；只能陈述本实验合同。
- 不写“无人机端到端飞行成功率 84%”；84% 是 Planner 契约通过率。
