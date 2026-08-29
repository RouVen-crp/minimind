# 面试问答

## 1. 你复现了什么，个人工作是什么？

上游提供 MiniMind 架构和训练脚本。我完成环境与哈希证据、随机初始化 Pretrain、个人 checkpoint 上的通用 Full SFT、断点迁移、validation loss/PPL、UAV 数据转换与防泄漏、统一评测，以及 Full SFT vs LoRA 单变量实验。

## 2. 为什么不能用官方权重写成“从头训练”？

官方权重只能证明环境可推理。我的主链是随机初始化 → 本人 `pretrain_768.pth` → 本人 `full_sft_768.pth`，每个 checkpoint 都有来源和 SHA256。

## 3. Pretrain 与 SFT 的核心差别？

两者都是 causal next-token loss；Pretrain 基本让所有文本 token 参与 loss，SFT 只监督 assistant 回复，prompt token 用 ignore index 屏蔽。

## 4. label 为什么右移？

位置 t 的 hidden state 预测 token t+1；logits 去掉最后位置，labels 去掉第一个位置后计算交叉熵。

## 5. 梯度累积解决什么？

用多次 micro-batch 前后向换取一次 optimizer step，在显存不变时增大有效 batch；更新前再做梯度裁剪。

## 6. Full SFT 与 LoRA 如何受控？

共同 base、400 条数据、seed 42、batch 16、500 steps、20 epochs、length 768、AdamW 和 LR 5e-5；只改变全参数或 rank-16 低秩参数更新。

## 7. LoRA 为什么省显存？

base 冻结，不保存其梯度和 Adam 状态。本次可训练参数从 63.912M 降到 0.393M，峰值显存从 7232 降到 5038 MiB。

## 8. 为什么 LoRA 比 Full SFT 差？

结论只适用于固定合同。相同 LR 未必分别最优；64M base、rank 16、500 steps 可能限制容量。受控实验不能证明 LoRA 普遍更差。

## 9. 为什么 loss 很低，语义仍不高？

大量易预测结构 token 会稀释少量关键坐标错误，推理还有自回归误差累积。Full SFT loss 0.0045，但 test 语义准确率只有 35%。

## 10. 如何防 test 泄漏？

只派生 train/validation，test 保留在 UAV 项目；检查 split ID 与归一化 instruction hash 交集为零。validation 冻结方案后，每个变体 test 一次。

## 11. 如何区分 Planner 与控制器失败？

先评估 JSON、schema、语义和拒绝，语义正确才进入仿真，避免把 Planner 错误和 PID/TD3 控制错误混在一起。

## 12. 遇到哪些工程问题？

Windows PyTorch DLL 污染、断点迁移、进程防重复、评测文件覆盖、LoRA 的 FP16/FP32 dtype 不一致；原则是先保全 checkpoint 和证据，再做最小修复和回归测试。
