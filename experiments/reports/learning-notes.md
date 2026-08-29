# MiniMind 微调源码学习笔记

## 数据、label 与 loss mask

Pretrain 是连续文本的 next-token prediction，绝大多数有效 token 都参与交叉熵。SFT 先通过 chat template 拼接 system/user/assistant；`SFTDataset` 只让 assistant 回复区 token 参与 loss，system、user、padding 被 ignore index 屏蔽。两者都是 Causal LM：位置 t 只能使用不晚于 t 的上下文预测 t+1。PPL 为 `exp(mean loss)`，只能衡量 token 预测，不直接等价于 JSON 或规划正确。

## 模型结构

- RMSNorm 按均方根缩放；RoPE 把位置旋转进 query/key。
- GQA 让多个 query head 共享较少 key/value head，降低 KV cache 与推理带宽。
- SwiGLU 是门控前馈网络；causal mask 防止训练时看见未来答案。

## 优化、显存与 LoRA

脚本使用 AdamW、学习率调度、梯度裁剪、混合精度和可配置梯度累积。有效 batch 为 `batch_size × accumulation_steps × world_size`，本次受控实验固定为 16。

Full SFT 为 63.912M 参数保存梯度和 Adam 状态。LoRA 冻结 base，仅在方形 Linear 层增加 rank-16 的 `BA` 增量，本次只有 0.393M（0.61%）参数可训练。实测峰值显存为 5038 vs 7232 MiB；适配器文件 797098 B，Full 权重 137684380 B。

## Checkpoint 与 resume

普通 `.pth` 用于推理；严格 resume 还需 model、optimizer、scaler、epoch 和 step，才能恢复动量、loss scale 与学习率进度。只加载模型权重重新训练不等于严格续训。阶段 1 从 step 9000 的本人 resume checkpoint 迁移恢复。

## 为什么任务指标优先

UAV Full SFT 最终 train loss 0.0045，但 test 语义准确率仍为 35%；LoRA loss 0.1433，test JSON 合法率 88%、安全拒绝为 0%。loss 是 teacher-forcing token 平均误差，不能覆盖自回归累积、严格 schema、数值几何和风险模板泛化，因此必须保留独立 split、逐样本输出和失败分类。
