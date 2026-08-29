# 阶段 1：MiniMind Pretrain 复现

Dense 64M 模型从随机初始化开始，以 `pretrain_t2t_mini.jsonl` 训练 1 epoch。早期 9000 step 含本地运行，随后从本人 resume checkpoint 迁移到 A10 完成；因此 6267 秒是远程 `9000 → 39695` 段的可审计耗时，不冒充完整 epoch 总耗时。

- 总进度：39695/39695，exit code 0。
- 远程恢复段：2026-08-28 17:32:13～19:16:40，6267 秒。
- 首次本地 500 step：loss 7.5222 → 6.2078，无 OOM/NaN。
- 最终 loss 2.0220；日志采样最低 1.8691（step 32100）。
- validation：128 条、26921 supervised tokens，loss 2.0310，PPL 7.6219。
- 远程峰值显存 6832 MiB、峰值温度 62°C、平均 GPU 利用率 97.14%。
- `pretrain_768.pth` SHA256：`5b7a32e65505d68acef34285223feefa1eaf3d84c4571d3dc4f233d8e4280e74`。

固定样例证明 checkpoint 可加载和生成，但 Pretrain 只学习语言建模，不应期待稳定的指令遵循。
