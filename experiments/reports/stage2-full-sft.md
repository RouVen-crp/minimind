# 阶段 2：通用 Full SFT 复现

唯一起点是本人的 `pretrain_768.pth`；执行 `python -u train_full_sft.py --epochs 1 --from_weight pretrain`，未使用官方模型权重。

- 进度：56608/56608，exit code 0。
- 运行：2026-08-28 20:49:04～2026-08-29 00:33:26，13462 秒（3:44:22）。
- loss：step 100 为 2.2232，step 500 为 1.9706，最终 1.7028；日志采样最低 1.4397（step 48700）。
- validation：64 条、26243 supervised tokens，loss 1.6457，PPL 5.1846。
- 峰值显存 7332 MiB、峰值温度 62°C、平均 GPU 利用率 98.20%。
- `full_sft_768.pth` SHA256：`ceb382eaaaa98587c528e1ae3cc830603f55a221bd2766212cb40dcefd8420a1`。

固定样例已能产生连贯问答，但仍有事实错误和截断。该 checkpoint 被锁定为 UAV Zero、Full SFT、LoRA 的共同起点。
