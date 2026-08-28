# MiniMind 复现实验

本目录保存 MiniMind-3 复现与无人机领域适配的可审计证据。大体积数据、模型权重和 checkpoint 不进入 Git，只在清单中记录相对路径、文件大小与 SHA-256。

## 目录

- `manifests/`：代码、环境、硬件、数据与模型快照。
- `configs/`：每次训练或评测的完整参数。
- `logs/`：保留原始运行日志和异常记录。
- `metrics/`：机器可读的逐样本结果与汇总指标。
- `reports/`：曲线、失败分析和阶段报告。
- `ROADMAP.md`：从当前状态到简历项目完成度的执行路线。

## 当前基线

- 环境：`minimind`
- Python：3.11.15 64-bit
- PyTorch：2.6.0+cu124
- GPU：NVIDIA GeForce RTX 4060 Laptop GPU 8 GB
- CUDA 可用：是
- 上游代码快照：`d65ef2c00ebc6082f9df11541e1b191655eddb00`
- 工作分支：`codex/minimind-repro`

Pretrain 已迁移至远程 A10，并从 step 9000 断点继续第 1 个 epoch。无人机阶段 3A 的数据转换和评测适配已完成；真实 MiniMind Zero 评测等待阶段 2 的个人 Full SFT checkpoint。
