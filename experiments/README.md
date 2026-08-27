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

官方权重推理验收已完成。下一步从 [ROADMAP.md](./ROADMAP.md) 的阶段 1（确定性数据拆分与冒烟集）开始。
