# MiniMind 复现实验

本目录保存 MiniMind-3 复现与无人机领域适配的可审计证据。大体积数据、模型权重和 checkpoint 不进入 Git，只在清单中记录相对路径、文件大小与 SHA-256。

## 目录

- `manifests/`：代码、环境、硬件、数据与模型快照。
- `configs/`：每次训练或评测的完整参数。
- `logs/`：保留原始运行日志和异常记录。
- `metrics/`：机器可读的逐样本结果与汇总指标。
- `reports/`：曲线、失败分析和阶段报告。
- `ROADMAP.md`：从当前状态到简历项目完成度的执行路线。
- `MATH_PROJECT.md`：数学后训练项目入口、复现方法、结果与口径边界。

## 当前基线

- 环境：`minimind`
- Python：3.11.15 64-bit
- PyTorch：2.6.0+cu124
- GPU：NVIDIA GeForce RTX 4060 Laptop GPU 8 GB
- CUDA 可用：是
- 上游代码快照：`d65ef2c00ebc6082f9df11541e1b191655eddb00`
- 工作分支：`codex/minimind-repro`

当前已完成两条可审计实验主线：无人机领域 Full SFT / LoRA 适配，以及单张 A10 上的 MiniMind 数学 SFT warm-up + 规则奖励 GRPO 全流程。数学实验完整覆盖 17,176 道去重 DAPO 题目并完成保存、断点恢复、严格重载和固定评测；AIME 仍为 0/30，因此只主张工程闭环，不主张数学推理能力提升。
