# MiniMind 数学后训练项目

## 项目定位

在单张 NVIDIA A10 上，为 63.9M 参数 MiniMind 构建可审计的数学后训练流程：公开数据固定与去重、少量数学 SFT warm-up、可验证整数答案奖励、clipped GRPO、零优势保护、独立超时监督、checkpoint/resume、严格重载和独立评测。

个人工作集中在训练与评测工程。MiniMind 模型结构来自上游项目；本项目没有声称自研模型架构，也不是完整复现 DAPO 论文算法。

## 最终结果

| 项目 | 结果 |
|---|---:|
| 模型 | MiniMind 63,912,192 参数 |
| 硬件 | 单张 NVIDIA A10 |
| DAPO 有效去重题目 | 17,176 |
| 采样回答 | 137,408 |
| 可解析回答 | 136,674（99.466%） |
| 规则奖励正确回答 | 3,994（2.906%） |
| 产生非零参数更新的题组 | 2,428 |
| 完整覆盖 | 17,176 / 17,176 |
| 训练墙钟 | 9,280.741 秒（154.68 分钟，含断点续跑） |
| AIME 2024 greedy | 0/30 → 0/30 |
| GSM8K 64题留出 greedy | 1/64 → 1/64 |
| 最终权重严格重载 | 通过，逐张量一致 |

工程链路成功；现有独立评测没有证明数学泛化能力提升。详细解释见 [训练后检查](reports/math-post-eval-20260901.md)。

## 代码入口

- `scripts/prepare_math_data.py`：固定DAPO/AIME版本，去重、冲突和评测重叠审计。
- `scripts/prepare_math_warmup.py`：固定GSM8K人工解答和DAPO答案格式样本。
- `scripts/math_warmup.py`：assistant-only监督、EOS/padding mask和保存重载。
- `scripts/math_pilot.py`：规则奖励、clipped GRPO、零优势不更新、断点与评测。
- `scripts/run_math_readiness.py`、`scripts/run_math_full.py`：累计预算、独立watchdog、显式全量授权和防重复启动。
- `scripts/math_checkpoint_eval.py`：warm-up/final checkpoint留出集对照。
- `scripts/test_math_*.py`：14项奖励、mask、梯度方向和全量保护测试。

完整实验参数在 `configs/math-grpo-full-*.json`，紧凑机器可读结果在 [math-final-summary-20260901.json](metrics/math-final-summary-20260901.json)。

## 数据与产物

GitHub 不保存大权重、原始数据和逐题日志。它们已下载到本地并逐项验证SHA256：

- 本地归档：`out/math-archive-20260901/`（Git忽略）。
- 最终模型：`experiments/checkpoints/math-grpo-full-20260901.pth`（Git忽略）。
- 可上传哈希清单：[math-local-archive-20260901.json](manifests/math-local-archive-20260901.json)。
- 最终模型SHA256：`82b8e338466e238688125c7a480d699a6213ab62b88f3149fb8b3eb25c73d4bb`。

归档包含通用SFT起点、数学warm-up权重、最终权重、含优化器/RNG的恢复断点、官方原始parquet、精确处理后的DAPO/AIME和GSM8K数据。17个文件共约1.68 GiB，全部SHA匹配。

## 复现与验证

在项目根目录、具备项目依赖和CUDA的环境执行：

```bash
python -m unittest discover -s scripts -p 'test_math*.py' -v
python scripts/run_math_full.py --config experiments/configs/math-grpo-full-pending.json --dry-run
```

数据来源、版本与许可见 [DAPO研究](reports/dapo-math-17k-research.md) 和 [warm-up来源](reports/math-warmup-sources.md)。全量运行历史、失败上限和续跑证据见 [完整运行报告](reports/math-full-20260831.md)。

## 表述边界

- 可以说完成了规则奖励GRPO数学后训练工程闭环。
- 可以说完整覆盖17,176题、产生2,428次真实非零更新并验证断点恢复。
- 不能说复现了完整DAPO算法。
- 不能说数学能力、AIME或推理能力得到提升。
- 不能把训练集在线正确率当作独立泛化指标。
