# MiniMind-3 复现与无人机领域适配行动路线

## 当前状态

- [x] 创建独立 Conda 环境 `minimind`。
- [x] 安装并验证 `torch==2.6.0+cu124`。
- [x] CUDA 可用，GPU 为 RTX 4060 Laptop 8 GB。
- [x] 下载 Pretrain mini、SFT mini 与官方 `minimind-3` 模型。
- [x] 固定上游代码快照并建立个人 Fork 分支。
- [x] 保存环境、硬件、数据和官方模型证据。

基线证据见 `manifests/baseline-manifest.json`。官方 `minimind-3` 只用于环境/推理基线，不属于从随机初始化训练得到的个人 checkpoint。

## 阶段 1：官方权重推理验收

在仓库根目录执行：

```powershell
conda activate minimind
python eval_llm.py --load_from ./minimind-3 --max_new_tokens 256
```

选择自动测试模式 `0`。

通过条件：

- 无 DLL、CUDA 或权重加载错误；
- CUDA 保持可用；
- 至少完成固定提示集生成；
- 保存运行命令、日志和 tokens/s。

产物：

- `configs/<run-id>-official-inference.json`
- `logs/<run-id>-official-inference.log`
- `metrics/<run-id>-official-inference.json`

## 阶段 2：确定性数据拆分与冒烟集

新增一个数据准备脚本，以固定 seed 和稳定哈希规则生成：

- Pretrain smoke：256～1000 条；
- SFT smoke：128～500 条；
- 独立 validation/test；
- 每个输出文件的 SHA-256 和样本数清单。

禁止从 validation/test 回流训练。生成数据保存在 `dataset/`，只提交生成脚本与清单，不提交大体积 JSONL。

通过条件：同一输入与 seed 重跑时，输出文件哈希完全一致。

## 阶段 3：Pretrain 冒烟训练

所有训练脚本从 `trainer/` 目录执行。RTX 4060 首轮保守配置：

```powershell
cd trainer
python train_pretrain.py `
  --epochs 1 `
  --batch_size 2 `
  --accumulation_steps 16 `
  --max_seq_len 256 `
  --num_workers 0 `
  --log_interval 10 `
  --save_interval 50 `
  --data_path ../dataset/smoke/pretrain_smoke.jsonl `
  --save_dir ../out/smoke `
  --save_weight pretrain_smoke
```

通过条件：

- 完成至少 100～500 个 micro steps；
- 无 OOM、NaN 或 Inf；
- loss 总体下降；
- checkpoint 可保存、加载和断点恢复；
- 记录峰值显存、wall-clock、tokens/s 与最终配置。

未通过时只改变一个参数，优先顺序：`batch_size` → `max_seq_len` → `accumulation_steps`。不启用 MoE、WandB 或 `torch.compile`。

## 阶段 4：SFT 冒烟训练

从 Pretrain smoke checkpoint 继续：

```powershell
python train_full_sft.py `
  --epochs 1 `
  --batch_size 1 `
  --accumulation_steps 16 `
  --max_seq_len 512 `
  --num_workers 0 `
  --log_interval 10 `
  --save_interval 50 `
  --data_path ../dataset/smoke/sft_smoke.jsonl `
  --save_dir ../out/smoke `
  --from_weight pretrain_smoke `
  --save_weight full_sft_smoke
```

通过条件与 Pretrain 相同，并额外要求：SFT 正确加载 Pretrain 权重，生成结果体现 chat template 和基础指令跟随能力。

## 阶段 5：评估完整训练成本

不要套用官方 RTX 3090 的 2 小时口径。根据冒烟实测计算完整数据的预计 step 数和 wall-clock，并记录笔记本温度、功耗和持续负载。

决策：

- 时间和散热可接受：本地继续；
- 单阶段预计数天或持续热负载不可接受：使用云端 3090；
- 云端必须复用相同 commit、数据哈希、依赖快照和配置。

## 阶段 6：完成最小从零复现

主链路只有：

```text
随机初始化 Dense 64M
  → Pretrain mini 1 epoch
  → 从 Pretrain checkpoint 开始 Full SFT mini 1 epoch
```

至少保存：随机基线、Pretrain checkpoint、Full SFT checkpoint、train/validation loss、PPL、耗时、峰值显存、tokens/s、恢复记录和固定样例。

严禁把官方 `minimind-3` 权重放入这条个人从零训练链路。

## 阶段 7：独立评测

实现固定 held-out 评测器，并比较：

1. 随机初始化模型；
2. Pretrain checkpoint；
3. Full SFT checkpoint。

最低指标：validation loss、PPL、固定提示集结果、失败分类。结果必须同时提供逐样本 JSONL 和汇总 JSON，不能只保留聊天截图。

## 阶段 8：受控实验——全参 SFT vs LoRA

两组实验必须使用相同的 Pretrain 起点、SFT 子集、seed、有效 batch、步数、序列长度和评测集，只改变微调方式。

报告：可训练参数量与比例、优化器/显存开销、wall-clock、loss/PPL、任务指标和失败样例。

## 阶段 9：接入无人机 v4

将无人机 v4 的 400 条训练任务转换为 MiniMind chat JSONL，不修改原有 50 条独立测试标签。比较：

1. MiniMind-3 Zero；
2. MiniMind-3 无人机领域适配；
3. Qwen2.5-0.5B LoRA Planner。

复用指标：JSON 合法率、契约通过率、语义准确率、安全拒绝率、合法误拒率、字段级错误；Planner 与 PyBullet 控制器结果分开归因。

MiniMind 不要求超过 Qwen。实验目标是解释 64M 模型在严格结构化规划中的能力边界。

## 暂不执行

- MoE；
- DPO、PPO、GRPO、CISPO；
- Tool Calling 与 Agentic RL；
- 知识蒸馏。

只有阶段 1～9 的证据完整后，再按明确研究问题选择进阶扩展。

## 简历准入门槛

只有完成以下内容后才写入正式简历：

- 可审计的 Pretrain → Full SFT 主链路；
- 独立验证集和机器可读指标；
- 至少一个单变量受控实验；
- 无人机 v4 无泄漏领域迁移；
- 明确区分上游复现内容与个人实现/实验。
