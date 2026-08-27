# MiniMind-3 复现与无人机领域适配行动路线

## 当前状态

- [x] 创建独立 Conda 环境 `minimind`。
- [x] 安装并验证 `torch==2.6.0+cu124`。
- [x] CUDA 可用，GPU 为 RTX 4060 Laptop 8 GB。
- [x] 下载 Pretrain mini、SFT mini 与官方 `minimind-3` 模型。
- [x] 固定上游代码快照并建立个人 Fork 分支。
- [x] 保存环境、硬件、数据和官方模型证据。
- [x] 完成官方权重推理验收。
- [x] 生成并复验确定性 smoke/validation/test 数据拆分及 SHA-256 清单。
- [x] 完成全量 Pretrain 前 500 step 稳定性验证：无 OOM/NaN，loss 从 7.5222 降至 6.2078。

基线证据见 `manifests/baseline-manifest.json`。官方 `minimind-3` 只用于环境/推理基线，不属于从随机初始化训练得到的个人 checkpoint。

## 阶段 1：本地完成 Pretrain mini 1 epoch

当前全量训练已经通过前 500 step 稳定性验证，实测约 7.3 小时/epoch，本阶段继续在 RTX 4060 本地完成。

最终训练合同：Dense 64M、`pretrain_t2t_mini`、1 epoch、seed 42。若当前进程使用默认 `epochs=2`，在 checkpoint 保存后以 `--epochs 1 --from_resume 1` 续训，确保最终只完成 1 epoch。

验收证据：`pretrain_768.pth`、完整命令和配置、训练日志、loss 曲线、wall-clock、峰值显存、tokens/s、断点恢复记录与固定 Pretrain 样例。官方权重不得进入该训练链路。

## 阶段 2：通用 Full SFT mini 1 epoch

从个人 `pretrain_768.pth` 开始通用全参数 SFT。先运行 500 step 测量真实速度、显存和散热，再决定执行位置：预计不超过 12～16 小时则继续本地；明显超过该范围或持续热负载不可接受时，才切换远程 3090。

训练命令必须显式设置 `--epochs 1 --from_weight pretrain`。保存 `full_sft_768.pth`、训练/验证 loss、PPL、资源指标和固定指令样例。该 checkpoint 是后续所有无人机领域实验的唯一共同起点。

## 阶段 3：并行接入无人机 v4 数据与评测

该阶段不占用 GPU，可以在 Pretrain/SFT 运行期间进行：

- 将无人机 v4 的 400 条训练任务确定性转换为 MiniMind chat JSONL；
- 保持原有 50 条独立测试标签不可变，禁止进入训练；
- 记录源文件、转换文件、模板和代码 SHA-256；
- 复用 JSON 合法性、契约、语义、安全拒绝、合法误拒和字段级评测器；
- 为 MiniMind 输出增加适配层，不修改评测口径。

通过条件：转换可重复、train/test 无交集、MiniMind Zero 输出可进入同一评测器。

## 阶段 4：无人机领域全参 SFT vs LoRA 受控实验

不再单独做一轮通用数据上的全参 SFT vs LoRA；受控实验直接落在无人机任务上。全部变体从同一个个人 `full_sft_768.pth` 出发：

1. MiniMind Zero：不做无人机领域训练；
2. MiniMind UAV Full SFT：400 条训练任务全参数微调；
3. MiniMind UAV LoRA：相同训练任务进行 LoRA；
4. 已有 Qwen2.5-0.5B LoRA Planner：作为现有系统基线。

MiniMind Full SFT 与 LoRA 必须固定相同数据、seed、有效 batch、训练步数、序列长度和测试集，只改变微调方式。400 条领域数据在 RTX 4060 本地完成，不使用远程服务器。

报告可训练参数量及比例、峰值显存、wall-clock、loss/PPL、JSON 合法率、契约通过率、语义准确率、安全拒绝率、合法误拒率和字段级错误。MiniMind 不要求超过 Qwen，目标是解释 64M 模型的能力边界与微调代价。

## 阶段 5：报告、简历与面试证据

形成两层证据：

- 底层复现：随机初始化 → Pretrain → Full SFT 的 loss/PPL、checkpoint 与资源曲线；
- 领域微调：Zero / UAV Full SFT / UAV LoRA / Qwen LoRA 的逐样本 JSONL、汇总 JSON 和失败分类。

面试必须能结合源码和实验说明：Pretrain 与 SFT label/mask 的差异、causal attention、GQA/RoPE/RMSNorm/SwiGLU、LoRA 的低秩更新与 rank/alpha、可训练参数和优化器显存、为什么 loss 下降不等于 JSON/语义指标提升，以及 Planner 与 PyBullet 失败如何分层归因。

## 暂不执行

- MoE；
- DPO、PPO、GRPO、CISPO；
- Tool Calling 与 Agentic RL；
- 知识蒸馏。

只有阶段 1～5 的证据完整后，再按明确研究问题选择进阶扩展。

## 简历准入门槛

只有完成以下内容后才写入正式简历：

- 可审计的 Pretrain → Full SFT 主链路；
- 独立验证集和机器可读指标；
- 至少一个单变量受控实验；
- 无人机 v4 无泄漏领域迁移；
- 明确区分上游复现内容与个人实现/实验。
