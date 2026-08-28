# 阶段 3A：无人机 v4 数据转换与 MiniMind 评测适配

## 结论

阶段 3A 已完成数据与评测基础设施，尚未产生 MiniMind Zero 模型成绩。真实模型 validation 与 blind test 必须等待阶段 2 的个人 `full_sft_768.pth`。

## 数据协议

- 源数据：UAV contract v4，固定 seed 42。
- train：400 条，其中拒绝任务 16 条。
- validation：50 条，其中拒绝任务 5 条。
- blind test：50 条；阶段 3A 只自动读取 schema、数量、SHA-256、ID 与归一化指令哈希，不生成派生文件。
- 正常任务的 assistant target 是 `plan`；拒绝任务必须使用 `expected_response`，不能误用记录中残留的 `plan`。
- 训练和验证数据转换为 MiniMind `conversations`，固定为 system/user/assistant 三轮，并补齐 `reasoning_content`、`tools`、`tool_calls` 字符串字段。

源目录未包含可发现的 Git 元数据，因此不能记录可靠 commit；改用源 manifest、三份数据和五个评测模块的 SHA-256 作为权威溯源。完整清单见 `experiments/manifests/uav-v4-minimind.json`。

## 转换结果

| Split | 行数 | 拒绝样本 | 字节 | SHA-256 |
| --- | ---: | ---: | ---: | --- |
| train | 400 | 16 | 828070 | `fb192d7f48fdb19d5b19db13c5362a7b7432b6b8343a10e8e65174ec96406011` |
| validation | 50 | 5 | 100293 | `15a85f1755c6f68a048c7e92fbffad34dc345138c4b89172c51fd49b4b45419c` |

train/validation/test 两两之间：ID 交集为 0，NFKC + casefold + 空白归一化后的 instruction SHA-256 交集为 0。完整派生 JSONL 位于 `dataset/uav_v4_minimind/`，由 Git 忽略，可用脚本确定性重建。

## MiniMind 数据装载验收

使用项目真实 tokenizer 和 `SFTDataset(max_length=768)` 装载 train：

- 行数：400；
- tensor shape：`[768]`；
- 抽检首样本 supervised assistant tokens：43；
- system/user/padding labels 保持 `-100`，只有 assistant answer 参与 Causal LM loss。

## 评测适配

`MiniMindPlanner` 实现毕业设计评测器需要的 `generate` / `generate_many` 接口：

1. 直接复用 `drone_planner.prompts.build_messages`，避免复制 prompt 合同；
2. 加载个人 MiniMind 原生 PyTorch checkpoint；
3. 使用 greedy decoding（`do_sample=False`）；
4. 只返回 completion，不把 prompt 混入评测输出；
5. 输出直接进入原有 JSON、约束、语义、安全拒绝、合法误拒和字段级错误评测。

评测 CLI 默认只允许 validation。读取 blind test 必须同时设置 `--split test --allow-blind-test`，该命令只允许在阶段 3B 最终盲测时运行。

## Smoke 结果

50 条 validation 标签通过 oracle planner 回放到既有评测器，验证适配链路：JSON object rate、约束通过率、语义准确率、安全拒绝正确率、端到端契约通过率均为 1.0，合法误拒率为 0.0。

这些值只证明“数据 → planner 接口 → 评测行 → 汇总指标”的接线正确，`is_model_metric=false`，不得写入简历或作为模型性能。

## 复现命令

```powershell
python scripts/prepare_uav_minimind_data.py `
  --uav-project-root D:/CODE/graduation_project/gradproject-main

python -m unittest `
  tests.test_prepare_uav_minimind_data `
  tests.test_uav_minimind_adapter `
  tests.test_prepare_experiment_data

python scripts/smoke_uav_evaluation.py `
  --uav-project-root D:/CODE/graduation_project/gradproject-main
```

## 阶段 3B 阻塞项

- 阶段 2完成并生成个人 `out/full_sft_768.pth`；
- 先运行 validation，固定所有推理参数；
- 参数冻结后一次性运行 50 条 blind test；
- 保存逐样本结果、汇总、失败分解与 checkpoint/config 哈希。
