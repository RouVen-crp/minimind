"""CPU-only summary; separates coverage throughput from effective learning."""
import argparse
import json
from pathlib import Path
import statistics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--report', required=True)
    args = parser.parse_args()
    run = Path(args.run_dir)
    summary = json.loads((run / 'summary.json').read_text(encoding='utf-8'))
    manifest = json.loads((Path(args.data_dir) / 'manifest.json').read_text(encoding='utf-8'))
    supervisor = json.loads((run / 'supervisor.json').read_text(encoding='utf-8'))
    groups = summary.get('group_metrics', [])
    generated = sum(len(g['correct']) for g in groups)
    correct = sum(sum(g['correct']) for g in groups)
    parsed = sum(sum(g['parsed']) for g in groups)
    truncated = sum(sum(g['truncated']) for g in groups)
    times = [g['generation_seconds'] + g['update_path_seconds'] for g in groups]
    evaluation = [g for g in summary.get('aime_before', []) if 'correct' in g]
    n = manifest['train_questions']
    estimated = {}
    if times:
        # Empirical range, not a confidence interval. Fixed sampling/model/length assumptions.
        quartiles = statistics.quantiles(times, n=4, method='inclusive') if len(times) > 1 else times * 3
        estimated = {'coverage_only_lower_hours': n * quartiles[0] / 3600,
                     'coverage_only_mean_hours': n * statistics.mean(times) / 3600,
                     'coverage_only_upper_hours': n * quartiles[2] / 3600,
                     'interpretation': 'empirical p25-p75 time-per-question extrapolation, not confidence bounds',
                     'assumptions': 'one rollout group per unique question; same generation cap; no dynamic resampling',
                     'exclusions': 'additional SFT, teacher data, complete optimizer cost if no steps observed, length drift, long prompts not sampled, full eval/checkpoint overhead'}
    metrics = {'unique_training_questions': n, 'aime_questions': manifest['aime_questions'],
               'pilot_groups': len(groups), 'rollout_responses': generated,
               'correct_responses': correct, 'parsed_responses': parsed, 'truncated_responses': truncated,
               'correctness_signal_groups': sum(g['correctness_signal'] for g in groups),
               'optimizer_steps': summary.get('optimizer_steps', 0),
               'nonzero_parameter_updates': summary.get('nonzero_parameter_updates', 0),
               'aime_before_evaluated': len(evaluation),
               'aime_before_correct': sum(sum(g['correct']) for g in evaluation),
               'gpu_wall_seconds': supervisor['gpu_wall_used_seconds'],
               'peak_allocated_mib_observed_in_groups': max([g['peak_allocated_mib'] for g in groups], default=0),
               'checkpoint_reload_exact': summary.get('checkpoint_tensors_exact', False),
               'status': summary['status'], 'estimate': estimated}
    (run / 'analysis.json').write_text(json.dumps(metrics, indent=2) + '\n', encoding='utf-8')
    lines = ['# MiniMind 数学首轮实测', '',
             '本轮仅小规模可行性验证；未执行全量训练、数学 SFT 或教师调用。', '',
             '## 实测结果', '',
             '| 指标 | 结果 |', '| --- | --- |',
             f'| 状态 | {summary["status"]} |',
             f'| 实际模型参数 | {summary.get("parameters", "unknown"):,} |' if isinstance(summary.get('parameters'), int) else '| 实际参数 | 未加载 |',
             f'| DAPO 去重、冲突及重叠处理后 | {n} 题 |',
             f'| AIME 训前评测 | {metrics["aime_before_correct"]}/{len(evaluation)}；总题数 {manifest["aime_questions"]} |',
             f'| RL 采样 | {len(groups)} 组，{generated} 个回答 |',
             f'| 可解析 / 正确 / 达长度上限未EOS | {parsed} / {correct} / {truncated} |',
             f'| 有正确性差异的组 | {metrics["correctness_signal_groups"]} |',
             f'| 优化器步数 / 实际参数变化次数 | {metrics["optimizer_steps"]} / {metrics["nonzero_parameter_updates"]} |',
             f'| 首轮GPU进程墙钟（含初始化和测量开销） | {metrics["gpu_wall_seconds"]:.2f} 秒 |',
             f'| RL组内观测峰值 allocated 显存 | {metrics["peak_allocated_mib_observed_in_groups"]:.1f} MiB |',
             f'| 权重保存、严格重载逐tensor一致 | {metrics["checkpoint_reload_exact"]} |', '',
             '## 时间估算与限制', '']
    if estimated:
        lines.append(f'相同采样配置遍历 {n} 题：按组耗时均值粗估 **{estimated["coverage_only_mean_hours"]:.2f} 小时**；'
                     f'按观测组耗时四分位外推约 {estimated["coverage_only_lower_hours"]:.2f}–{estimated["coverage_only_upper_hours"]:.2f} 小时。')
        lines.extend(['', '这是覆盖工作量估算，不是有效学习时间承诺，也不是统计置信区间。样本小、生成长度会漂移；未包括后续 SFT、教师、额外重采样及全程评测/检查点开销。',
                      f'样本配置：每题 {summary["config"]["num_generations"]} 个回答，最多 {summary["config"]["max_new_tokens"]} 新 token，prompt 上限 {summary["config"]["max_prompt_tokens"]}。',
                      f'超出该 prompt 上限未覆盖的训练题：{summary.get("ineligible_long_prompts", "unknown")}。'])
    else:
        lines.append('没有完整RL组耗时，不能给出可靠全量估算。')
    if not metrics['optimizer_steps']:
        lines.extend(['', '**没有有效正确性训练更新。本轮全流程验收未达成。**零方差组测量了前向/反向路径，但没有调用优化器以制造虚假变化；全量估算未实测优化器更新成本。',
                      '权重重载验证通过不等于学习发生；未重复跑无变化权重的训后AIME，也未将训前结果复制成训后成绩。',
                      '下一步应先讨论数学 SFT 的解答来源与预算，再决定是否继续；不建议直接将零信号运行扩到全量。'])
    lines.extend(['', '## 审计入口', '', f'- 数据 manifest：`{args.data_dir}/manifest.json`',
                  f'- 运行目录：`{run}`（summary、events、responses、gpu、supervisor、analysis）',
                  '- 奖励规则：只接受唯一末行 `Answer: 整数`；正负号与可选美元括号规范化，正确+1/错误-1。与官方宽松解析器有明确差异；格式率与正确率分开记录。',
                  '- 训练/评测只检查规范化题面精确重叠，不能保证近重复或历史预训练无污染。',
                  '- 8项CPU单元测试验证解析、防猜答案列表、去重冲突、EOS遮罩、零优势、梯度方向及裁剪。', ''])
    Path(args.report).write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps(metrics, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
