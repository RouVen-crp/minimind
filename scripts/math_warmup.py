"""Bounded, supervised math preparation. Human solutions + labelled answer-only rows."""
import argparse
import json
import math
import os
from pathlib import Path
import random
import signal
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from scripts.math_pilot import read_rows, write_json


def collate(rows):
    length = max(len(r['input_ids']) for r in rows)
    ids = torch.tensor([r['input_ids'] + [0] * (length - len(r['input_ids'])) for r in rows])
    labels = torch.tensor([r['labels'] + [-100] * (length - len(r['labels'])) for r in rows])
    return ids, labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--checkpoint', default='out/full_sft_768.pth')
    parser.add_argument('--steps', type=int, default=576)
    parser.add_argument('--learning-rate', type=float, default=5e-5)
    args = parser.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    deadline = float(os.environ['MATH_PILOT_DEADLINE'])
    stop = [False]
    signal.signal(signal.SIGTERM, lambda *unused: stop.__setitem__(0, True))
    torch.set_num_threads(4)
    torch.manual_seed(20260831)
    torch.cuda.manual_seed_all(20260831)
    rng = random.Random(20260831)
    started = time.time()
    config = MiniMindConfig(hidden_size=768, num_hidden_layers=8, use_moe=False)
    model = MiniMindForCausalLM(config)
    baseline = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    model.load_state_dict(baseline, strict=True)
    model.cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.0)
    train = read_rows(Path(args.data_dir) / 'sft_train.jsonl')
    validation = read_rows(Path(args.data_dir) / 'sft_validation.jsonl')
    summary = {'status': 'running', 'config': vars(args), 'started_unix': started,
               'train_examples': len(train), 'steps': 0, 'losses': [], 'learning_rate': args.learning_rate}
    def emit(event, **values):
        print(json.dumps({'event': event, 'unix': time.time(), **values}), flush=True)
        write_json(out / 'summary.json', summary)
    def evaluate():
        total, count = 0.0, 0
        model.eval()
        with torch.no_grad():
            for offset in range(0, len(validation), 8):
                ids, labels = collate(validation[offset:offset + 8])
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    loss = model(ids.cuda(), labels=labels.cuda()).loss
                n = labels[:, 1:].ne(-100).sum().item()
                total += loss.item() * n
                count += n
        return {'loss': total / max(count, 1), 'supervised_tokens': count}
    summary['validation_before'] = evaluate()
    emit('warmup_initialized', validation=summary['validation_before'])
    order = list(range(len(train)))
    for step in range(args.steps):
        if stop[0] or time.time() >= deadline - 45:
            summary['stop_reason'] = 'stage_time_limit'
            break
        if step % math.ceil(len(order) / 8) == 0:
            rng.shuffle(order)
        start = (step * 8) % len(order)
        batch = [train[i] for i in order[start:start + 8]]
        ids, labels = collate(batch)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        lr = args.learning_rate * min(1., (step + 1) / 20) * (0.1 + 0.9 * (1 + math.cos(math.pi * step / args.steps)) / 2)
        optimizer.param_groups[0]['lr'] = lr
        with torch.autocast('cuda', dtype=torch.bfloat16):
            result = model(ids.cuda(), labels=labels.cuda())
            loss = result.loss
        if not torch.isfinite(loss):
            raise FloatingPointError('nonfinite SFT loss')
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        summary['steps'] = step + 1
        if step % 32 == 0 or step + 1 == args.steps:
            metrics = {'step': step + 1, 'loss': loss.item(), 'grad_norm': grad_norm.item(), 'lr': lr}
            summary['losses'].append(metrics)
            emit('warmup_step', **metrics)
        del result, loss, ids, labels
    optimizer.zero_grad(set_to_none=True)
    summary['validation_after'] = evaluate()
    state = {n: p.detach().cpu() for n, p in model.state_dict().items()}
    summary['max_parameter_delta_from_baseline'] = max((state[n].float() - baseline[n].float()).abs().max().item() for n in state)
    torch.save(state, out / 'warmup.pth')
    loaded = torch.load(out / 'warmup.pth', map_location='cpu', weights_only=True)
    summary['checkpoint_exact'] = all(torch.equal(state[n], loaded[n]) for n in state)
    model.load_state_dict(loaded, strict=True)
    summary.update({'status': 'completed', 'wall_seconds': time.time() - started,
                    'peak_allocated_mib': torch.cuda.max_memory_allocated() / 1024**2,
                    'checkpoint': str((out / 'warmup.pth').resolve())})
    emit('warmup_finished', **summary)


if __name__ == '__main__':
    main()
