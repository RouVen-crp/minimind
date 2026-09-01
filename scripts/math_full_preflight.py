"""Synthetic maximum-shape memory check, NEVER an optimization or quality result."""
import argparse
import json
from pathlib import Path
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from transformers import AutoTokenizer
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output-dir', required=True)
    p.add_argument('--checkpoint', required=True)
    args = p.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(99)
    rows = [json.loads(x) for x in Path('dataset/math-dapo-20260831/train.jsonl').read_text().splitlines()]
    longest = max(rows, key=lambda r: r['prompt_tokens'])
    tokenizer = AutoTokenizer.from_pretrained('model', local_files_only=True)
    text = tokenizer.apply_chat_template(longest['prompt'], tokenize=False, add_generation_prompt=True, open_thinking=False)
    prefix = tokenizer.encode(text, add_special_tokens=False)
    model = MiniMindForCausalLM(MiniMindConfig(hidden_size=768, num_hidden_layers=8, use_moe=False))
    model.load_state_dict(torch.load(args.checkpoint, map_location='cpu', weights_only=True), strict=True)
    model.cuda().train()
    reference = MiniMindForCausalLM(model.config).cuda().eval().requires_grad_(False)
    reference.load_state_dict(model.state_dict())
    ids = torch.tensor([prefix + [5] * 256] * 8, device='cuda')
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        old = model(ids, logits_to_keep=257).logits[:, :-1].float().log_softmax(-1).gather(-1, ids[:, -256:, None]).squeeze(-1)
        ref = reference(ids, logits_to_keep=257).logits[:, :-1].float().log_softmax(-1).gather(-1, ids[:, -256:, None]).squeeze(-1)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        logits = model(ids, logits_to_keep=257).logits[:, :-1].float()
        logps = logits.log_softmax(-1).gather(-1, ids[:, -256:, None]).squeeze(-1)
        loss = -logps.mean() + (ref - logps).square().mean() * 0.04
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
    torch.cuda.synchronize()
    result = {'status': 'passed', 'synthetic_only': True, 'optimizer_steps': 0,
              'question_id_for_shape': longest['id'], 'prompt_tokens': len(prefix),
              'completion_tokens': 256, 'batch_size': 8,
              'forward_backward_seconds': time.perf_counter() - started,
              'finite_gradient_norm': norm.item(),
              'peak_allocated_mib': torch.cuda.max_memory_allocated() / 1024**2,
              'peak_reserved_mib': torch.cuda.max_memory_reserved() / 1024**2,
              'notes': 'Fixed dummy completion tests allocation only; no optimizer, no checkpoint, not a mathematical reward/update result.'}
    (Path(args.output_dir) / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
