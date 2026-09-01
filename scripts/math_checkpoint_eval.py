"""Compare frozen math checkpoints on the same held-out GSM8K rows."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import sys
import time

import torch
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from scripts.math_data import verify_answer


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def collate(rows):
    length = max(len(r['input_ids']) for r in rows)
    ids = torch.tensor([r['input_ids'] + [0] * (length - len(r['input_ids'])) for r in rows])
    labels = torch.tensor([r['labels'] + [-100] * (length - len(r['labels'])) for r in rows])
    return ids, labels


def decode_completions(model, tokenizer, row, count, sample, max_new_tokens):
    prompt = tokenizer.apply_chat_template(row['prompt'], tokenize=False,
                                           add_generation_prompt=True, open_thinking=False)
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    inputs = torch.tensor([prompt_ids] * count, device='cuda')
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        output = model.generate(input_ids=inputs, max_new_tokens=max_new_tokens,
                                do_sample=sample, temperature=1.0, top_k=0, top_p=1.0,
                                eos_token_id=tokenizer.eos_token_id, use_cache=True,
                                logits_to_keep=1).clone()
    completion = output[:, len(prompt_ids):]
    eos = completion.eq(tokenizer.eos_token_id)
    mask = (eos.cumsum(1) - eos.long()).eq(0)
    texts = [tokenizer.decode(tokens[keep].tolist(), skip_special_tokens=True)
             for tokens, keep in zip(completion, mask)]
    return [{'response': text, **verify_answer(text, row['answer'])} for text in texts]


def evaluate(label, checkpoint, rows, tokenizer, max_new_tokens, samples):
    torch.manual_seed(20260901)
    torch.cuda.manual_seed_all(20260901)
    random.seed(20260901)
    model = MiniMindForCausalLM(MiniMindConfig(hidden_size=768, num_hidden_layers=8, use_moe=False))
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    model.load_state_dict(state, strict=True)
    del state
    model.cuda().eval()
    total_loss = 0.0
    supervised_tokens = 0
    with torch.no_grad():
        for offset in range(0, len(rows), 8):
            ids, labels = collate(rows[offset:offset + 8])
            with torch.autocast('cuda', dtype=torch.bfloat16):
                loss = model(ids.cuda(), labels=labels.cuda()).loss
            count = labels[:, 1:].ne(-100).sum().item()
            total_loss += loss.item() * count
            supervised_tokens += count
    details = []
    for row in rows:
        greedy = decode_completions(model, tokenizer, row, 1, False, max_new_tokens)[0]
        sampled = decode_completions(model, tokenizer, row, samples, True, max_new_tokens)
        details.append({'id': row['id'], 'answer': row['answer'], 'greedy': greedy,
                        'sampled': sampled})
    result = {
        'label': label,
        'checkpoint': checkpoint,
        'checkpoint_sha256': sha256(checkpoint),
        'validation_loss': total_loss / supervised_tokens,
        'supervised_tokens': supervised_tokens,
        'questions': len(rows),
        'greedy_correct': sum(x['greedy']['correct'] for x in details),
        'greedy_parsed': sum(x['greedy']['parsed'] for x in details),
        'sampled_responses': len(rows) * samples,
        'sampled_correct': sum(sum(y['correct'] for y in x['sampled']) for x in details),
        'sampled_parsed': sum(sum(y['parsed'] for y in x['sampled']) for x in details),
        'pass_at_samples': sum(any(y['correct'] for y in x['sampled']) for x in details),
        'details': details,
    }
    del model
    torch.cuda.empty_cache()
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True)
    p.add_argument('--checkpoint', action='append', required=True,
                   help='label=checkpoint_path; repeat for each checkpoint')
    p.add_argument('--tokenizer', default='model')
    p.add_argument('--output', required=True)
    p.add_argument('--max-new-tokens', type=int, default=256)
    p.add_argument('--samples', type=int, default=8)
    args = p.parse_args()
    rows = [json.loads(x) for x in Path(args.data).read_text(encoding='utf-8').splitlines() if x]
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    started = time.time()
    results = []
    for item in args.checkpoint:
        label, checkpoint = item.split('=', 1)
        results.append(evaluate(label, checkpoint, rows, tokenizer,
                                args.max_new_tokens, args.samples))
    summary = {'status': 'completed', 'started_unix': started,
               'finished_unix': time.time(), 'wall_seconds': time.time() - started,
               'data': args.data, 'data_sha256': sha256(args.data),
               'samples_per_question': args.samples, 'results': results}
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps({**summary, 'results': [{k: v for k, v in r.items() if k != 'details'}
                                            for r in results]}, indent=2), flush=True)


if __name__ == '__main__':
    main()
