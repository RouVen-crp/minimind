"""Small human-solution + answer-format warm-up, with separate DAPO probes."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import re
from transformers import AutoTokenizer
from math_data import canonical_integer, content_id, problem_text

GSM_COMMIT = '3101c7d5072418e28b9008a6636bde82a006892c'
GSM_URL = f'https://raw.githubusercontent.com/openai/grade-school-math/{GSM_COMMIT}/grade_school_math/data/train.jsonl'


def rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line]


def write_rows(path, data):
    path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in data), encoding='utf-8')


def encode_supervision(tokenizer, messages, answer_text, max_length=768):
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, open_thinking=False)
    prefix = tokenizer.encode(prompt, add_special_tokens=False)
    suffix = tokenizer.encode(answer_text, add_special_tokens=False) + [tokenizer.eos_token_id]
    if len(prefix) + len(suffix) > max_length:
        return None  # Never truncate away the ground-truth final answer.
    return {'input_ids': prefix + suffix, 'labels': [-100] * len(prefix) + suffix,
            'prompt_tokens': len(prefix), 'supervised_tokens': len(suffix)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gsm-train', required=True)
    parser.add_argument('--dapo-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'manifest.json').exists():
        raise RuntimeError('Prepared directory already exists')
    tokenizer = AutoTokenizer.from_pretrained('model', local_files_only=True)
    dapo = rows(Path(args.dapo_dir) / 'train.jsonl')
    aime = rows(Path(args.dapo_dir) / 'aime.jsonl')
    eval_keys = {r['problem_key'] for r in aime}
    rng = random.Random(20260831)
    gsm = []
    for raw in rows(args.gsm_train):
        reasoning, sep, final = raw['answer'].rpartition('####')
        answer = canonical_integer(final.strip().replace(',', ''))
        if not sep or answer is None:
            continue
        reasoning = re.sub(r'<<.*?>>', '', reasoning).strip()
        messages = [{'role': 'user', 'content': 'Solve the following math problem step by step. '
                     'End with a single line Answer: followed by the integer answer.\n\n' + raw['question']}]
        if content_id(problem_text(messages)) in eval_keys:
            continue
        target = reasoning + '\nAnswer: ' + answer
        encoded = encode_supervision(tokenizer, messages, target)
        if encoded:
            gsm.append(dict(encoded, id='gsm-' + content_id(raw['question']), source='gsm8k_human_solution',
                            prompt=messages, target=target, answer=answer))
    rng.shuffle(gsm)
    gsm_train, gsm_validation = gsm[:1024], gsm[1024:1088]
    candidates = []
    for row in dapo:
        encoded = encode_supervision(tokenizer, row['prompt'], 'Answer: ' + row['answer'])
        if encoded:
            candidates.append(dict(encoded, id=row['id'], source='dapo_answer_format_only',
                                   prompt=row['prompt'], target='Answer: ' + row['answer'], answer=row['answer']))
    rng.shuffle(candidates)
    format_train = candidates[:512]
    warmup_ids = {r['id'] for r in format_train}
    warmup_problem_keys = {r['problem_key'] for r in dapo if r['id'] in warmup_ids}
    probe_pool = [r for r in dapo if r['id'] not in warmup_ids and r['problem_key'] not in warmup_problem_keys]
    # A fixed random diagnostic pool, not answer-based selection of easy/known-correct rows.
    rng.shuffle(probe_pool)
    probe_pool = probe_pool[:256]
    mixed = gsm_train + format_train
    rng.shuffle(mixed)
    write_rows(out / 'sft_train.jsonl', mixed)
    write_rows(out / 'sft_validation.jsonl', gsm_validation)
    write_rows(out / 'dapo_probe.jsonl', probe_pool)
    manifest = {'gsm_url': GSM_URL, 'gsm_commit': GSM_COMMIT,
                'gsm_file_sha256': hashlib.sha256(Path(args.gsm_train).read_bytes()).hexdigest(),
                'gsm_human_solution_train': len(gsm_train), 'gsm_validation': len(gsm_validation),
                'dapo_answer_format_train': len(format_train), 'dapo_probe_questions': len(probe_pool),
                'probe_overlap_with_dapo_warmup': len({r['id'] for r in probe_pool} & warmup_ids),
                'seed': 20260831, 'max_sft_tokens': 768,
                'total_sft_examples': len(mixed), 'uses_external_teacher_api': False,
                'is_self_distillation': False,
                'length_exclusion': 'Only complete prompt+target+EOS <=768 tokens; no target truncation',
                'near_duplicate_contamination': 'not_comprehensively_excluded',
                'prepared_sha256': {name: hashlib.sha256((out / name).read_bytes()).hexdigest()
                                    for name in ['sft_train.jsonl', 'sft_validation.jsonl', 'dapo_probe.jsonl']}}
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(manifest), flush=True)


if __name__ == '__main__':
    main()
