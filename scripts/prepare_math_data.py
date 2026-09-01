"""Download pinned official files and audit DAPO/AIME without loading a GPU."""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path

from math_data import QuestionIndex

DAPO_REVISION = '65877096c24ffa7abc4e4fa5edb95cf3413a5674'
DOMAINS = ('https://huggingface.co', 'https://hf-mirror.com')


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def get_metadata(repo, revision=None):
    import requests
    errors = []
    for domain in DOMAINS:
        url = f'{domain}/api/datasets/{repo}' + (f'/revision/{revision}' if revision else '')
        try:
            response = requests.get(url, params={'blobs': 'true'}, timeout=(10, 20))
            response.raise_for_status()
            value = response.json()
            if revision and value['sha'] != revision:
                raise ValueError('revision mismatch')
            return value, url
        except Exception as exc:
            errors.append(f'{url}: {type(exc).__name__}: {exc}')
    raise RuntimeError('\n'.join(errors))


def download(repo, revision, filename, target, expected_sha):
    import requests
    if not expected_sha:
        raise ValueError('Require source LFS SHA256 before downloading')
    if target.exists():
        if sha256(target) == expected_sha:
            return {'cached': True, 'sha256': expected_sha}
        raise ValueError(f'Existing file hash mismatch: {target}')
    errors = []
    for domain in DOMAINS:
        url = f'{domain}/datasets/{repo}/resolve/{revision}/{filename}'
        try:
            print(json.dumps({'event': 'download', 'url': url}), flush=True)
            with requests.get(url, stream=True, timeout=(10, 45)) as response:
                response.raise_for_status()
                with open(str(target) + '.partial', 'wb') as f:
                    for block in response.iter_content(1024 * 1024):
                        f.write(block)
            observed = sha256(str(target) + '.partial')
            if observed != expected_sha:
                raise ValueError('Downloaded content does not match source LFS SHA256')
            os.replace(str(target) + '.partial', target)
            return {'url': url, 'sha256': observed, 'verified_lfs_hash': True}
        except Exception as exc:
            errors.append(f'{url}: {type(exc).__name__}: {exc}')
    raise RuntimeError('\n'.join(errors))


def dump_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def write_rows(path, rows):
    with path.open('w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')


def main():
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer
    import numpy as np
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--tokenizer', default='model')
    args = parser.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'manifest.json').exists():
        raise RuntimeError('Completed data directory exists; do not overwrite')
    started = time.time()
    collected, manifest = {}, {'created_unix': started, 'sources': {}}
    for name, repo, revision in [('dapo', 'BytedTsinghua-SIA/DAPO-Math-17k', DAPO_REVISION),
                                 ('aime', 'BytedTsinghua-SIA/AIME-2024', None)]:
        metadata, metadata_url = get_metadata(repo, revision)
        revision = metadata['sha']
        parquet_files = [s for s in metadata['siblings'] if s['rfilename'].endswith('.parquet')]
        index = QuestionIndex()
        files = []
        for number, entry in enumerate(parquet_files):
            path = out / f'{name}-{number}.parquet'
            file_info = download(repo, revision, entry['rfilename'], path,
                                 (entry.get('lfs') or {}).get('sha256'))
            files.append(dict(file_info, source_file=entry['rfilename']))
            for batch in pq.ParquetFile(path).iter_batches(batch_size=2048):
                for row in batch.to_pylist():
                    index.add(row)
        rows, conflicts, stats = index.finish()
        collected[name] = rows
        write_rows(out / f'{name}-conflicts.jsonl', conflicts)
        manifest['sources'][name] = {'repo': repo, 'revision': revision,
                                     'metadata_url': metadata_url, 'files': files, 'audit': stats}
        print(json.dumps({'event': 'audited', 'dataset': name, **stats}), flush=True)
    evaluation_keys = {x['problem_key'] for x in collected['aime']}
    overlaps = [x for x in collected['dapo'] if x['problem_key'] in evaluation_keys]
    train = [x for x in collected['dapo'] if x['problem_key'] not in evaluation_keys]
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    for rows in (train, collected['aime']):
        for row in rows:
            text = tokenizer.apply_chat_template(row['prompt'], tokenize=False,
                                                 add_generation_prompt=True, open_thinking=False)
            row['prompt_tokens'] = len(tokenizer.encode(text, add_special_tokens=False))
    lengths = [x['prompt_tokens'] for x in train]
    manifest.update({'train_questions': len(train), 'aime_questions': len(collected['aime']),
                     'exact_normalized_problem_overlaps_removed': len(overlaps),
                     'near_duplicate_and_pretraining_contamination': 'not_excluded',
                     'prompt_tokens': {str(p): float(np.percentile(lengths, p)) for p in [0, 25, 50, 75, 90, 95, 99, 100]},
                     'prompt_over_1024': sum(n > 1024 for n in lengths),
                     'cpu_prep_seconds': time.time() - started})
    write_rows(out / 'train.jsonl', train)
    write_rows(out / 'aime.jsonl', collected['aime'])
    write_rows(out / 'evaluation-overlaps.jsonl', overlaps)
    manifest['prepared_sha256'] = {n: sha256(out / n) for n in ('train.jsonl', 'aime.jsonl')}
    dump_json(out / 'manifest.json', manifest)
    print(json.dumps(manifest, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
