"""Reviewable full-run launcher. Dry-run first; execution requires explicit approval."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def validate_config(config):
    if config.get('model') != 'MiniMind-63.9M' or config.get('epochs') != 1:
        raise ValueError('Only the reviewed one-epoch MiniMind configuration is supported')
    for path, expected in config['required_sha256'].items():
        if file_sha256(path) != expected:
            raise ValueError('File fingerprint changed: ' + path)
    rows = [json.loads(x) for x in Path(config['data_dir'], 'train.jsonl').read_text(encoding='utf-8').splitlines()]
    if len(rows) != config['train_questions'] or len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Question count/uniqueness mismatch')
    if max(r['prompt_tokens'] for r in rows) > config['max_prompt_tokens']:
        raise ValueError('Full configuration would truncate or omit a question')
    if config['num_generations'] < 2 or config['wall_time_limit_seconds'] < 300:
        raise ValueError('Invalid GRPO group size or runtime allowance')
    if Path(config['output_dir']).exists():
        raise ValueError('Output directory exists; never overwrite or silently resume')
    return len(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--execute', action='store_true')
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--resume-checkpoint')
    args = p.parse_args()
    config = json.loads(Path(args.config).read_text(encoding='utf-8'))
    validate_config(config)
    command = [sys.executable, '-u', 'scripts/math_grpo_readiness.py', '--data-dir', config['data_dir'],
               '--output-dir', config['output_dir'], '--checkpoint', config['checkpoint'], '--full', '--execute-full']
    for key in ['num_generations', 'max_prompt_tokens', 'max_new_tokens', 'learning_rate', 'beta', 'seed', 'save_interval', 'zero_signal_stop']:
        command += ['--' + key.replace('_', '-'), str(config[key])]
    if args.resume_checkpoint:
        command += ['--resume-checkpoint', args.resume_checkpoint]
    if args.dry_run or not args.execute:
        print(json.dumps({'status': 'dry_run_passed', 'execution_authorized': config['execution_authorized'],
                          'train_questions': config['train_questions'], 'command': command,
                          'wall_time_limit_seconds_proposed': config['wall_time_limit_seconds']}, indent=2))
        return
    if config.get('execution_authorized') is not True:
        p.error('User approval is still pending in the reviewed config; full training was not started')
    query = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True)
    if query.strip():
        raise RuntimeError('GPU has running compute processes; do not overlap')
    out = Path(config['output_dir'])
    out.mkdir(parents=True, exist_ok=False)
    (out / 'approved-config.json').write_text(json.dumps(config, indent=2) + '\n', encoding='utf-8')
    started = time.time()
    deadline = started + config['wall_time_limit_seconds']
    child = None
    result = {'started_unix': started, 'deadline_unix': deadline, 'command': command, 'full_training': True}
    try:
        with (out / 'stdout.log').open('x', encoding='utf-8') as log:
            child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                                     env=dict(os.environ, MATH_PILOT_DEADLINE=str(deadline - 5), TOKENIZERS_PARALLELISM='false'))
            result['pid'] = child.pid
            (out / 'supervisor.json').write_text(json.dumps(result, indent=2))
            while child.poll() is None:
                if time.time() >= deadline - 10:
                    os.killpg(child.pid, signal.SIGTERM)
                    try:
                        child.wait(timeout=max(0.1, deadline - time.time() - 1))
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL)
                        child.wait(timeout=2)
                    result['watchdog_stop'] = True
                    break
                time.sleep(1)
            result['exit_code'] = child.returncode
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait(timeout=2)
        result.update({'finished_unix': time.time(), 'wall_seconds': time.time() - started})
        (out / 'supervisor.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
