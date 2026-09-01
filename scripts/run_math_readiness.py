"""Serial stage supervisor with a durable cumulative GPU ledger, never full training."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def save(path, value):
    temporary = Path(str(path) + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['warmup', 'grpo', 'memory'], required=True)
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--seconds', type=int, required=True)
    parser.add_argument('--checkpoint', default='out/full_sft_768.pth')
    parser.add_argument('--steps', type=int, default=576)
    parser.add_argument('--resume-checkpoint')
    parser.add_argument('--stop-after-updates', type=int, default=4)
    args = parser.parse_args()
    if args.stage == 'grpo' and args.seconds < 240:
        parser.error('GRPO needs >=240s stage allowance for its 180s finalization reserve; actual use remains metered')
    ledger = Path('experiments/runtime/math-readiness-state.json')
    lock = open(str(ledger) + '.lock', 'w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state = json.loads(ledger.read_text())
    if state.get('phase') == 'running':
        raise RuntimeError('Previous attempt has not been reconciled; do not reset budget')
    remaining = state['gpu_wall_budget_seconds'] - state['gpu_wall_used_seconds']
    if remaining < 60 or args.seconds > remaining or args.seconds < 30:
        raise RuntimeError('Requested stage exceeds remaining cumulative GPU budget')
    out = Path(args.run_dir)
    out.mkdir(parents=True, exist_ok=False)
    if args.stage == 'warmup':
        command = [sys.executable, '-u', 'scripts/math_warmup.py', '--data-dir', 'dataset/math-warmup-20260831',
                   '--output-dir', str(out), '--checkpoint', args.checkpoint, '--steps', str(args.steps)]
    elif args.stage == 'grpo':
        command = [sys.executable, '-u', 'scripts/math_grpo_readiness.py', '--data-dir', 'dataset/math-dapo-20260831',
                   '--probe-file', 'dataset/math-warmup-20260831/dapo_probe.jsonl',
                   '--output-dir', str(out), '--checkpoint', args.checkpoint,
                   '--stop-after-updates', str(args.stop_after_updates)]
        if args.resume_checkpoint:
            command += ['--resume-checkpoint', args.resume_checkpoint]
    else:
        command = [sys.executable, '-u', 'scripts/math_full_preflight.py', '--output-dir', str(out),
                   '--checkpoint', args.checkpoint]
    started = time.time()
    deadline = started + args.seconds
    attempt = {'stage': args.stage, 'run_dir': str(out.resolve()), 'started_unix': started,
               'deadline_unix': deadline, 'command': command, 'supervisor_pid': os.getpid()}
    state['attempts'].append(attempt)
    state.update({'phase': 'running', 'current_stage': args.stage, 'active_run': str(out.resolve()),
                  'hard_deadline_unix': deadline})
    save(ledger, state)
    child = None
    try:
        with (out / 'stdout.log').open('x', encoding='utf-8') as log:
            child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                                     env=dict(os.environ, MATH_PILOT_DEADLINE=str(deadline - 5), TOKENIZERS_PARALLELISM='false'))
            attempt['pid'] = child.pid
            save(ledger, state)
            while child.poll() is None:
                if time.time() >= deadline - 10:
                    os.killpg(child.pid, signal.SIGTERM)
                    try:
                        child.wait(timeout=max(0.1, deadline - time.time() - 1))
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL)
                        child.wait(timeout=2)
                    attempt['watchdog_stop'] = True
                    break
                time.sleep(1)
        attempt['exit_code'] = child.returncode
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait(timeout=2)
        elapsed = time.time() - started
        attempt.update({'finished_unix': time.time(), 'gpu_wall_seconds': elapsed})
        state['gpu_wall_used_seconds'] += elapsed
        state['phase'] = 'stage_finished'
        save(ledger, state)
        save(out / 'supervisor.json', attempt)
    print(json.dumps(state), flush=True)


if __name__ == '__main__':
    main()
