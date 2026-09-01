"""Persistent one-shot budget ledger and independent process-group watchdog."""
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
    temp = Path(str(path) + '.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
    os.replace(temp, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--data-dir', required=True)
    args = parser.parse_args()
    run = Path(args.run_dir)
    run.mkdir(parents=True, exist_ok=True)
    state_path = Path('experiments/runtime/math-pilot-state.json')
    lock = open(str(state_path) + '.lock', 'w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state = json.loads(state_path.read_text())
    if state['phase'] != 'ready' or state.get('attempts'):
        raise RuntimeError('Pilot is not ready or was already attempted; no automatic relaunch')
    start = time.time()
    remaining = max(0, 1800 - state.get('gpu_wall_used_seconds', 0))
    if remaining <= 0:
        raise RuntimeError('GPU budget exhausted')
    deadline = start + remaining
    env = dict(os.environ, MATH_PILOT_DEADLINE=str(deadline - 5), TOKENIZERS_PARALLELISM='false')
    command = [sys.executable, '-u', 'scripts/math_pilot.py', '--data-dir', args.data_dir,
               '--output-dir', str(run)]
    state.update({'phase': 'launching', 'run_dir': str(run.resolve()), 'started_unix': start,
                  'hard_deadline_unix': deadline, 'attempts': 1, 'command': command,
                  'supervisor_pid': os.getpid()})
    save(state_path, state)
    child = None
    def request_stop(*unused):
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    try:
        with (run / 'stdout.log').open('x', encoding='utf-8') as logfile:
            child = subprocess.Popen(command, stdout=logfile, stderr=subprocess.STDOUT,
                                     env=env, start_new_session=True)
            state.update({'phase': 'running', 'pid': child.pid})
            save(state_path, state)
            with (run / 'gpu.jsonl').open('x', encoding='utf-8') as gpu_log:
                while child.poll() is None:
                    if time.time() >= deadline - 10:
                        os.killpg(child.pid, signal.SIGTERM)
                        try:
                            child.wait(timeout=max(0.1, deadline - time.time() - 1))
                        except subprocess.TimeoutExpired:
                            os.killpg(child.pid, signal.SIGKILL)
                            child.wait(timeout=2)
                        state['watchdog_stop'] = True
                        break
                    try:
                        query = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,utilization.gpu,temperature.gpu',
                                                '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=2)
                        gpu_log.write(json.dumps({'unix': time.time(), 'csv': query.stdout.strip()}) + '\n')
                        gpu_log.flush()
                    except subprocess.SubprocessError:
                        pass
                    time.sleep(min(2, max(0.1, deadline - time.time() - 10)))
        state['exit_code'] = child.returncode
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait(timeout=2)
        state.update({'phase': 'finished', 'finished_unix': time.time(),
                      'gpu_wall_used_seconds': state.get('gpu_wall_used_seconds', 0) + time.time() - start})
        save(state_path, state)
        save(run / 'supervisor.json', state)
    print(json.dumps(state), flush=True)


if __name__ == '__main__':
    main()
