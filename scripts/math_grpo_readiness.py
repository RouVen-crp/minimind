"""Explicitly bounded GRPO readiness entry; full execution requires a separate opt-in."""
import argparse
import json
from pathlib import Path
import time
from math_pilot import Pilot, read_rows


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', required=True)
    p.add_argument('--probe-file')
    p.add_argument('--output-dir', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--tokenizer', default='model')
    p.add_argument('--max-groups', type=int, default=128)
    p.add_argument('--num-generations', type=int, default=8)
    p.add_argument('--max-prompt-tokens', type=int, default=3072)
    p.add_argument('--max-new-tokens', type=int, default=256)
    p.add_argument('--zero-signal-stop', type=int, default=64)
    p.add_argument('--learning-rate', type=float, default=3e-7)
    p.add_argument('--beta', type=float, default=0.04)
    p.add_argument('--seed', type=int, default=43)
    p.add_argument('--stop-after-updates', type=int, default=4)
    p.add_argument('--save-interval', type=int, default=16)
    p.add_argument('--resume-checkpoint')
    p.add_argument('--full', action='store_true')
    p.add_argument('--execute-full', action='store_true')
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args(argv)
    args.skip_length_benchmark = True
    if args.full:
        if not args.execute_full and not args.dry_run:
            p.error('Full training is not implicit; requires separate --execute-full authorization')
        if args.probe_file:
            p.error('Full training cannot use the small diagnostic file')
        args.max_groups = len(read_rows(Path(args.data_dir) / 'train.jsonl'))
        args.stop_after_updates = 0
    elif args.max_groups > 256:
        p.error('Readiness mode must stay <=256 groups')
    if args.num_generations < 2:
        p.error('GRPO requires >=2 generations')
    return args


def main():
    args = parse_args()
    if args.dry_run:
        print(json.dumps(vars(args), indent=2))
        return
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'summary.json').exists():
        raise RuntimeError('Run directory contains previous results; choose a new directory')
    pilot = Pilot(args)
    try:
        pilot.run()
    except BaseException as exc:
        pilot.summary.update({'status': 'budget_stop' if isinstance(exc, TimeoutError) else 'failed',
                              'error': f'{type(exc).__name__}: {exc}', 'finished_unix': time.time(),
                              'full_workflow_complete': False})
        pilot.emit('stopped', error=str(exc))
        raise


if __name__ == '__main__':
    main()
