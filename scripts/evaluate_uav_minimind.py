"""Run MiniMind through the existing UAV contract-v4 evaluator."""

from __future__ import annotations

import argparse
import importlib
import json
import re
import sys
from pathlib import Path

from scripts.uav_minimind_adapter import MiniMindPlanner


def _load_external(project_root: Path):
    root = str(project_root.resolve())
    sys.path.insert(0, root)
    try:
        evaluation = importlib.import_module("drone_planner.evaluation")
        experiments = importlib.import_module("drone_planner.experiments")
        return evaluation.evaluate_records_batched, experiments._summary, experiments._write_breakdowns
    finally:
        sys.path.remove(root)


def _read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--uav-project-root", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--allow-blind-test", action="store_true")
    parser.add_argument("--checkpoint", type=Path, default=Path("out/full_sft_768.pth"))
    parser.add_argument("--lora-checkpoint", type=Path)
    parser.add_argument("--variant", help="Stable experiment label used in evaluator rows and output filenames.")
    parser.add_argument("--tokenizer-path", type=Path, default=Path("model"))
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/metrics/uav-v4-minimind"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.split == "test" and not args.allow_blind_test:
        raise SystemExit("Blind test is locked. Use --allow-blind-test only for the approved stage 3B run.")
    dataset_dir = args.dataset_dir or args.uav_project_root / "artifacts" / "dataset_contract_v4"
    records = _read(dataset_dir / f"{args.split}.jsonl")
    planner = MiniMindPlanner(
        project_root=args.uav_project_root,
        checkpoint=args.checkpoint,
        lora_checkpoint=args.lora_checkpoint,
        tokenizer_path=args.tokenizer_path,
        device=args.device,
    )
    evaluate_records, summarise, write_breakdowns = _load_external(args.uav_project_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    variant = args.variant or (
        args.lora_checkpoint.stem
        if args.lora_checkpoint is not None
        else "minimind_zero" if args.checkpoint.stem == "full_sft_768" else args.checkpoint.stem
    )
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", variant):
        raise SystemExit("--variant may contain only letters, digits, dot, underscore, and hyphen")
    rows_path = args.output_dir / f"{variant}_{args.split}.jsonl"
    evaluate_records(records, planner, variant, rows_path, batch_size=args.batch_size)
    rows = _read(rows_path)
    summary = {"variant": variant, "split": args.split, **summarise(rows)}
    (args.output_dir / f"{variant}_{args.split}_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_breakdowns(rows, args.output_dir, f"{variant}_{args.split}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
