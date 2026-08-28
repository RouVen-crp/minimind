"""Exercise the UAV evaluator wiring on validation labels without loading a model."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path


class ValidationOraclePlanner:
    """Return validation labels solely to verify adapter/evaluator plumbing."""

    def __init__(self, records: list[dict]) -> None:
        self.responses = {
            record["instruction"]: json.dumps(
                record.get("expected_response") or record["plan"],
                ensure_ascii=False,
                separators=(",", ":"),
            )
            for record in records
        }

    def generate_many(self, instructions: list[str], _examples=None) -> list[str]:
        return [self.responses[instruction] for instruction in instructions]


def _read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--uav-project-root", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/metrics/uav-v4-smoke"))
    args = parser.parse_args()
    dataset_dir = args.dataset_dir or args.uav_project_root / "artifacts" / "dataset_contract_v4"
    records = _read(dataset_dir / "validation.jsonl")

    root = str(args.uav_project_root.resolve())
    sys.path.insert(0, root)
    try:
        evaluation = importlib.import_module("drone_planner.evaluation")
        experiments = importlib.import_module("drone_planner.experiments")
    finally:
        sys.path.remove(root)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = args.output_dir / "validation_oracle_wiring.jsonl"
    evaluation.evaluate_records_batched(
        records,
        ValidationOraclePlanner(records),
        "validation_oracle_wiring",
        rows_path,
        batch_size=8,
    )
    rows = _read(rows_path)
    summary = {
        "variant": "validation_oracle_wiring",
        "split": "validation",
        "is_model_metric": False,
        "purpose": "adapter/evaluator wiring smoke only",
        **experiments._summary(rows),
    }
    summary_path = args.output_dir / "validation_oracle_wiring_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

