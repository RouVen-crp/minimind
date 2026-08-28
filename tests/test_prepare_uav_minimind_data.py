import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scripts.prepare_uav_minimind_data import build_conversion


def _write_jsonl(path: Path, rows: list[dict]) -> str:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _record(identifier: str, instruction: str, rejection: bool = False) -> dict:
    record = {
        "id": identifier,
        "category": "unsafe" if rejection else "point",
        "instruction": instruction,
        "plan": {"mission_id": identifier},
    }
    if rejection:
        record.update(
            expect_rejection=True,
            expected_response={"status": "needs_clarification", "reason": "unsafe"},
        )
    return record


def _messages(record: dict) -> list[dict]:
    target = record.get("expected_response") or record["plan"]
    return [
        {"role": "system", "content": "contract"},
        {"role": "user", "content": record["instruction"]},
        {"role": "assistant", "content": json.dumps(target, ensure_ascii=False, separators=(",", ":"))},
    ]


class PrepareUavMiniMindDataTests(unittest.TestCase):
    def _source(self, root: Path, collision: bool = False) -> Path:
        dataset = root / "artifacts" / "dataset_contract_v4"
        package = root / "drone_planner"
        dataset.mkdir(parents=True)
        package.mkdir()
        for name in ("prompts.py", "contracts.py", "validation.py", "evaluation.py", "experiments.py"):
            (package / name).write_text("# fixture\n", encoding="utf-8")
        rows = {
            "train": [_record("train-1", "训练任务"), _record("train-2", "危险训练", True)],
            "validation": [_record("validation-1", "训练任务" if collision else "验证任务")],
            "test": [_record("test-1", "盲测任务", True)],
        }
        hashes = {split: _write_jsonl(dataset / f"{split}.jsonl", split_rows) for split, split_rows in rows.items()}
        (dataset / "manifest.json").write_text(
            json.dumps(
                {
                    "dataset_version": "v4",
                    "splits": {split: len(split_rows) for split, split_rows in rows.items()},
                    "sha256": hashes,
                }
            ),
            encoding="utf-8",
        )
        return dataset

    def test_conversion_is_deterministic_and_keeps_test_blind(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = self._source(root)
            first = build_conversion(
                project_root=root, dataset_dir=dataset, output_dir=root / "first", training_messages=_messages
            )
            second = build_conversion(
                project_root=root, dataset_dir=dataset, output_dir=root / "second", training_messages=_messages
            )
            for split in ("train", "validation"):
                self.assertEqual(first["outputs"][split]["sha256"], second["outputs"][split]["sha256"])
            self.assertFalse((root / "first" / "uav_v4_test.jsonl").exists())
            train_rows = [json.loads(line) for line in (root / "first" / "uav_v4_train.jsonl").read_text(encoding="utf-8").splitlines()]
            rejection_target = json.loads(train_rows[1]["conversations"][-1]["content"])
            self.assertEqual(rejection_target["status"], "needs_clarification")
            self.assertEqual(set(train_rows[0]["conversations"][0]), {"role", "content", "reasoning_content", "tools", "tool_calls"})
            self.assertTrue(all(not counts["id"] and not counts["normalised_instruction_sha256"] for counts in first["cross_split_intersections"].values()))

    def test_normalised_instruction_leakage_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = self._source(root, collision=True)
            with self.assertRaisesRegex(ValueError, "cross-split leakage"):
                build_conversion(
                    project_root=root, dataset_dir=dataset, output_dir=root / "output", training_messages=_messages
                )


if __name__ == "__main__":
    unittest.main()

