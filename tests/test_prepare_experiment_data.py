import argparse
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scripts.prepare_experiment_data import build_manifest


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


class PrepareExperimentDataTests(unittest.TestCase):
    def _args(self, root: Path, output_name: str) -> argparse.Namespace:
        return argparse.Namespace(
            pretrain_source=str(root / "pretrain.jsonl"),
            sft_source=str(root / "sft.jsonl"),
            output_dir=str(root / output_name),
            manifest=str(root / f"{output_name}.json"),
            seed=42,
            train_bps=6_000,
            validation_bps=2_000,
            pretrain_train=10,
            pretrain_validation=5,
            pretrain_test=5,
            sft_train=10,
            sft_validation=5,
            sft_test=5,
        )

    def test_outputs_are_deterministic_disjoint_and_hashed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_jsonl(root / "pretrain.jsonl", [{"text": f"pretrain-{i}"} for i in range(300)])
            _write_jsonl(
                root / "sft.jsonl",
                [
                    {"conversations": [{"role": "user", "content": f"question-{i}"}, {"role": "assistant", "content": f"answer-{i}"}]}
                    for i in range(300)
                ],
            )

            first = build_manifest(self._args(root, "first"))
            second = build_manifest(self._args(root, "second"))

            for dataset_name in ("pretrain", "sft"):
                first_files = first["datasets"][dataset_name]["files"]
                second_files = second["datasets"][dataset_name]["files"]
                seen = set()
                for split in ("train", "validation", "test"):
                    self.assertEqual(first_files[split]["sha256"], second_files[split]["sha256"])
                    first_path = Path(first_files[split]["path"])
                    second_path = Path(second_files[split]["path"])
                    self.assertEqual(first_path.read_bytes(), second_path.read_bytes())
                    rows = set(first_path.read_text(encoding="utf-8").splitlines())
                    self.assertTrue(seen.isdisjoint(rows))
                    seen.update(rows)
                    self.assertEqual(
                        hashlib.sha256(first_path.read_bytes()).hexdigest().upper(),
                        first_files[split]["sha256"],
                    )

    def test_invalid_schema_fails_with_line_number(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_jsonl(root / "pretrain.jsonl", [{"wrong": "field"}])
            _write_jsonl(
                root / "sft.jsonl",
                [{"conversations": [{"role": "user", "content": "ok"}]} for _ in range(100)],
            )
            args = self._args(root, "invalid")
            args.pretrain_train = 1
            args.pretrain_validation = 0
            args.pretrain_test = 0
            with self.assertRaisesRegex(ValueError, "pretrain line 1"):
                build_manifest(args)


if __name__ == "__main__":
    unittest.main()
