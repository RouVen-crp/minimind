"""Deterministically convert UAV contract-v4 data to MiniMind chat JSONL."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import sys
import unicodedata
from pathlib import Path
from typing import Callable


SOURCE_SPLITS = ("train", "validation", "test")
OUTPUT_SPLITS = ("train", "validation")
EVALUATION_MODULES = ("prompts.py", "contracts.py", "validation.py", "evaluation.py", "experiments.py")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(path: Path, split: str) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{split} line {line_number}: invalid JSON") from error
            required = {"id", "category", "instruction", "plan"}
            if not isinstance(record, dict) or not required.issubset(record):
                raise ValueError(f"{split} line {line_number}: missing required UAV fields")
            if not all(isinstance(record[key], str) for key in ("id", "category", "instruction")):
                raise ValueError(f"{split} line {line_number}: id/category/instruction must be strings")
            if not isinstance(record["plan"], dict):
                raise ValueError(f"{split} line {line_number}: plan must be an object")
            if record.get("expect_rejection") and not isinstance(record.get("expected_response"), dict):
                raise ValueError(f"{split} line {line_number}: rejection target must be expected_response")
            rows.append(record)
    return rows


def _instruction_identity(instruction: str) -> str:
    normalized = " ".join(unicodedata.normalize("NFKC", instruction).casefold().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _load_training_messages(project_root: Path) -> Callable[[dict], list[dict]]:
    root = str(project_root.resolve())
    sys.path.insert(0, root)
    try:
        module = importlib.import_module("drone_planner.prompts")
        return module.training_messages
    finally:
        sys.path.remove(root)


def _normalise_message(message: dict) -> dict[str, str]:
    if not isinstance(message, dict) or message.get("role") not in {"system", "user", "assistant"}:
        raise ValueError("training message has an invalid role")
    if not isinstance(message.get("content"), str):
        raise ValueError("training message content must be a string")
    return {
        "role": message["role"],
        "content": message["content"],
        "reasoning_content": str(message.get("reasoning_content") or ""),
        "tools": str(message.get("tools") or ""),
        "tool_calls": str(message.get("tool_calls") or ""),
    }


def _write_conversations(
    path: Path,
    records: list[dict],
    training_messages: Callable[[dict], list[dict]],
) -> dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    rejection_rows = 0
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            messages = [_normalise_message(message) for message in training_messages(record)]
            if [message["role"] for message in messages] != ["system", "user", "assistant"]:
                raise ValueError(f"{record['id']}: expected system/user/assistant conversation")
            expected_target = record.get("expected_response") or record["plan"]
            actual_target = json.loads(messages[-1]["content"])
            if actual_target != expected_target:
                raise ValueError(f"{record['id']}: assistant target does not match source label")
            rejection_rows += int(bool(record.get("expect_rejection")))
            payload = json.dumps(
                {"conversations": messages}, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            )
            handle.write(payload + "\n")
    os.replace(temporary, path)
    return {
        "path": path.as_posix(),
        "rows": len(records),
        "rejection_rows": rejection_rows,
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def build_conversion(
    *,
    project_root: Path,
    dataset_dir: Path,
    output_dir: Path,
    training_messages: Callable[[dict], list[dict]] | None = None,
) -> dict[str, object]:
    source_manifest_path = dataset_dir / "manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    records: dict[str, list[dict]] = {}
    source_files: dict[str, dict[str, object]] = {}
    ids: dict[str, set[str]] = {}
    instruction_hashes: dict[str, set[str]] = {}

    for split in SOURCE_SPLITS:
        path = dataset_dir / f"{split}.jsonl"
        actual_sha = _sha256(path)
        expected_sha = source_manifest["sha256"][split].lower()
        if actual_sha != expected_sha:
            raise ValueError(f"{split}: SHA-256 differs from source manifest")
        rows = _read_jsonl(path, split)
        if len(rows) != source_manifest["splits"][split]:
            raise ValueError(f"{split}: row count differs from source manifest")
        split_ids = {row["id"] for row in rows}
        if len(split_ids) != len(rows):
            raise ValueError(f"{split}: duplicate IDs")
        records[split] = rows
        ids[split] = split_ids
        instruction_hashes[split] = {_instruction_identity(row["instruction"]) for row in rows}
        source_files[split] = {
            "path": f"artifacts/dataset_contract_v4/{split}.jsonl",
            "rows": len(rows),
            "bytes": path.stat().st_size,
            "sha256": actual_sha,
        }

    intersections: dict[str, dict[str, int]] = {}
    for left_index, left in enumerate(SOURCE_SPLITS):
        for right in SOURCE_SPLITS[left_index + 1 :]:
            label = f"{left}__{right}"
            id_count = len(ids[left] & ids[right])
            instruction_count = len(instruction_hashes[left] & instruction_hashes[right])
            intersections[label] = {"id": id_count, "normalised_instruction_sha256": instruction_count}
            if id_count or instruction_count:
                raise ValueError(f"{label}: cross-split leakage detected")

    builder = training_messages or _load_training_messages(project_root)
    outputs = {
        split: _write_conversations(output_dir / f"uav_v4_{split}.jsonl", records[split], builder)
        for split in OUTPUT_SPLITS
    }
    module_files = {}
    for name in EVALUATION_MODULES:
        path = project_root / "drone_planner" / name
        module_files[f"drone_planner/{name}"] = {"bytes": path.stat().st_size, "sha256": _sha256(path)}

    return {
        "schema_version": 1,
        "conversion": "uav-contract-v4-to-minimind-chat-v1",
        "source": {
            "project_root_argument": str(project_root),
            "dataset_version": source_manifest["dataset_version"],
            "manifest_sha256": _sha256(source_manifest_path),
            "git_revision": None,
            "provenance_note": "Source directory has no discoverable Git metadata; file SHA-256 values are authoritative.",
            "files": source_files,
            "evaluation_modules": module_files,
        },
        "policy": {
            "train": "converted to MiniMind SFT conversations",
            "validation": "converted for development evaluation",
            "test": "blind: schema/hash/leakage checks only; no derived JSONL emitted in stage 3A",
            "target": "expected_response for rejection rows, otherwise plan",
            "ordering": "preserve source order",
        },
        "cross_split_intersections": intersections,
        "outputs": outputs,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--uav-project-root", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("dataset/uav_v4_minimind"))
    parser.add_argument("--manifest", type=Path, default=Path("experiments/manifests/uav-v4-minimind.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_dir = args.dataset_dir or args.uav_project_root / "artifacts" / "dataset_contract_v4"
    manifest = build_conversion(
        project_root=args.uav_project_root,
        dataset_dir=dataset_dir,
        output_dir=args.output_dir,
    )
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.manifest.with_suffix(args.manifest.suffix + ".tmp")
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, args.manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

