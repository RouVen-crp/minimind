"""Create deterministic, disjoint MiniMind smoke/validation/test subsets."""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable


SPLIT_POLICY_VERSION = "sha256-content-v1"


def _stable_digest(domain: bytes, seed: int, identity: bytes) -> bytes:
    return hashlib.sha256(domain + b"\0" + str(seed).encode("ascii") + b"\0" + identity).digest()


def _partition(identity: bytes, seed: int, train_bps: int, validation_bps: int) -> str:
    bucket = int.from_bytes(_stable_digest(b"split", seed, identity)[:8], "big") % 10_000
    if bucket < train_bps:
        return "train"
    if bucket < train_bps + validation_bps:
        return "validation"
    return "test"


@dataclass
class BoundedSelector:
    limit: int
    heap: list[tuple[int, bytes, bytes]] = field(default_factory=list)
    identities: set[bytes] = field(default_factory=set)

    def add(self, rank: int, identity: bytes, payload: bytes) -> None:
        if self.limit == 0 or identity in self.identities:
            return
        entry = (-rank, identity, payload)
        if len(self.heap) < self.limit:
            heapq.heappush(self.heap, entry)
            self.identities.add(identity)
            return
        if rank >= -self.heap[0][0]:
            return
        removed = heapq.heapreplace(self.heap, entry)
        self.identities.remove(removed[1])
        self.identities.add(identity)

    def ordered(self) -> list[bytes]:
        return [payload for _, _, payload in sorted(self.heap, key=lambda item: (-item[0], item[1]))]


def _validate_pretrain(record: object, line_number: int) -> None:
    if not isinstance(record, dict) or not isinstance(record.get("text"), str):
        raise ValueError(f"pretrain line {line_number}: expected an object with string field 'text'")


def _validate_sft(record: object, line_number: int) -> None:
    if not isinstance(record, dict) or not isinstance(record.get("conversations"), list):
        raise ValueError(f"sft line {line_number}: expected an object with list field 'conversations'")
    if not record["conversations"]:
        raise ValueError(f"sft line {line_number}: conversations must not be empty")
    for message_index, message in enumerate(record["conversations"]):
        if not isinstance(message, dict):
            raise ValueError(f"sft line {line_number}, message {message_index}: expected an object")
        if not isinstance(message.get("role"), str) or not isinstance(message.get("content"), str):
            raise ValueError(
                f"sft line {line_number}, message {message_index}: role and content must be strings"
            )


def _display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _write_jsonl(path: Path, rows: list[bytes]) -> dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    digest = hashlib.sha256()
    with temporary.open("wb") as output:
        for row in rows:
            encoded = row + b"\n"
            output.write(encoded)
            digest.update(encoded)
    os.replace(temporary, path)
    return {
        "path": _display_path(path),
        "samples": len(rows),
        "bytes": path.stat().st_size,
        "sha256": digest.hexdigest().upper(),
    }


def prepare_dataset(
    *,
    name: str,
    source: Path,
    output_dir: Path,
    seed: int,
    counts: dict[str, int],
    validator: Callable[[object, int], None],
    train_bps: int = 9_800,
    validation_bps: int = 100,
) -> dict[str, object]:
    if train_bps < 0 or validation_bps < 0 or train_bps + validation_bps > 10_000:
        raise ValueError("partition basis points must define a valid train/validation/test split")
    selectors = {split: BoundedSelector(counts[split]) for split in ("train", "validation", "test")}
    source_digest = hashlib.sha256()
    source_rows = 0
    empty_rows = 0

    with source.open("rb") as input_file:
        for line_number, raw_with_newline in enumerate(input_file, start=1):
            source_digest.update(raw_with_newline)
            raw = raw_with_newline.rstrip(b"\r\n")
            if not raw:
                empty_rows += 1
                continue
            try:
                record = json.loads(raw)
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise ValueError(f"{name} line {line_number}: invalid UTF-8 JSON") from error
            validator(record, line_number)
            source_rows += 1
            identity = hashlib.sha256(raw).digest()
            split = _partition(identity, seed, train_bps, validation_bps)
            rank = int.from_bytes(_stable_digest(b"rank", seed, identity), "big")
            selectors[split].add(rank, identity, raw)

    for split, selector in selectors.items():
        if len(selector.heap) != counts[split]:
            raise ValueError(
                f"{name}: requested {counts[split]} unique {split} rows, found {len(selector.heap)}"
            )

    files = {}
    for split, selector in selectors.items():
        suffix = "train_smoke" if split == "train" else split
        files[split] = _write_jsonl(output_dir / f"{name}_{suffix}.jsonl", selector.ordered())

    return {
        "source": {
            "path": _display_path(source),
            "bytes": source.stat().st_size,
            "rows": source_rows,
            "empty_rows": empty_rows,
            "sha256": source_digest.hexdigest().upper(),
        },
        "files": files,
    }


def build_manifest(args: argparse.Namespace) -> dict[str, object]:
    output_dir = Path(args.output_dir)
    common = {
        "output_dir": output_dir,
        "seed": args.seed,
        "train_bps": args.train_bps,
        "validation_bps": args.validation_bps,
    }
    pretrain = prepare_dataset(
        name="pretrain",
        source=Path(args.pretrain_source),
        counts={"train": args.pretrain_train, "validation": args.pretrain_validation, "test": args.pretrain_test},
        validator=_validate_pretrain,
        **common,
    )
    sft = prepare_dataset(
        name="sft",
        source=Path(args.sft_source),
        counts={"train": args.sft_train, "validation": args.sft_validation, "test": args.sft_test},
        validator=_validate_sft,
        **common,
    )
    return {
        "schema_version": 1,
        "policy": {
            "version": SPLIT_POLICY_VERSION,
            "seed": args.seed,
            "assignment": f"SHA-256 domain-separated content hash; {args.train_bps}/{args.validation_bps}/{10000 - args.train_bps - args.validation_bps} basis points",
            "selection": "lowest domain-separated SHA-256 ranks per partition",
            "ordering": "ascending selection rank then content hash",
            "duplicate_identity": "SHA-256 of the source JSONL row without CR/LF",
        },
        "datasets": {"pretrain": pretrain, "sft": sft},
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pretrain-source", default="dataset/pretrain_t2t_mini.jsonl")
    parser.add_argument("--sft-source", default="dataset/sft_t2t_mini.jsonl")
    parser.add_argument("--output-dir", default="dataset/experiment_splits")
    parser.add_argument("--manifest", default="experiments/manifests/data-splits.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-bps", type=int, default=9_800)
    parser.add_argument("--validation-bps", type=int, default=100)
    parser.add_argument("--pretrain-train", type=int, default=512)
    parser.add_argument("--pretrain-validation", type=int, default=128)
    parser.add_argument("--pretrain-test", type=int, default=128)
    parser.add_argument("--sft-train", type=int, default=256)
    parser.add_argument("--sft-validation", type=int, default=64)
    parser.add_argument("--sft-test", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build_manifest(args)
    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, manifest_path)
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
