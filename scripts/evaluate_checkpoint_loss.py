"""Compute token-weighted validation loss and perplexity for MiniMind checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataset.lm_dataset import PretrainDataset, SFTDataset
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("pretrain", "sft"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--tokenizer-path", type=Path, default=Path("model"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-seq-len", type=int, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--hidden-size", type=int, default=768)
    parser.add_argument("--num-hidden-layers", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    dataset_type = PretrainDataset if args.kind == "pretrain" else SFTDataset
    dataset = dataset_type(args.data_path, tokenizer, max_length=args.max_seq_len)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    config = MiniMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers, use_moe=False)
    model = MiniMindForCausalLM(config)
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"), strict=True)
    model = model.eval().to(args.device)
    device_type = "cuda" if args.device.startswith("cuda") else "cpu"
    if device_type == "cuda":
        model = model.half()
        torch.cuda.reset_peak_memory_stats()

    total_nll = 0.0
    supervised_tokens = 0
    started = time.perf_counter()
    with torch.inference_mode():
        for input_ids, labels in loader:
            input_ids = input_ids.to(args.device, non_blocking=True)
            labels = labels.to(args.device, non_blocking=True)
            context = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if device_type == "cuda" else torch.no_grad()
            with context:
                result = model(input_ids, labels=labels)
            token_count = int((labels[:, 1:] != -100).sum().item())
            total_nll += float(result.loss.float().item()) * token_count
            supervised_tokens += token_count

    loss = total_nll / supervised_tokens
    payload = {
        "kind": args.kind,
        "checkpoint": args.checkpoint.as_posix(),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "data_path": args.data_path.as_posix(),
        "data_sha256": file_sha256(args.data_path),
        "samples": len(dataset),
        "supervised_tokens": supervised_tokens,
        "loss": loss,
        "perplexity": math.exp(loss),
        "batch_size": args.batch_size,
        "max_seq_len": args.max_seq_len,
        "duration_seconds": time.perf_counter() - started,
        "peak_gpu_memory_mib": round(torch.cuda.max_memory_allocated() / 1024**2, 3) if device_type == "cuda" else None,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
