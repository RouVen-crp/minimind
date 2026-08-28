"""MiniMind planner adapter for the graduation project's UAV evaluator."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Callable


def load_prompt_builder(project_root: Path) -> Callable[..., list[dict]]:
    root = str(project_root.resolve())
    sys.path.insert(0, root)
    try:
        return importlib.import_module("drone_planner.prompts").build_messages
    finally:
        sys.path.remove(root)


class MiniMindPlanner:
    def __init__(
        self,
        *,
        project_root: Path,
        checkpoint: Path,
        tokenizer_path: Path,
        device: str = "cuda",
        hidden_size: int = 768,
        num_hidden_layers: int = 8,
        max_new_tokens: int = 512,
        model=None,
        tokenizer=None,
        message_builder: Callable[..., list[dict]] | None = None,
    ) -> None:
        self.project_root = project_root
        self.checkpoint = checkpoint
        self.tokenizer_path = tokenizer_path
        self.device = device
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.max_new_tokens = max_new_tokens
        self.message_builder = message_builder or load_prompt_builder(project_root)
        self.model = model
        self.tokenizer = tokenizer
        if self.model is None or self.tokenizer is None:
            self._load_model()

    def _load_model(self) -> None:
        import torch
        from transformers import AutoTokenizer

        from model.model_minimind import MiniMindConfig, MiniMindForCausalLM

        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
        config = MiniMindConfig(
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            use_moe=False,
        )
        self.model = MiniMindForCausalLM(config)
        state = torch.load(self.checkpoint, map_location=self.device)
        self.model.load_state_dict(state, strict=True)
        if self.device.startswith("cuda"):
            self.model = self.model.half()
        self.model = self.model.eval().to(self.device)

    def generate(
        self,
        instruction: str,
        feedback: str | None = None,
        examples: list[dict] | None = None,
    ) -> str:
        import torch

        messages = self.message_builder(instruction, feedback, examples)
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            open_thinking=False,
        )
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True).to(self.device)
        with torch.no_grad():
            generated = self.model.generate(
                inputs=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.0,
            )
        prompt_tokens = inputs["input_ids"].shape[-1]
        return self.tokenizer.decode(generated[0][prompt_tokens:], skip_special_tokens=True).strip()

    def generate_many(self, instructions: list[str], examples: list[dict] | None = None) -> list[str]:
        return [self.generate(instruction, examples=examples) for instruction in instructions]

