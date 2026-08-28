import unittest
from pathlib import Path

import torch

from scripts.uav_minimind_adapter import MiniMindPlanner


class _Batch(dict):
    def to(self, _device):
        return self


class _Tokenizer:
    pad_token_id = 0
    eos_token_id = 2

    def apply_chat_template(self, messages, **_kwargs):
        self.messages = messages
        return "rendered"

    def __call__(self, _prompt, **_kwargs):
        return _Batch(input_ids=torch.tensor([[10, 11]]), attention_mask=torch.tensor([[1, 1]]))

    def decode(self, tokens, **_kwargs):
        self.decoded = tokens.tolist()
        return ' {"status":"needs_clarification"} '


class _Model:
    def generate(self, **_kwargs):
        return torch.tensor([[10, 11, 20, 21]])


class MiniMindAdapterTests(unittest.TestCase):
    def test_generate_uses_contract_messages_and_decodes_only_completion(self) -> None:
        tokenizer = _Tokenizer()
        calls = []

        def builder(instruction, feedback, examples):
            calls.append((instruction, feedback, examples))
            return [{"role": "user", "content": instruction}]

        planner = MiniMindPlanner(
            project_root=Path("unused"),
            checkpoint=Path("unused.pth"),
            tokenizer_path=Path("unused"),
            device="cpu",
            model=_Model(),
            tokenizer=tokenizer,
            message_builder=builder,
        )
        response = planner.generate("任务", feedback="修复", examples=[{"id": "example"}])
        self.assertEqual(response, '{"status":"needs_clarification"}')
        self.assertEqual(tokenizer.decoded, [20, 21])
        self.assertEqual(calls[0][0:2], ("任务", "修复"))


if __name__ == "__main__":
    unittest.main()

