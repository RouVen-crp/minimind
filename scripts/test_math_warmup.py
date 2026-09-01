import unittest
from transformers import AutoTokenizer
from prepare_math_warmup import encode_supervision
from math_warmup import collate


class SupervisionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = AutoTokenizer.from_pretrained('model', local_files_only=True)

    def test_targets_include_answer_and_eos_without_prompt_leak(self):
        row = encode_supervision(self.tokenizer, [{'role': 'user', 'content': 'Compute 2 + 3.'}], '2 + 3 = 5.\nAnswer: 5')
        n = row['prompt_tokens']
        self.assertEqual(row['labels'][:n], [-100] * n)
        self.assertEqual(row['labels'][n:], row['input_ids'][n:])
        self.assertEqual(row['labels'][-1], self.tokenizer.eos_token_id)
        self.assertIn('Answer: 5', self.tokenizer.decode(row['labels'][n:-1]))

    def test_long_target_is_rejected_instead_of_truncated(self):
        self.assertIsNone(encode_supervision(self.tokenizer, [{'role': 'user', 'content': 'Compute.'}],
                                             'reasoning ' * 200 + '\nAnswer: 5', max_length=64))

    def test_batch_padding_is_never_supervised(self):
        ids, labels = collate([{'input_ids': [1, 3, 2], 'labels': [-100, 3, 2]},
                               {'input_ids': [1, 3, 4, 2], 'labels': [-100, 3, 4, 2]}])
        self.assertEqual(ids[0, -1].item(), 0)
        self.assertEqual(labels[0, -1].item(), -100)


if __name__ == '__main__':
    unittest.main()
