import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from math_grpo_readiness import parse_args
from run_math_full import validate_config, file_sha256


class FullGuardTests(unittest.TestCase):
    def test_full_is_never_implicit(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parse_args(['--data-dir', '.', '--output-dir', 'unused', '--checkpoint', 'unused', '--full'])

    def test_small_mode_rejects_full_sized_group_count(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parse_args(['--data-dir', '.', '--output-dir', 'unused', '--checkpoint', 'unused', '--max-groups', '1000'])

    def test_manifest_content_change_and_long_prompt_are_rejected(self):
        with tempfile.TemporaryDirectory() as name:
            folder = Path(name)
            data = folder / 'train.jsonl'
            data.write_text(json.dumps({'id': 'one', 'prompt_tokens': 12}) + '\n')
            config = {'model': 'MiniMind-63.9M', 'epochs': 1, 'required_sha256': {str(data): file_sha256(data)},
                      'data_dir': str(folder), 'train_questions': 1, 'max_prompt_tokens': 12,
                      'num_generations': 8, 'wall_time_limit_seconds': 3600, 'output_dir': str(folder / 'out')}
            self.assertEqual(validate_config(config), 1)
            config['max_prompt_tokens'] = 11
            with self.assertRaisesRegex(ValueError, 'truncate'):
                validate_config(config)
            config['max_prompt_tokens'] = 12
            data.write_text(json.dumps({'id': 'two', 'prompt_tokens': 12}) + '\n')
            with self.assertRaisesRegex(ValueError, 'fingerprint'):
                validate_config(config)


if __name__ == '__main__':
    unittest.main()
