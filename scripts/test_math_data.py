import unittest
from math_data import QuestionIndex, verify_answer


class MathDataTests(unittest.TestCase):
    def test_integer_formats_and_reasoning(self):
        for text, answer in [('work\nAnswer: -3', '-3'), ('Answer: $1007$', '1007'),
                             ('Answer: +0034', '34'), ('Answer: \u22123', '-3')]:
            self.assertTrue(verify_answer(text, answer)['correct'])

    def test_guess_lists_and_multiple_answers_do_not_get_credit(self):
        for text in ['34', 'Reasoning contains 34.', 'Answer: 1, 34, 56',
                     'Answer: 34\nAnswer: 35', 'Answer: 34\nActually 35',
                     'Answer: 34+0', 'Answer: $34', 'Answer: 3.4e1']:
            self.assertFalse(verify_answer(text, '34')['correct'], text)

    def test_wrong_but_parseable_answer(self):
        result = verify_answer('Answer: 1007', '-3')
        self.assertTrue(result['parsed'])
        self.assertEqual(result['reward'], -1)

    def test_duplicates_and_conflicts_are_not_randomly_split(self):
        index = QuestionIndex()
        def add(text, answer, sid):
            index.add({'prompt': [{'role': 'user', 'content': text}],
                       'reward_model': {'ground_truth': answer}, 'extra_info': {'index': sid}})
        add('same question', '1', 'a')
        add('same  question', '01', 'b')
        add('conflict', '2', 'c')
        add('conflict', '3', 'd')
        add('bad', 'unknown', 'e')
        valid, conflicts, stats = index.finish()
        self.assertEqual(len(valid), 1)
        self.assertEqual(valid[0]['copies'], 2)
        self.assertEqual(len(conflicts), 1)
        self.assertEqual(stats['invalid_rows']['non_integer_answer'], 1)


if __name__ == '__main__':
    unittest.main()
