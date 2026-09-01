"""DAPO adapters and a deliberately strict, auditable integer verifier."""
import hashlib
import json
import re
import unicodedata
from collections import Counter


def canonical_integer(value):
    text = unicodedata.normalize('NFKC', str(value)).strip().replace('\u2212', '-')
    if not re.fullmatch(r'[+-]?\d+', text, flags=re.ASCII) or len(text) > 100:
        return None
    return str(int(text))


def verify_answer(text, ground_truth):
    """Only one final Answer: line; never match a guessed number in reasoning.

    This is our stricter rule, not a byte-for-byte copy of DAPO's verifier.
    Accept signs, leading zeroes, and optional $...$ around an integer.
    Multiple answer markers, lists, expressions, and trailing prose fail.
    """
    markers = re.findall(r'(?i)\banswer\s*:', text)
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    match = re.fullmatch(r'Answer:\s*(\$?)([+\-\u2212]?\d+)\1', lines[-1], re.I) if lines else None
    answer = canonical_integer(match.group(2)) if match and len(markers) == 1 else None
    expected = canonical_integer(ground_truth)
    correct = answer is not None and expected is not None and answer == expected
    return {'parsed': answer is not None, 'prediction': answer, 'correct': correct,
            'reward': 1.0 if correct else -1.0}


def canonical_prompt(messages):
    return json.dumps([{'role': m['role'], 'content': re.sub(r'\s+', ' ', m['content']).strip()}
                       for m in messages], ensure_ascii=False, sort_keys=True)


def problem_text(messages):
    text = '\n'.join(m['content'] for m in messages if m['role'] == 'user')
    text = re.sub(r'^Solve the following math problem step by step\..*?\n\s*\n', '', text, flags=re.S)
    text = re.sub(r'\n\s*Remember to put your answer.*$', '', text, flags=re.S)
    return re.sub(r'\s+', '', unicodedata.normalize('NFKC', text)).lower()


def content_id(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


class QuestionIndex:
    def __init__(self):
        self.rows = 0
        self.invalid = Counter()
        self.groups = {}

    def add(self, row):
        self.rows += 1
        messages = row.get('prompt')
        answer = canonical_integer((row.get('reward_model') or {}).get('ground_truth'))
        if not isinstance(messages, list) or not messages or any(
            not isinstance(m, dict) or m.get('role') not in ('system', 'user')
            or not isinstance(m.get('content'), str) or not m['content'].strip() for m in messages
        ):
            self.invalid['invalid_prompt'] += 1
            return
        if answer is None:
            self.invalid['non_integer_answer'] += 1
            return
        key = canonical_prompt(messages)
        entry = self.groups.setdefault(key, {
            'id': content_id(key), 'prompt': messages, 'answers': set(), 'copies': 0,
            'source_ids': [], 'problem_key': content_id(problem_text(messages)),
            'language': 'contains_cjk' if re.search(r'[\u4e00-\u9fff]', key) else 'other',
        })
        entry['answers'].add(answer)
        entry['copies'] += 1
        sid = str((row.get('extra_info') or {}).get('index', ''))
        if sid not in entry['source_ids'] and len(entry['source_ids']) < 4:
            entry['source_ids'].append(sid)

    def finish(self):
        valid, conflicts = [], []
        for entry in self.groups.values():
            item = dict(entry, answers=sorted(entry['answers']))
            if len(item['answers']) != 1:
                conflicts.append(item)
            else:
                item['answer'] = item.pop('answers')[0]
                valid.append(item)
        valid.sort(key=lambda x: x['id'])
        return valid, conflicts, {
            'raw_rows': self.rows, 'invalid_rows': dict(self.invalid),
            'unique_prompt_groups': len(self.groups), 'valid_questions': len(valid),
            'conflicting_prompt_groups': len(conflicts),
            'conflicting_raw_rows': sum(x['copies'] for x in conflicts),
            'duplicate_valid_rows_removed': sum(x['copies'] - 1 for x in valid),
            'languages': dict(Counter(x['language'] for x in valid)),
        }
