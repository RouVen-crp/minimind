"""One bounded GRPO feasibility pilot; never launches SFT or a full epoch."""
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import signal
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.math_data import verify_answer


def write_json(path, value):
    temporary = Path(str(path) + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    os.replace(temporary, path)


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line]


def completion_mask(ids, eos_id):
    # Include the first EOS, exclude every token after it (including repeated EOS).
    eos = ids.eq(eos_id)
    return (eos.cumsum(dim=1) - eos.long()).eq(0)


def clipped_grpo_loss(logps, old_logps, reference_logps, rewards, mask, beta=0.04, epsilon=0.2):
    import torch
    advantages = (rewards - rewards.mean()) / (rewards.std(unbiased=False) + 1e-4)
    ratio = (logps - old_logps).exp()
    surrogate = torch.minimum(ratio * advantages[:, None],
                              ratio.clamp(1 - epsilon, 1 + epsilon) * advantages[:, None])
    delta = reference_logps - logps
    kl = delta.exp() - delta - 1
    per_sequence = ((-surrogate + beta * kl) * mask).sum(1) / mask.sum(1).clamp_min(1)
    return per_sequence.mean(), advantages, kl


class Pilot:
    def __init__(self, args):
        import torch
        self.torch = torch
        self.args = args
        self.out = Path(args.output_dir)
        self.deadline = float(os.environ['MATH_PILOT_DEADLINE'])
        self.started = time.time()
        self.summary = {'status': 'initializing', 'started_unix': self.started,
                        'deadline_unix': self.deadline, 'config': vars(args),
                        'optimizer_steps': 0, 'nonzero_parameter_updates': 0,
                        'group_metrics': [], 'benchmark_only': [], 'aime_before': [], 'aime_after': []}
        self.stop_requested = False
        signal.signal(signal.SIGTERM, lambda *unused: setattr(self, 'stop_requested', True))

    def emit(self, event, **values):
        row = {'event': event, 'unix': time.time(), **values}
        with (self.out / 'events.jsonl').open('a', encoding='utf-8') as f:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
        print(json.dumps(row, ensure_ascii=False), flush=True)
        write_json(self.out / 'summary.json', self.summary)

    def check_time(self, reserve=0):
        if self.stop_requested or time.time() >= self.deadline - reserve:
            raise TimeoutError('Pilot GPU wall-clock budget reached')

    def synchronize(self):
        self.torch.cuda.synchronize()

    def save_resume(self, next_index):
        torch = self.torch
        path = self.out / 'resume.pth'
        payload = {'model': {n: p.detach().cpu() for n, p in self.model.state_dict().items()},
                   'optimizer': self.optimizer.state_dict(), 'next_index': next_index,
                   'summary': self.summary, 'selected_ids': self.summary['selected_ids'],
                   'reference_checkpoint': str(Path(self.args.checkpoint).resolve()),
                   'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all(),
                   'python_rng': random.getstate()}
        torch.save(payload, str(path) + '.tmp')
        os.replace(str(path) + '.tmp', path)
        self.summary['resume_next_index'] = next_index

    def generate(self, row, count, max_new_tokens, sample):
        torch = self.torch
        self.check_time(reserve=90)
        prompt = self.tokenizer.apply_chat_template(row['prompt'], tokenize=False,
                                                    add_generation_prompt=True, open_thinking=False)
        ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        if len(ids) > self.args.max_prompt_tokens:
            raise ValueError('Prompt too long; do not silently truncate mathematics')
        inputs = torch.tensor([ids] * count, device='cuda')
        self.synchronize()
        start = time.perf_counter()
        self.model.eval()
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            # Temperature 1 and no top-k/p truncation: sampled and scored policies match.
            output = self.model.generate(input_ids=inputs, max_new_tokens=max_new_tokens,
                                         do_sample=sample, temperature=1.0, top_k=0, top_p=1.0,
                                         eos_token_id=self.tokenizer.eos_token_id, use_cache=True,
                                         logits_to_keep=1).clone()
        self.synchronize()
        elapsed = time.perf_counter() - start
        completion = output[:, len(ids):]
        mask = completion_mask(completion, self.tokenizer.eos_token_id)
        texts = [self.tokenizer.decode(tokens[keep].tolist(), skip_special_tokens=True)
                 for tokens, keep in zip(completion, mask)]
        checked = [verify_answer(text, row['answer']) for text in texts]
        metrics = {'question_id': row['id'], 'language': row['language'], 'prompt_tokens': len(ids),
                   'max_new_tokens': max_new_tokens, 'num_generations': count, 'sampling': sample,
                   'generation_seconds': elapsed, 'completion_tokens': mask.sum(1).tolist(),
                   'truncated': (~completion.eq(self.tokenizer.eos_token_id).any(1)).tolist(),
                   'correct': [r['correct'] for r in checked], 'parsed': [r['parsed'] for r in checked]}
        with (self.out / 'responses.jsonl').open('a', encoding='utf-8') as f:
            for number, (text, result) in enumerate(zip(texts, checked)):
                f.write(json.dumps({'phase': self.summary['status'], 'question_id': row['id'],
                                    'generation': number, 'ground_truth': row['answer'],
                                    'response': text, **result}, ensure_ascii=False) + '\n')
        return output, completion, mask, checked, metrics

    def logps(self, model, output, completion):
        torch = self.torch
        with torch.autocast('cuda', dtype=torch.bfloat16):
            logits = model(output, logits_to_keep=completion.size(1) + 1).logits[:, :-1].float()
        log_probs = logits.log_softmax(-1)
        selected = log_probs.gather(-1, completion[..., None]).squeeze(-1)
        with torch.no_grad():
            entropy = -(log_probs.exp() * log_probs).sum(-1)
        return selected, entropy

    def eval_aime(self, rows, phase):
        self.summary['status'] = phase
        for row in rows:
            if row['prompt_tokens'] > self.args.max_prompt_tokens:
                self.summary[phase].append({'question_id': row['id'], 'skipped': 'prompt_too_long'})
                continue
            output, completion, mask, checked, metrics = self.generate(row, 1, self.args.max_new_tokens, False)
            self.summary[phase].append(metrics)
            self.emit('evaluation_question', phase=phase, **metrics)
            del output, completion, mask

    def run(self):
        import torch
        from transformers import AutoTokenizer
        from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
        torch.set_num_threads(4)
        torch.manual_seed(self.args.seed)
        random.seed(self.args.seed)
        torch.cuda.manual_seed_all(self.args.seed)
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.tokenizer, local_files_only=True)
        config = MiniMindConfig(hidden_size=768, num_hidden_layers=8, use_moe=False)
        self.model = MiniMindForCausalLM(config)
        weights = torch.load(self.args.checkpoint, map_location='cpu', weights_only=True)
        self.model.load_state_dict(weights, strict=True)
        del weights
        self.model.to('cuda')
        self.reference = copy.deepcopy(self.model).eval().requires_grad_(False)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.args.learning_rate, weight_decay=0.0)
        self.summary['parameters'] = sum(p.numel() for p in self.model.parameters())
        self.summary['gpu'] = torch.cuda.get_device_name()
        self.summary['torch'] = torch.__version__
        self.emit('initialized', parameters=self.summary['parameters'])
        train_path = Path(getattr(self.args, 'probe_file', None) or Path(self.args.data_dir) / 'train.jsonl')
        train = read_rows(train_path)
        self.summary['data_sha256'] = hashlib.sha256(train_path.read_bytes()).hexdigest()
        self.summary['reference_sha256'] = hashlib.sha256(Path(self.args.checkpoint).read_bytes()).hexdigest()
        aime = read_rows(Path(self.args.data_dir) / 'aime.jsonl')
        eligible = sorted([r for r in train if r['prompt_tokens'] <= self.args.max_prompt_tokens],
                          key=lambda r: (r['prompt_tokens'], r['id']))
        self.summary['eligible_train_questions'] = len(eligible)
        self.summary['ineligible_long_prompts'] = len(train) - len(eligible)
        if getattr(self.args, 'full', False) and len(eligible) != len(train):
            raise ValueError('Full coverage would drop long prompts; adjust configuration explicitly')
        # Systematic length-stratified sample, followed by a fixed random order.
        count = min(self.args.max_groups, len(eligible))
        chosen = [eligible[min(len(eligible) - 1, int((i + 0.5) * len(eligible) / count))]
                  for i in range(count)]
        # Include the longest prompt to test the complete dataset's memory envelope.
        # This is a feasibility sample, not an unbiased performance estimator.
        if chosen:
            chosen[-1] = eligible[-1]
        random.Random(self.args.seed).shuffle(chosen)
        self.summary['selected_ids'] = [r['id'] for r in chosen]
        start_index = 0
        if getattr(self.args, 'resume_checkpoint', None):
            resumed = torch.load(self.args.resume_checkpoint, map_location='cpu', weights_only=False)
            if resumed['selected_ids'] != self.summary['selected_ids']:
                raise ValueError('Resume dataset/order changed')
            if resumed['reference_checkpoint'] != str(Path(self.args.checkpoint).resolve()):
                raise ValueError('Resume reference checkpoint changed')
            old_config = resumed['summary']['config']
            for fingerprint in ['data_sha256', 'reference_sha256']:
                previous_hash = resumed['summary'].get(fingerprint)
                if previous_hash is not None and previous_hash != self.summary[fingerprint]:
                    raise ValueError('Resume content fingerprint changed: ' + fingerprint)
                if getattr(self.args, 'full', False) and previous_hash is None:
                    raise ValueError('Legacy diagnostic checkpoint cannot resume a full run')
            for name in ['num_generations', 'max_new_tokens', 'max_prompt_tokens', 'beta', 'learning_rate', 'seed']:
                if old_config[name] != getattr(self.args, name):
                    raise ValueError('Resume policy configuration changed: ' + name)
            self.model.load_state_dict(resumed['model'], strict=True)
            self.optimizer.load_state_dict(resumed['optimizer'])
            previous = resumed['summary']
            previous.update({'config': vars(self.args), 'resumed_from': self.args.resume_checkpoint})
            self.summary = previous
            start_index = resumed['next_index']
            torch.set_rng_state(resumed['torch_rng'])
            torch.cuda.set_rng_state_all(resumed['cuda_rng'])
            random.setstate(resumed['python_rng'])
            del resumed
            self.emit('resumed', next_index=start_index)
        else:
            self.eval_aime(aime, 'aime_before')
        self.summary['status'] = 'grpo_pilot'
        no_signal = 0
        next_index = start_index
        for index, row in enumerate(chosen[start_index:], start=start_index):
            self.check_time(reserve=180)
            self.torch.cuda.reset_peak_memory_stats()
            output, completion, mask, results, metrics = self.generate(
                row, self.args.num_generations, self.args.max_new_tokens, True)
            rewards = torch.tensor([r['reward'] for r in results], device='cuda')
            self.synchronize()
            started = time.perf_counter()
            with torch.no_grad():
                old_logps, _ = self.logps(self.model, output, completion)
                reference_logps, _ = self.logps(self.reference, output, completion)
            self.model.train()
            current_logps, entropy = self.logps(self.model, output, completion)
            loss, advantages, kl = clipped_grpo_loss(current_logps, old_logps, reference_logps,
                                                    rewards, mask, beta=self.args.beta)
            if not torch.isfinite(loss):
                raise FloatingPointError('nonfinite loss')
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0, error_if_nonfinite=True)
            signal_present = rewards.std(unbiased=False).item() > 0
            # All-equal rewards provide no relative correctness signal. Never call AdamW just
            # to create a weight-decay change and pretend that correctness was learned.
            changed_max = 0.0
            if signal_present:
                before = {n: p.detach().clone() for n, p in self.model.named_parameters()}
                self.optimizer.step()
                changed_max = max((p.detach() - before[n]).abs().max().item()
                                  for n, p in self.model.named_parameters())
                del before
                self.summary['optimizer_steps'] += 1
                self.summary['nonzero_parameter_updates'] += int(changed_max > 0)
            self.optimizer.zero_grad(set_to_none=True)
            self.synchronize()
            metrics.update({'update_path_seconds': time.perf_counter() - started,
                            'correctness_signal': signal_present, 'optimizer_step': signal_present,
                            'loss': loss.item(), 'gradient_norm': grad_norm.item(),
                            'max_parameter_delta': changed_max,
                            'advantage_std': advantages.std(unbiased=False).item(),
                            'kl': (kl.detach() * mask).sum().item() / mask.sum().item(),
                            'entropy': (entropy * mask).sum().item() / mask.sum().item(),
                            'peak_allocated_mib': torch.cuda.max_memory_allocated() / 1024**2,
                            'peak_reserved_mib': torch.cuda.max_memory_reserved() / 1024**2})
            self.summary['group_metrics'].append(metrics)
            next_index = index + 1
            self.emit('grpo_group', **metrics)
            no_signal = 0 if signal_present else no_signal + 1
            del output, completion, mask, current_logps, old_logps, reference_logps, entropy, loss, kl
            if getattr(self.args, 'save_interval', 0) and next_index % self.args.save_interval == 0:
                self.save_resume(next_index)
            if getattr(self.args, 'stop_after_updates', 0) and self.summary['nonzero_parameter_updates'] >= self.args.stop_after_updates:
                self.summary['early_stop_reason'] = 'readiness_update_target_reached'
                break
            if no_signal >= self.args.zero_signal_stop:
                self.summary['early_stop_reason'] = 'consecutive_zero_variance_correctness_groups'
                break
        # One longer rollout measures generation-length sensitivity, not a training claim.
        self.save_resume(next_index)
        if time.time() < self.deadline - 240 and eligible and not getattr(self.args, 'skip_length_benchmark', False):
            self.summary['status'] = 'length_benchmark'
            output, completion, mask, results, metrics = self.generate(
                eligible[len(eligible) // 2], self.args.num_generations,
                min(1024, 2 * self.args.max_new_tokens), True)
            self.summary['benchmark_only'].append(metrics)
            self.emit('length_benchmark', **metrics)
            del output, completion, mask
        self.summary['status'] = 'checkpoint_verification'
        self.check_time(reserve=60)
        checkpoint_path = self.out / 'pilot.pth'
        state = {name: value.detach().cpu() for name, value in self.model.state_dict().items()}
        torch.save(state, checkpoint_path)
        reloaded = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        self.summary['checkpoint_tensors_exact'] = all(torch.equal(state[n], reloaded[n]) for n in state)
        self.model.load_state_dict(reloaded, strict=True)
        self.summary['checkpoint_reloaded_strictly'] = True
        self.summary['checkpoint_sha256'] = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
        del state, reloaded
        self.emit('checkpoint_reloaded', exact=self.summary['checkpoint_tensors_exact'])
        if self.summary['optimizer_steps']:
            self.eval_aime(aime, 'aime_after')
        else:
            self.summary['aime_after_skipped'] = 'No optimizer updates; do not waste budget or label copied baseline as rerun'
        self.summary['status'] = 'completed' if self.summary['nonzero_parameter_updates'] else 'needs_math_warmup'
        self.summary['finished_unix'] = time.time()
        self.summary['worker_wall_seconds'] = time.time() - self.started
        self.summary['covered_questions'] = next_index
        self.summary['planned_questions'] = len(chosen)
        self.summary['all_planned_questions_covered'] = next_index == len(chosen)
        self.summary['full_workflow_complete'] = bool(self.summary['nonzero_parameter_updates'] and
                                                     self.summary['aime_after'] and
                                                     self.summary['checkpoint_reloaded_strictly'] and
                                                     (not getattr(self.args, 'full', False) or next_index == len(chosen)))
        if getattr(self.args, 'full', False) and next_index < len(chosen):
            self.summary['status'] = 'full_stopped_before_complete_coverage'
        self.emit('finished', status=self.summary['status'], full_workflow_complete=self.summary['full_workflow_complete'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--checkpoint', default='out/full_sft_768.pth')
    parser.add_argument('--tokenizer', default='model')
    parser.add_argument('--max-groups', type=int, default=24)
    parser.add_argument('--num-generations', type=int, default=4)
    parser.add_argument('--max-prompt-tokens', type=int, default=3072)
    parser.add_argument('--max-new-tokens', type=int, default=512)
    parser.add_argument('--zero-signal-stop', type=int, default=16)
    parser.add_argument('--learning-rate', type=float, default=3e-7)
    parser.add_argument('--beta', type=float, default=0.04)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if args.num_generations < 2:
        parser.error('GRPO requires at least two completions')
    pilot = Pilot(args)
    try:
        pilot.run()
    except BaseException as exc:
        pilot.summary.update({'status': 'budget_stop' if isinstance(exc, TimeoutError) else 'failed',
                              'error': f'{type(exc).__name__}: {exc}', 'finished_unix': time.time(),
                              'full_workflow_complete': False})
        pilot.emit('stopped', error=str(exc))
        raise


if __name__ == '__main__':
    main()
