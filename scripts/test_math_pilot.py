import unittest
import torch
from math_pilot import completion_mask, clipped_grpo_loss


class PilotMathTests(unittest.TestCase):
    def test_mask_includes_first_eos_not_padding(self):
        ids = torch.tensor([[7, 2, 2, 2], [2, 2, 2, 2], [7, 8, 9, 10]])
        self.assertEqual(completion_mask(ids, 2).tolist(),
                         [[True, True, False, False], [True, False, False, False],
                          [True, True, True, True]])

    def test_zero_variance_has_zero_correctness_gradient(self):
        logps = torch.full((4, 3), -2.0, requires_grad=True)
        loss, advantages, _ = clipped_grpo_loss(logps, logps.detach(), logps.detach(),
                                                torch.full((4,), -1.), torch.ones_like(logps))
        loss.backward()
        self.assertTrue(torch.equal(advantages, torch.zeros(4)))
        self.assertTrue(torch.equal(logps.grad, torch.zeros_like(logps)))

    def test_correct_completion_increases_probability_and_masks_suffix(self):
        logps = torch.full((2, 3), -2.0, requires_grad=True)
        mask = torch.tensor([[1., 1., 0.], [1., 1., 1.]])
        loss, _, _ = clipped_grpo_loss(logps, logps.detach(), logps.detach(),
                                       torch.tensor([1., -1.]), mask)
        loss.backward()
        self.assertLess(logps.grad[0, 0].item(), 0)
        self.assertGreater(logps.grad[1, 0].item(), 0)
        self.assertEqual(logps.grad[0, 2].item(), 0)

    def test_clip_blocks_overlarge_positive_advantage_ratio(self):
        current = torch.tensor([[0.5], [0.]], requires_grad=True)
        old = torch.zeros_like(current)
        loss, _, _ = clipped_grpo_loss(current, old, old, torch.tensor([1., -1.]),
                                       torch.ones_like(current), beta=0)
        loss.backward()
        self.assertAlmostEqual(current.grad[0, 0].item(), 0)
        self.assertGreater(current.grad[1, 0].item(), 0)


if __name__ == '__main__':
    unittest.main()
