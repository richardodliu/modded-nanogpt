import tempfile
from pathlib import Path
import unittest
import torch
from nanogpt_research.checkpoint import CheckpointManager
from nanogpt_research.data import distributed_data_generator, shard_num_tokens
from nanogpt_research.losses import (
    cross_entropy_with_centered_logit_regularization as loss_fn,
)
from nanogpt_research.model import GPT
from nanogpt_research.initialization import initialize_shared_nonzero_parameters


class CoreTests(unittest.TestCase):
    def test_loss_gradients(self):
        for regularizer in (0.0, 0.1):
            x = torch.randn(2, 3, 7, requires_grad=True)
            y = torch.tensor([[1, 2, -1], [3, 4, 5]])
            actual, stats = loss_fn(x, y, regularizer, measure_logit_variance=False)
            expected = torch.nn.functional.cross_entropy(
                x.flatten(0, 1), y.flatten(), ignore_index=-1, reduction="sum"
            )
            expected = expected + regularizer * x.var(-1, correction=0)[y != -1].sum()
            torch.testing.assert_close(actual, expected)
            torch.testing.assert_close(
                torch.autograd.grad(actual, x, retain_graph=True)[0],
                torch.autograd.grad(expected, x)[0],
            )
            self.assertEqual(stats[-1].item(), 5)

    def test_stream_crosses_shards_and_resumes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for index, (start, end) in enumerate(((0, 7), (7, 19))):
                header = torch.zeros(256, dtype=torch.int32)
                header[:3] = torch.tensor([20240520, 1, end - start])
                path = root / f"{index}.bin"
                path.write_bytes(
                    header.numpy().tobytes()
                    + torch.arange(start, end).to(torch.uint16).numpy().tobytes()
                )
            batches = distributed_data_generator(
                root, "*.bin", 8, seq_len=4, target_device="cpu"
            )
            next(batches)
            x, y = next(batches)
            self.assertEqual(x.flatten().tolist(), list(range(8, 16)))
            self.assertEqual(y.flatten().tolist(), list(range(9, 17)))
            resumed = next(
                distributed_data_generator(
                    root, "*.bin", 8, seq_len=4, start_step=1, target_device="cpu"
                )
            )
            torch.testing.assert_close(x, resumed[0])
            (root / "bad.bin").write_bytes(b"incomplete")
            with self.assertRaises(RuntimeError):
                shard_num_tokens(root / "bad.bin")

    def test_checkpoint_exact_resume(self):
        torch.manual_seed(41)
        model = torch.nn.Linear(4, 3)
        opt = torch.optim.AdamW(model.parameters(), lr=0.02)
        x = torch.randn(2, 4)

        def step():
            opt.zero_grad()
            model(x).square().sum().backward()
            opt.step()

        step()
        with tempfile.TemporaryDirectory() as temp:
            manager = CheckpointManager(temp, {"recipe": "unit"})
            manager.save(model, [opt], step=1, consumed_tokens=8)
            random_after = torch.rand(3)
            step()
            expected = {k: v.clone() for k, v in model.state_dict().items()}
            (Path(temp) / ".step_000000003.tmp-interrupted").mkdir()
            progress = manager.load(model, [opt])
            self.assertEqual(progress.step, 1)
            torch.testing.assert_close(torch.rand(3), random_after, rtol=0, atol=0)
            step()
            for key, value in model.state_dict().items():
                torch.testing.assert_close(value, expected[key], rtol=0, atol=0)
            with self.assertRaises(ValueError):
                CheckpointManager(temp, {"recipe": "other"}).load(model, [opt])
            (Path(temp) / "step_000000001/rank_00000.pt").unlink()
            with self.assertRaises(RuntimeError):
                manager.load(model, [opt])

    def test_tiny_model_backward(self):
        model = GPT(64, 1, 128)
        initialize_shared_nonzero_parameters(model, 0)
        loss, _ = model(torch.randint(64, (1, 4)), torch.randint(64, (1, 4)))
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(
            all(
                p.grad is not None and torch.isfinite(p.grad).all()
                for p in model.parameters()
            )
        )
