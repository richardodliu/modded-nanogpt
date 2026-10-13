"""Two CPU ranks verify optimizer ownership, synchronization and exact resume."""

import tempfile
from pathlib import Path
import unittest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from nanogpt_research.checkpoint import CheckpointManager
from nanogpt_research.optimizers.track3 import MuonH


def worker(rank, rendezvous, checkpoint_dir):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo", init_method="file://" + rendezvous, rank=rank, world_size=2
    )
    try:
        torch.manual_seed(123)
        model = torch.nn.ParameterList(
            [torch.nn.Parameter(torch.randn(8, 8)) for _ in range(3)]
        )
        optimizer = MuonH(list(model), lr=0.01)

        def step():
            for p in model:
                p.grad = torch.randn_like(p)
            optimizer.step()

        step()
        manager = CheckpointManager(checkpoint_dir, {"test": "sharded"})
        manager.save(model, [optimizer], step=1, consumed_tokens=32)
        assert len(optimizer.state) == (2 if rank == 0 else 1)
        step()
        expected = [p.detach().clone() for p in model]
        manager.load(model, [optimizer])
        step()
        for actual, wanted in zip(model, expected):
            torch.testing.assert_close(actual, wanted, rtol=0, atol=0)
            copies = [torch.empty_like(actual) for _ in range(2)]
            dist.all_gather(copies, actual)
            torch.testing.assert_close(copies[0], copies[1], rtol=0, atol=0)
    finally:
        dist.destroy_process_group()


class DistributedTrack3Tests(unittest.TestCase):
    def test_rank_sharded_resume(self):
        with tempfile.TemporaryDirectory() as temp:
            mp.spawn(
                worker,
                args=(str(Path(temp) / "rendezvous"), str(Path(temp) / "checkpoint")),
                nprocs=2,
                join=True,
            )
