"""Deterministic cross-shard token streaming for the shared 10B dataset."""

from __future__ import annotations

from bisect import bisect_right
from pathlib import Path
from typing import Iterator

import torch
import torch.distributed as dist


MAGIC = 20240520
VERSION = 1
HEADER_INTS = 256


def shard_num_tokens(path: Path) -> int:
    if path.stat().st_size < HEADER_INTS * 4:
        raise RuntimeError(f"truncated token shard header: {path}")
    header = torch.from_file(str(path), False, HEADER_INTS, dtype=torch.int32)
    if int(header[0]) != MAGIC or int(header[1]) != VERSION:
        raise RuntimeError(f"invalid NanoGPT token shard header: {path}")
    count = int(header[2])
    if count < 0 or path.stat().st_size != HEADER_INTS * 4 + count * 2:
        raise RuntimeError(f"token shard length does not match header: {path}")
    return count


def load_shard(path: Path, *, pin_memory: bool = True) -> torch.Tensor:
    count = shard_num_tokens(path)
    with path.open("rb", buffering=0) as handle:
        tokens = torch.empty(count, dtype=torch.uint16, pin_memory=pin_memory)
        handle.seek(HEADER_INTS * 4)
        nbytes = handle.readinto(tokens.numpy())
    if nbytes != count * 2:
        raise RuntimeError(f"truncated token shard: {path}")
    return tokens


def resolve_shards(root: Path, pattern: str) -> list[Path]:
    files = sorted(root.glob(pattern))
    if not files:
        raise RuntimeError(f"no token shards match {pattern!r} under {root}")
    return files


def capacity_tokens(files: list[Path]) -> int:
    return sum(shard_num_tokens(path) for path in files)


def capacity_steps(files: list[Path], batch_tokens: int) -> int:
    if batch_tokens <= 0:
        raise ValueError("batch_tokens must be positive")
    return max((capacity_tokens(files) - 1) // batch_tokens, 0)


class ShardTokenReader:
    """Read a logical concatenation of shards while retaining one-token overlap."""

    def __init__(self, files: list[Path], *, pin_memory: bool = True):
        if not files:
            raise ValueError("at least one shard is required")
        self.files = files
        self.lengths = [shard_num_tokens(path) for path in files]
        self.prefix = [0]
        for length in self.lengths:
            self.prefix.append(self.prefix[-1] + length)
        self.pin_memory = pin_memory
        self._cached_index = -1
        self._cached_tokens: torch.Tensor | None = None

    @property
    def total_tokens(self) -> int:
        return self.prefix[-1]

    def _tokens(self, index: int) -> torch.Tensor:
        if index != self._cached_index:
            self._cached_tokens = load_shard(
                self.files[index], pin_memory=self.pin_memory
            )
            self._cached_index = index
        assert self._cached_tokens is not None
        return self._cached_tokens

    def read(self, offset: int, count: int) -> torch.Tensor:
        if offset < 0 or count <= 0 or offset + count > self.total_tokens:
            raise RuntimeError(
                f"token span [{offset}, {offset + count}) exceeds capacity {self.total_tokens}"
            )
        pieces: list[torch.Tensor] = []
        remaining = count
        position = offset
        while remaining:
            index = min(bisect_right(self.prefix, position) - 1, len(self.files) - 1)
            local_offset = position - self.prefix[index]
            take = min(remaining, self.lengths[index] - local_offset)
            pieces.append(self._tokens(index)[local_offset : local_offset + take])
            position += take
            remaining -= take
        return pieces[0] if len(pieces) == 1 else torch.cat(pieces)


def distributed_data_generator(
    root: Path,
    pattern: str,
    batch_tokens: int,
    *,
    seq_len: int = 1024,
    start_step: int = 0,
    target_device: str | torch.device = "cuda",
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    files = resolve_shards(root, pattern)
    device = torch.device(target_device)
    reader = ShardTokenReader(files, pin_memory=device.type == "cuda")
    distributed = dist.is_available() and dist.is_initialized()
    world_size = dist.get_world_size() if distributed else 1
    rank = dist.get_rank() if distributed else 0
    if (
        batch_tokens <= 0
        or seq_len <= 0
        or start_step < 0
        or int(start_step) != start_step
    ):
        raise ValueError(
            "batch_tokens and seq_len must be positive; start_step must be a nonnegative integer"
        )
    if batch_tokens % world_size or batch_tokens % seq_len:
        raise ValueError("batch_tokens must be divisible by world_size and seq_len")
    local_tokens = batch_tokens // world_size
    if local_tokens % seq_len:
        raise ValueError("per-rank token count must be divisible by seq_len")
    step = int(start_step)
    while True:
        global_offset = step * batch_tokens
        begin = rank * local_tokens
        # Keep exactly the same global stream while loading only this rank's span.
        if global_offset + batch_tokens + 1 > reader.total_tokens:
            raise RuntimeError("insufficient tokens for a complete global batch")
        local = reader.read(global_offset + begin, local_tokens + 1)
        inputs = local[:-1].to(target_device, dtype=torch.int32, non_blocking=True)
        targets = local[1:].to(target_device, dtype=torch.int64, non_blocking=True)
        yield inputs.view(-1, seq_len), targets.view(-1, seq_len)
        step += 1
