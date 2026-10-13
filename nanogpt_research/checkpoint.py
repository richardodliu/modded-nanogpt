"""Atomic checkpoints for replicated models and replicated or rank-sharded optimizers.

All optimizer state is saved per rank. This intentionally trades a little storage
for a simple contract that also covers Muon's distributed parameter ownership.
Only a fully committed directory is published through the latest-checkpoint marker.
The format is new; it does not silently load historical experiment checkpoints.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import random
import re
import shutil
import uuid

import torch
import torch.distributed as dist

FORMAT_VERSION = 1
MARKER = ".checkpoint-complete"


def rank_world() -> tuple[int, int]:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    return 0, 1


def barrier() -> None:
    if rank_world()[1] > 1:
        dist.barrier()


def fsync_dir(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def save_file(value: object, path: Path) -> None:
    with path.open("wb") as stream:
        torch.save(value, stream)
        stream.flush()
        os.fsync(stream.fileno())


@dataclass(frozen=True)
class Progress:
    step: int = 0
    consumed_tokens: int = 0
    extra: dict = field(default_factory=dict)


class CheckpointManager:
    def __init__(self, directory: str | Path, config: dict, *, keep_last: int = 2):
        self.directory = Path(directory).expanduser().resolve()
        self.config = json.loads(json.dumps(config, sort_keys=True))
        if keep_last < 1:
            raise ValueError("keep_last must be positive")
        self.keep_last = keep_last
        if rank_world()[0] == 0:
            self.directory.mkdir(parents=True, exist_ok=True)
        barrier()

    def save(
        self,
        model,
        optimizers,
        *,
        step: int,
        consumed_tokens: int,
        extra: dict | None = None,
    ) -> Path:
        if step < 0 or consumed_tokens < 0:
            raise ValueError("negative training progress")
        rank, world = rank_world()
        token = f"step_{step:09d}"
        target = self.directory / token
        temporary = f".{token}.tmp-{uuid.uuid4().hex}" if rank == 0 else None
        if world > 1:
            values = [temporary]
            dist.broadcast_object_list(values, src=0)
            temporary = values[0]
        staging = self.directory / temporary
        if rank == 0:
            if target.exists():
                raise FileExistsError(f"immutable checkpoint already exists: {target}")
            staging.mkdir()
        barrier()
        device = next(model.parameters()).device
        payload = {
            "optimizers": [optimizer.state_dict() for optimizer in optimizers],
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": (
                torch.cuda.get_rng_state(device) if device.type == "cuda" else None
            ),
            "python_rng": random.getstate(),
        }
        save_file(payload, staging / f"rank_{rank:05d}.pt")
        if rank == 0:
            save_file(
                {"model": model.state_dict(), "extra": extra or {}},
                staging / "model.pt",
            )
        barrier()
        if rank == 0:
            manifest = {
                "format_version": FORMAT_VERSION,
                "world_size": world,
                "config": self.config,
                "step": step,
                "consumed_tokens": consumed_tokens,
                "optimizer_count": len(optimizers),
            }
            manifest_path = staging / "manifest.json"
            with manifest_path.open("w") as stream:
                json.dump(manifest, stream, sort_keys=True, indent=2)
                stream.flush()
                os.fsync(stream.fileno())
            fsync_dir(staging)
            os.replace(staging, target)
            fsync_dir(self.directory)
            marker = self.directory / f".{MARKER}.{uuid.uuid4().hex}"
            with marker.open("w") as stream:
                stream.write(token + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(marker, self.directory / MARKER)
            fsync_dir(self.directory)
            committed = sorted(
                p
                for p in self.directory.glob("step_*")
                if p.is_dir() and re.fullmatch(r"step_\d{9,}", p.name)
            )
            for old in committed[: -self.keep_last]:
                if old != target:
                    shutil.rmtree(old)
        barrier()
        return target

    def load(self, model, optimizers) -> Progress:
        marker = self.directory / MARKER
        if not marker.exists():
            return Progress()
        token = marker.read_text().strip()
        if not re.fullmatch(r"step_\d{9,}", token):
            raise ValueError("invalid checkpoint marker")
        target = self.directory / token
        rank, world = rank_world()
        required = [target / "manifest.json", target / "model.pt"]
        required.extend(target / f"rank_{i:05d}.pt" for i in range(world))
        for path in required:
            if not path.is_file():
                raise RuntimeError(f"incomplete checkpoint: {path}")
        metadata = json.loads(required[0].read_text())
        if metadata["format_version"] != FORMAT_VERSION:
            raise ValueError("unsupported checkpoint format")
        if metadata["world_size"] != world or metadata["config"] != self.config:
            raise ValueError("checkpoint world size or configuration mismatch")
        if metadata["optimizer_count"] != len(optimizers):
            raise ValueError("checkpoint optimizer count mismatch")
        device = next(model.parameters()).device
        common = torch.load(
            target / "model.pt", map_location=device, weights_only=False
        )
        local = torch.load(
            target / f"rank_{rank:05d}.pt", map_location=device, weights_only=False
        )
        model.load_state_dict(common["model"], strict=True)
        for optimizer, state in zip(optimizers, local["optimizers"], strict=True):
            optimizer.load_state_dict(state)
        torch.set_rng_state(local["torch_rng"].cpu())
        if device.type == "cuda":
            torch.cuda.set_rng_state(local["cuda_rng"].cpu(), device)
        random.setstate(local["python_rng"])
        barrier()
        return Progress(metadata["step"], metadata["consumed_tokens"], common["extra"])
