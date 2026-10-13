"""Portable training entrypoint, independent of historical experiment scripts.

Use torchrun --standalone --nproc_per_node=8 --module nanogpt_research.train ...
Optimizer branches supply an explicit module:function factory; no plugin discovery
or cluster configuration is involved. Dataset files use the NanoGPT shard format.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import importlib
import json
import os
from pathlib import Path
import random

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from .checkpoint import CheckpointManager, rank_world
from .data import capacity_steps, distributed_data_generator, resolve_shards
from .initialization import initialize_shared_nonzero_parameters
from .model import GPT


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--data-root", type=Path, required=True)
    result.add_argument("--train-pattern", default="fineweb_train_*.bin")
    result.add_argument("--validation-pattern")
    result.add_argument("--checkpoint-dir", type=Path, required=True)
    result.add_argument("--steps", type=int, default=1000)
    result.add_argument("--batch-tokens", type=int, default=524288)
    result.add_argument("--seq-len", type=int, default=1024)
    result.add_argument("--microbatch-sequences", type=int, default=8)
    result.add_argument("--layers", type=int, default=12)
    result.add_argument("--width", type=int, default=768)
    result.add_argument("--vocab-size", type=int, default=50304)
    result.add_argument("--lr", type=float, default=0.001)
    result.add_argument("--seed", type=int, default=0)
    result.add_argument("--save-every", type=int, default=250)
    result.add_argument("--log-every", type=int, default=10)
    result.add_argument("--validate-every", type=int, default=125)
    result.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    result.add_argument(
        "--optimizer-factory", help="module:function returning optimizers(model, lr)"
    )
    return result


def optimizer_mode(optimizers, mode: str) -> None:
    for optimizer in optimizers:
        callback = getattr(optimizer, mode, None)
        if callback is not None:
            callback()


def main(argv=None) -> None:
    args = parser().parse_args(argv)
    for name in (
        "steps",
        "batch_tokens",
        "seq_len",
        "microbatch_sequences",
        "layers",
        "width",
        "vocab_size",
        "save_every",
        "log_every",
        "validate_every",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name} must be positive")
    if args.width % 128:
        raise ValueError(
            "width must be divisible by the model's 128-dimensional attention heads"
        )
    if args.device == "cuda":
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    device = (
        torch.device(args.device, torch.cuda.current_device())
        if args.device == "cuda"
        else torch.device("cpu")
    )
    if "RANK" in os.environ:
        dist.init_process_group("nccl" if device.type == "cuda" else "gloo")
    rank, world = rank_world()
    divisor = world * args.seq_len * args.microbatch_sequences
    if args.batch_tokens % divisor:
        raise ValueError(
            "batch_tokens must be divisible by world_size * seq_len * microbatch_sequences"
        )
    accumulation = args.batch_tokens // divisor
    files = resolve_shards(args.data_root, args.train_pattern)
    if capacity_steps(files, args.batch_tokens) < args.steps:
        raise ValueError(
            "training dataset does not contain enough tokens for the requested horizon"
        )
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    model = GPT(args.vocab_size, args.layers, args.width).to(device)
    initialize_shared_nonzero_parameters(model, args.seed)
    if args.optimizer_factory:
        module, name = args.optimizer_factory.split(":", 1)
        optimizers = list(
            getattr(importlib.import_module(module), name)(model, args.lr)
        )
    else:
        optimizers = [torch.optim.AdamW(model.parameters(), lr=args.lr)]
    if not optimizers:
        raise ValueError("optimizer factory returned no optimizers")
    if any(hasattr(optimizer, "step_func") for optimizer in optimizers):
        raise ValueError(
            "this entrypoint does not implement the loss-aware ScheduleFree+ step_func contract"
        )
    optimizer_mode(optimizers, "train")
    config = {
        k: str(v) if isinstance(v, Path) else v
        for k, v in vars(args).items()
        if k not in {"checkpoint_dir", "save_every", "log_every", "validate_every"}
    }
    # Bind the ordered data stream to its files and sizes, not just a glob string.
    config["shards"] = [(str(path.resolve()), path.stat().st_size) for path in files]
    checkpoint = CheckpointManager(args.checkpoint_dir, config)
    progress = checkpoint.load(model, optimizers)
    if progress.consumed_tokens != progress.step * args.batch_tokens:
        raise ValueError("checkpoint step and consumed-token counts disagree")
    wrapped = (
        DistributedDataParallel(
            model, device_ids=[device.index] if device.type == "cuda" else None
        )
        if world > 1
        else model
    )
    batches = distributed_data_generator(
        args.data_root,
        args.train_pattern,
        args.batch_tokens,
        seq_len=args.seq_len,
        start_step=progress.step,
        target_device=device,
    )
    validation_batch = None
    if args.validation_pattern:
        validation_batch = next(
            distributed_data_generator(
                args.data_root,
                args.validation_pattern,
                args.batch_tokens,
                seq_len=args.seq_len,
                target_device=device,
            )
        )
    if rank == 0:
        print(
            json.dumps(
                {
                    "parameters": sum(p.numel() for p in model.parameters()),
                    "world_size": world,
                    "resume_step": progress.step,
                    "config": config,
                }
            ),
            flush=True,
        )
    for step in range(progress.step, args.steps):
        inputs, targets = next(batches)
        loss_sum = torch.zeros((), device=device, dtype=torch.float64)
        for optimizer in optimizers:
            optimizer.zero_grad(set_to_none=True)
        for micro in range(accumulation):
            begin = micro * args.microbatch_sequences
            end = begin + args.microbatch_sequences
            context = (
                wrapped.no_sync()
                if world > 1 and micro < accumulation - 1
                else nullcontext()
            )
            with context:
                loss, _ = wrapped(inputs[begin:end], targets[begin:end])
                (loss / (args.batch_tokens // world)).backward()
            loss_sum += loss.detach().double()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        for optimizer in optimizers:
            optimizer.step()
        if world > 1:
            dist.all_reduce(loss_sum)
        completed = step + 1
        if rank == 0 and (completed % args.log_every == 0 or completed == args.steps):
            print(
                json.dumps(
                    {
                        "step": completed,
                        "train_loss": (loss_sum / args.batch_tokens).item(),
                    }
                ),
                flush=True,
            )
        if validation_batch is not None and (
            completed % args.validate_every == 0 or completed == args.steps
        ):
            optimizer_mode(optimizers, "eval")
            model.eval()
            with torch.no_grad():
                value = torch.zeros((), device=device, dtype=torch.float64)
                x, y = validation_batch
                for begin in range(0, len(x), args.microbatch_sequences):
                    loss, _ = model(
                        x[begin : begin + args.microbatch_sequences],
                        y[begin : begin + args.microbatch_sequences],
                    )
                    value += loss.double()
                if world > 1:
                    dist.all_reduce(value)
            if rank == 0:
                print(
                    json.dumps(
                        {
                            "step": completed,
                            "validation_loss": (value / args.batch_tokens).item(),
                        }
                    ),
                    flush=True,
                )
            model.train()
            optimizer_mode(optimizers, "train")
        if completed % args.save_every == 0 or completed == args.steps:
            checkpoint.save(
                model,
                optimizers,
                step=completed,
                consumed_tokens=completed * args.batch_tokens,
            )
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
