# Research code branches

`master` tracks the unmodified upstream tree. Research commits add `nanogpt_research/`, `tests/research/` and this directory; upstream trainers and records are unchanged.

The branches are divided by reusable code and actual import dependencies. Shared code is introduced once in `research/core`. Alignment algorithms share `research/geometry`; a historical mention of another experiment creates no dependency. `research/all` is an integration branch containing all code branches.

| Branch | Parent | Code |
|---|---|---|
| `research/core` | `master` | Shared model, loss, deterministic token stream, initialization and rank-aware checkpoints. |
| `research/geometry` | `research/core` | Scalar angular inverse and stored-weight angle measurement. |
| `research/muonh-track3` | `research/core` | MuonH, NorMuonH, tangent and preconditioner variants, KLSOAP-H, MuLoCo outer update, and Track-3 schedules. |
| `research/adam-hyperball` | `research/core` | AdamH with bias correction and Frobenius-radius retraction. |
| `research/muon-alignment` | `research/geometry` | Muon / MuonWD / MuonH per-matrix angular alignment. |
| `research/adam-alignment` | `research/geometry` | Sharded AdamW / AdamH direction with angular or normalized-step alignment. |
| `research/sgd-alignment` | `research/geometry` | Explicitly configured SGD / SGD-H and stored-weight angle refinement. |
| `research/schedulefree` | `research/core` | ScheduleFree+ reference optimizer and causal gradient-stability scheduler. |
| `research/normalization` | `research/core` | Derf, Internal-Taper calibration and optional Triton kernels. |
| `research/rmnp` | `research/core` | Native-row normalized momentum optimizer and cosine schedule. |
| `research/angular-mup` | `research/geometry` | Pure fixed-angle hyperball target geometry from the angular-muP path. |
| `research/radial-analysis` | `research/geometry` | Pure before/after stored-weight radial coupling calculations. |

## Using the code

Run from the repository root; the package is intentionally small and needs no plugin registry or installation. Core and CPU tests require PyTorch and NumPy. The default model is 12 layers × 768 width, **162,354,816 parameters** with vocab 50,304 and untied embeddings.

```bash
# Inspect flags without starting training.
python -m nanogpt_research.train --help

# Example single-node, eight-GPU invocation on research/muonh-track3.
# Replace paths and choose your own training hyperparameters before running.
torchrun --standalone --nproc_per_node=8 --module nanogpt_research.train \
  --data-root /path/to/fineweb --checkpoint-dir /path/to/checkpoints \
  --steps 1000 --optimizer-factory nanogpt_research.optimizers.track3.factory:optimizers \
  --lr 0.014

OMP_NUM_THREADS=1 python -m unittest discover -s tests/research -v
```

The core entrypoint is a minimal new integration example: its default is constant-LR AdamW, the optional MuonH factory uses the same supplied LR for both groups, and it does not reconstruct a published score or historical configuration. It has no cluster launcher or external monitoring dependency. The model's source formula and the optimizers' source formulas are preserved, but they do not establish end-to-end reproduction equivalence.

The data stream continues across shard boundaries and resumes at `step * batch_tokens`. It differs from older loaders that discarded shard tails. All ranks must use the same ordered dataset and checkpoint configuration. Every rank saves its own optimizer state and RNG; the checkpoint format is new and requires the same world size and configuration. Save/load calls are collective, assume a shared filesystem, and are not elastic recovery. MuLoCo outer-controller state is not supported by the generic checkpoint example. Source-specific algorithms that need targets, calibration callbacks or loss-aware steps require explicit callers; see branch notes.

## Source traceability

`provenance/SOURCE.*.json` records source-relative paths and SHA-256 hashes. Track-3 aliases explain exact implementation deduplication. Reorganization removes import-time training, hardcoded environment paths, source-string execution and automatic compile decorators; it makes runtime configuration explicit. Mathematical variants remain separate when their formulas differ.

Original source snapshots remain in local tags:

- `archive/yhxiao-main-20261013`
- `archive/yhxiao-new-20261013`
- `archive/track3-muonh-power04-3150`
- `archive/track3-optim-20261013`

These archives retain historical source context; active code branches omit historical experiment reports, monitoring/campaign scripts, datasets, weights and run outputs. Loop experiments are outside this code selection. They are not silently imported into a runnable dependency graph.

## Validation limits

CPU tests cover loss gradients, data continuity, optimizer updates, exact checkpoint resume, angular solutions and normalization calibration. Track-3 additionally exercises two Gloo ranks with sharded optimizer state. These checks do not establish eight-GPU NCCL performance, fused-kernel accuracy, training convergence or benchmark scores. No training job is submitted by this repository reorganization.
