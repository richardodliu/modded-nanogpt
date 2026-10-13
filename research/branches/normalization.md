# research/normalization

Parent: `research/core`.

Derf, Internal-Taper calibration and optional Triton kernels.

Use `replace_normalization(model, case="derf")` or a `taper_g*` case **before constructing optimizers**. For Taper call `prepare_step(model, update, warmup=..., case=...)` before forward and `finish_step(model)` once after the training update; update is one-based and the case must remain the same. Calibration buffers are included in model state. Reference PyTorch kernels are the default; the optional fused backend requires compatible CUDA/Triton/PyTorch and has not been GPU-validated in this reorganization.

Run from the repository root:

```bash
OMP_NUM_THREADS=1 python -m unittest discover -s tests/research -v
```
