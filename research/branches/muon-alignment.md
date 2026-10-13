# research/muon-alignment

Parent: `research/geometry`.

Muon / MuonWD / MuonH per-matrix angular alignment.

Depends on scalar geometry, not the published Track-3 implementation: the source direction and precision conventions are distinct. Call `set_target_step` before each aligned update; use quantization-aware refinement when the target applies to stored weights.

Run from the repository root:

```bash
OMP_NUM_THREADS=1 python -m unittest discover -s tests/research -v
```
