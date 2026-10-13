# research/angular-mup

Parent: `research/geometry`.

Pure fixed-angle hyperball target geometry from the angular-muP path.

Only model-independent `fixed_angle_target`, `angle_between` and their numerical helper are included. Megatron monkey-patching, width-transfer job recipes and result analysis remain in archive tags; this branch is not a complete Megatron integration.

Run from the repository root:

```bash
OMP_NUM_THREADS=1 python -m unittest discover -s tests/research -v
```
