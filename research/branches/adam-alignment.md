# research/adam-alignment

Parent: `research/geometry`.

Sharded AdamW / AdamH direction with angular or normalized-step alignment.

This optimizer has its own sharded state and alignment update. It depends on the scalar geometry branch, not on the standalone AdamH class; the two source algorithms differ in their state/update contract. Call `set_target_step` before each non-reference step.

Run from the repository root:

```bash
OMP_NUM_THREADS=1 python -m unittest discover -s tests/research -v
```
