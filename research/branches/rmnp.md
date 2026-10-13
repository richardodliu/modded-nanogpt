# research/rmnp

Parent: `research/core`.

Native-row normalized momentum optimizer and cosine schedule.

RMNP is adapted from Dominator-Index/RMNP, source commit a662ebc (Apache-2.0). Native output-row normalization and aspect scaling are preserved. Geometry logging and source-string trainer rewriting are removed.

Run from the repository root:

```bash
OMP_NUM_THREADS=1 python -m unittest discover -s tests/research -v
```
