# research/muonh-track3

Parent: `research/core`.

MuonH, NorMuonH, tangent and preconditioner variants, KLSOAP-H, MuLoCo outer update, and Track-3 schedules.

The published MuonH is `optimizers.track3.muonh.MuonH`; NorMuonH is exported alongside it. Variant module names retain source identifiers, and `SOURCE.track3-aliases.json` maps all 30 source scripts to 17 distinct implementations. Identical numerical kernels live in `track3/common.py`. `factory.optimizers(model, lr)` splits hidden 2D block matrices from auxiliary parameters; its constant LR is a convenience, not a full reproduction recipe. `schedules.track3` exposes the power/cooldown functions separately. MuLoCo is a separate outer controller (`maybe_outer_step`), not a drop-in `Optimizer`; callers must manage its outer state explicitly.

Reference extraction removes automatic `torch.compile` decorators. Compile/performance choices are left to callers. Default Muon direction here uses the original 12-step NS formula; the separate alignment branch retains its own numerical formula rather than conflating the two.

Run from the repository root:

```bash
OMP_NUM_THREADS=1 python -m unittest discover -s tests/research -v
```
