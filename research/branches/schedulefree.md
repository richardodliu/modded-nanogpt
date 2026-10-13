# research/schedulefree

Parent: `research/core`.

ScheduleFree+ reference optimizer and causal gradient-stability scheduler.

Use `AdamCScheduleFreePlusPaper.train()`, backpropagate the loss, then call `step_func(loss_value)`. Use `.eval()` for evaluation and restore `.train()` before training. The generic core trainer rejects this loss-aware optimizer until an appropriate caller is supplied. `GradientStabilityController` is independently usable and has `state_dict` / `load_state_dict`. The accompanying Meta license applies to the reference optimizer.

Run from the repository root:

```bash
OMP_NUM_THREADS=1 python -m unittest discover -s tests/research -v
```
