# research/sgd-alignment

Parent: `research/geometry`.

Explicitly configured SGD / SGD-H and stored-weight angle refinement.

`NormHSGD(named_parameters, family="sgd" | "sgdh", lr=..., momentum=..., dampening=..., nesterov=..., weight_decay=..., aligned=..., alignment_mode="h" | "elr" | "angle")` takes explicit hyperparameters. Supply `targets` by parameter name before each aligned step. The SGD branch applies decoupled decay, unlike `torch.optim.SGD` with nonzero weight_decay; SGD-H uses zero decay. The exact reference solver and numerical refinement remain separate modules.

Run from the repository root:

```bash
OMP_NUM_THREADS=1 python -m unittest discover -s tests/research -v
```
