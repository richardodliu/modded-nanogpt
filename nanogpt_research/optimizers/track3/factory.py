"""Partition hidden matrices and auxiliary parameters without overlapping ownership."""

import torch
from .muonh import MuonH


def optimizers(model, lr):
    matrices = [p for p in model.blocks.parameters() if p.ndim == 2]
    ids = {id(p) for p in matrices}
    auxiliary = [p for p in model.parameters() if id(p) not in ids]
    return [
        torch.optim.AdamW(auxiliary, lr=lr, betas=(0.8, 0.95), eps=1e-10),
        MuonH(matrices, lr=lr),
    ]
