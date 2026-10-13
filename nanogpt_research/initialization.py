"""Nonzero Track-3 initialization used by Hyperball optimizers."""

import torch


@torch.no_grad()
def initialize_shared_nonzero_parameters(model: torch.nn.Module, seed: int) -> None:
    """Apply the MuonH nonzero initialization with a recipe-local deterministic RNG.

    A dedicated generator makes the result independent of how many temporary models a
    source snapshot constructed before the comparison model. This lets the Muon,
    Muon+WD, and MuonH arms start from bit-identical parameters.
    """
    params = list(model.named_parameters())
    if not params:
        return
    generator = torch.Generator(device=params[0][1].device)
    generator.manual_seed(int(seed) + 0x4D554F4E)
    for name, param in params:
        public_name = name.removeprefix("_orig_mod.")
        value = param.data
        if public_name.endswith("weight"):
            if "embed" in public_name:
                value.normal_(generator=generator)
            else:
                value.normal_(
                    std=0.33**0.5 / value.size(-1) ** 0.5,
                    generator=generator,
                )
        elif public_name.endswith("bias"):
            value.zero_()
        elif public_name.endswith("gains"):
            value.normal_(mean=1, std=0, generator=generator)
        else:
            raise ValueError(f"Uninitialized parameter: {name}")
        if public_name.endswith(".attn.proj.weight"):
            value.mul_(1.25)
        elif public_name.endswith(".mlp.proj.weight"):
            value.mul_(3.0)
        elif public_name.endswith(".mlp.fc.weight"):
            value.mul_(1.5)
