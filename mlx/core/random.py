from __future__ import annotations

from mlx.core.ui import print_info


def apply_torch_seed(seed: int | None) -> None:
    if seed is None:
        return

    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print_info(f"Using PyTorch random seed={seed}")
