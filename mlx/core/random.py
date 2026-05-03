from __future__ import annotations

import random

from mlx.core.ui import print_info


def apply_global_seed(seed: int | None) -> None:
    if seed is None:
        return

    random.seed(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass

    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print_info(f"Using global random seed={seed}")
