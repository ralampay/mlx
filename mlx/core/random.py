from __future__ import annotations

import random


def capture_random_state() -> dict:
    import numpy as np
    import torch

    numpy_state = np.random.get_state()
    state = {
        "python": random.getstate(),
        "numpy": {
            "bit_generator": numpy_state[0],
            "state": numpy_state[1].tolist(),
            "position": numpy_state[2],
            "has_gauss": numpy_state[3],
            "cached_gaussian": numpy_state[4],
        },
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_random_state(state: dict) -> None:
    import numpy as np
    import torch

    if "python" in state:
        random.setstate(state["python"])
    numpy_state = state.get("numpy")
    if numpy_state:
        np.random.set_state(
            (
                numpy_state["bit_generator"],
                np.asarray(numpy_state["state"], dtype=np.uint32),
                int(numpy_state["position"]),
                int(numpy_state["has_gauss"]),
                float(numpy_state["cached_gaussian"]),
            )
        )
    if "torch" in state:
        torch.set_rng_state(state["torch"].cpu())
    if torch.cuda.is_available() and state.get("torch_cuda"):
        torch.cuda.set_rng_state_all(state["torch_cuda"])

def seed_everything(seed: int | None) -> None:
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


def apply_global_seed(seed: int | None) -> None:
    if seed is None:
        return

    seed_everything(seed)
    from mlx.core.ui import print_info

    print_info(f"Using global random seed={seed}")


__all__ = ["apply_global_seed", "capture_random_state", "restore_random_state", "seed_everything"]
