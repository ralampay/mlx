"""PyTorch resource creation and optimization isolated from commands."""

from contextlib import contextmanager

import numpy as np
import torch

from mlx.core.commands import emit
from mlx.core.exceptions import MLXUserError


class TorchTemporalBackend:
    @contextmanager
    def seeded(self, seed, device):
        try:
            target = torch.device(device)
            if target.type not in ("cpu", "cuda"):
                raise MLXUserError(
                    "Temporal training currently supports cpu and cuda devices."
                )
            devices = (
                [
                    target.index
                    if target.index is not None
                    else torch.cuda.current_device()
                ]
                if target.type == "cuda"
                else []
            )
            with torch.random.fork_rng(devices=devices):
                torch.random.default_generator.manual_seed(seed)
                if devices:
                    with torch.cuda.device(devices[0]):
                        torch.cuda.manual_seed(seed)
                yield target
        except (RuntimeError, AssertionError, ValueError) as exc:
            raise MLXUserError(
                f"Temporal model execution failed on '{device}': {exc}. Check device availability and model settings."
            ) from exc

    def fit(self, definition, config, training, validation, request, reporter):
        with self.seeded(request.random_seed, request.device) as device:
            model = definition.build(config).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=request.lr)
            rng = torch.Generator(device=device).manual_seed(request.random_seed)
            ordering = np.random.default_rng(request.random_seed)
            best, best_state, history = float("inf"), None, []
            for epoch in range(request.epochs):
                model.train()
                train_stats = self._epoch(
                    model,
                    training,
                    request,
                    device,
                    rng,
                    ordering.permutation(len(training)),
                    optimizer,
                )
                model.eval()
                # Reset validation noise each epoch for comparable stochastic objectives.
                validation_rng = torch.Generator(device=device).manual_seed(
                    request.random_seed + 1
                )
                with torch.no_grad():
                    val_stats = self._epoch(
                        model,
                        validation,
                        request,
                        device,
                        validation_rng,
                        np.arange(len(validation)),
                    )
                history.append(
                    {"epoch": epoch + 1, "train": train_stats, "val": val_stats}
                )
                if val_stats["loss"] < best:
                    best = val_stats["loss"]
                    best_state = {
                        key: value.detach().cpu().clone()
                        for key, value in model.state_dict().items()
                    }
                emit(
                    reporter,
                    "info",
                    f"Epoch {epoch + 1}: validation loss {val_stats['loss']:.6g}",
                    current=epoch + 1,
                    total=request.epochs,
                    payload=history[-1],
                )
            return best_state, history

    @staticmethod
    def _epoch(model, values, request, device, rng, order, optimizer=None):
        totals = {"loss": 0.0, "reconstruction": 0.0, "kl": 0.0}
        for start in range(0, len(order), request.batch_size):
            batch = torch.as_tensor(
                values[order[start : start + request.batch_size]], device=device
            )
            if optimizer is not None:
                optimizer.zero_grad()
            stats = model.objective(batch, request.kl_weight, rng)
            if any(
                value.ndim != 0 or not torch.isfinite(value).item()
                for value in stats.values()
            ):
                raise MLXUserError(
                    "VAE objective became non-finite. Inspect input ranges or reduce learning rate."
                )
            if optimizer is not None:
                stats["loss"].backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 10.0, error_if_nonfinite=True
                )
                optimizer.step()
            for key in totals:
                totals[key] += float(stats[key].detach()) * len(batch)
        return {key: value / len(values) for key, value in totals.items()}

    def sample(self, definition, checkpoint, request):
        with self.seeded(request.random_seed, request.device) as device:
            model = definition.build(checkpoint["config"]).to(device)
            model.load_state_dict(checkpoint["state_dict"], strict=True)
            model.eval()
            rng = torch.Generator(device=device).manual_seed(request.random_seed)
            with torch.no_grad():
                for start in range(0, request.num_sequences, request.batch_size):
                    values = model.sample(
                        min(request.batch_size, request.num_sequences - start), rng
                    )
                    if not torch.isfinite(values).all():
                        raise MLXUserError(
                            "Model generated non-finite values; inspect training diagnostics."
                        )
                    yield values.cpu().numpy()
