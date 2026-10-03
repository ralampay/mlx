"""Linear, tied, orthonormal-row projection with training-only initialization."""
import torch
from torch import nn

from mlx.core.exceptions import MLXUserError


class OrthogonalTiedAutoencoder(nn.Module):
    def __init__(self, input_dimensions, bottleneck_dimensions):
        super().__init__()
        if not 1 <= bottleneck_dimensions < input_dimensions:
            raise MLXUserError("Orthogonal tied dimensions must satisfy 1 <= bottleneck < input.")
        self.input_dimensions = input_dimensions
        self.bottleneck_dimensions = bottleneck_dimensions
        self.minimum_training_rows = bottleneck_dimensions
        self.weight = nn.Parameter(torch.eye(input_dimensions)[:bottleneck_dimensions].clone())
        self.initialization_metadata = {}

    def encode(self, values):
        return values @ self.weight.T

    def decode(self, latent):
        return latent @ self.weight

    def forward(self, values):
        return self.decode(self.encode(values))

    @torch.no_grad()
    def initialize_from_training_values(self, values):
        if values.ndim != 2 or values.shape[1] != self.input_dimensions or not torch.isfinite(values).all():
            raise MLXUserError("SVD initialization requires finite training vectors of the input width.")
        if len(values) < self.bottleneck_dimensions:
            raise MLXUserError("SVD initialization requires at least as many training rows as latent dimensions.")
        try:
            _, _, vh = torch.linalg.svd(values.to(dtype=torch.float64), full_matrices=False)
            self.weight.copy_(vh[:self.bottleneck_dimensions].to(self.weight))
            self.project_parameters_()
        except RuntimeError as exc:
            raise MLXUserError(f"Unable to initialize the orthogonal encoder from training SVD: {exc}") from exc
        self.initialization_metadata = {"method": "uncentered-svd", "training_rows": len(values),
                                        "centered": False, "constraint": "signed-reduced-qr"}

    @torch.no_grad()
    def project_parameters_(self):
        q, r = torch.linalg.qr(self.weight.T, mode="reduced")
        signs = torch.where(r.diagonal() < 0, -1., 1.)
        self.weight.copy_((q * signs).T)

    @torch.no_grad()
    def training_diagnostics(self):
        identity = torch.eye(self.bottleneck_dimensions, device=self.weight.device, dtype=self.weight.dtype)
        return {"orthogonality_error": torch.linalg.matrix_norm(self.weight @ self.weight.T - identity)}


class OrthogonalTiedDefinition:
    name = "orthogonal-tied"
    description = "Bias-free tied linear encoder with orthonormal rows and training-only SVD initialization."

    def build(self, config):
        unknown = set(config) - {"input_dimensions", "bottleneck_dimensions", "hidden_dimensions"}
        if unknown:
            raise MLXUserError(f"Unsupported orthogonal-tied options: {', '.join(sorted(unknown))}.")
        return OrthogonalTiedAutoencoder(int(config["input_dimensions"]), int(config["bottleneck_dimensions"]))
