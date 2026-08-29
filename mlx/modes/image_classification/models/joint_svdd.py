from __future__ import annotations

import torch
from torch import nn

from mlx.core.exceptions import MLXUserError
from mlx.core.deep_svdd import squared_euclidean_score
from mlx.modes.image_classification.ood.types import JointClassificationOutput


class JointDeepSVDDClassifier(nn.Module):
    def __init__(
        self,
        classifier: nn.Module,
        feature_dim: int,
        svdd_dim: int = 128,
        svdd_hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.classifier = classifier
        self.svdd_head = nn.Sequential(
            nn.Linear(feature_dim, svdd_hidden_dim, bias=False),
            nn.LeakyReLU(inplace=True),
            nn.Linear(svdd_hidden_dim, svdd_dim, bias=False),
        )
        self.register_buffer("svdd_center", torch.zeros(svdd_dim))
        self.register_buffer("svdd_threshold", torch.tensor(float("nan")))

    def forward(self, images: torch.Tensor) -> JointClassificationOutput:
        features = self.classifier.forward_features(images)
        return JointClassificationOutput(
            logits=self.classifier.classify_features(features),
            svdd_embedding=self.svdd_head(features),
        )

    def compute_svdd_score(self, svdd_embedding: torch.Tensor) -> torch.Tensor:
        return squared_euclidean_score(svdd_embedding, self.svdd_center)

    def is_in_distribution(self, svdd_embedding: torch.Tensor) -> torch.Tensor:
        if not torch.isfinite(self.svdd_threshold):
            raise MLXUserError(
                "The checkpoint contains a Deep SVDD model, but no calibrated rejection threshold. "
                "Run threshold calibration or use a final deployment checkpoint."
            )
        return self.compute_svdd_score(svdd_embedding) <= self.svdd_threshold
