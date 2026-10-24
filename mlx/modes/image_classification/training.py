"""PyTorch epoch execution; workflow ordering and persistence remain in train.py."""
from __future__ import annotations

from dataclasses import dataclass

from mlx.core.losses import validate_scalar_loss
from mlx.modes.image_classification.ood.deep_svdd import compute_svdd_loss


@dataclass(frozen=True)
class ClassificationEpochResult:
    loss: float
    classification_loss: float = 0.0
    svdd_loss: float = 0.0


class TrainOneShotEpoch:
    def __init__(self, model, loader, criterion, optimizer, device):
        self.model = model
        self.loader = loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device

    def execute(self) -> ClassificationEpochResult:
        self.model.train()
        running_loss = 0.0
        for first, second, label in self.loader:
            first, second, label = first.to(self.device), second.to(self.device), label.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(first, second)
            loss = self.criterion(output, label.unsqueeze(1))
            validate_scalar_loss(loss, training=True)
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()
        return ClassificationEpochResult(running_loss / len(self.loader))


class TrainClassificationEpoch:
    def __init__(self, model, loader, criterion, optimizer, device, *,
                 joint_svdd=False, svdd_weight=0.05, include_svdd=True):
        self.model = model
        self.loader = loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.joint_svdd = joint_svdd
        self.svdd_weight = svdd_weight
        self.include_svdd = include_svdd

    def execute(self) -> ClassificationEpochResult:
        self.model.train()
        running_loss = 0.0
        classification_total = 0.0
        svdd_total = 0.0
        for images, targets in self.loader:
            images, targets = images.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(images)
            if self.joint_svdd:
                classification_loss = self.criterion(output.logits, targets)
                validate_scalar_loss(classification_loss, training=True)
                svdd_loss = compute_svdd_loss(self.model, output.svdd_embedding)
                loss = classification_loss
                if self.include_svdd:
                    loss = loss + self.svdd_weight * svdd_loss
                classification_total += classification_loss.item()
                svdd_total += svdd_loss.item()
            else:
                loss = self.criterion(output, targets)
            validate_scalar_loss(loss, training=True)
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()
        batches = len(self.loader)
        return ClassificationEpochResult(
            running_loss / batches, classification_total / batches, svdd_total / batches,
        )
