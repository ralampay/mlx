from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class JointClassificationOutput:
    logits: torch.Tensor
    svdd_embedding: torch.Tensor
