import torch
import torch.nn as nn
from abc import ABC, abstractmethod

class BaseICOneShotModel(ABC, nn.Module):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, x1, x2):
        """ Forward pass comparing two input tensors """
        pass

    def predict(self, x1, x2):
        """Prediction wrapper in eval mode."""
        self.eval()
        with torch.no_grad():
            return self.forward(x1, x2)
