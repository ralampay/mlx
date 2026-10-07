"""Tensor-output traversal shared by adapter verification and export."""
import torch

def _flatten_tensors(value):
    if isinstance(value, torch.Tensor):
        return [value]
    if isinstance(value, dict):
        return [tensor for item in value.values() for tensor in _flatten_tensors(item)]
    if isinstance(value, (tuple, list)):
        return [tensor for item in value for tensor in _flatten_tensors(item)]
    return []


