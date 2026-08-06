"""Prototype pooling utilities."""

import torch
from torch import Tensor


def masked_average_pool(features: Tensor, masks: Tensor, eps: float = 1e-6) -> Tensor:
    """Pool one prototype per class mask.

    Args:
        features: Point features with shape ``[B, N, D]``.
        masks: Binary or soft masks with shape ``[B, C, N]`` or ``[B, N]``.
        eps: Numerical lower bound for the mask mass.

    Returns:
        Prototypes with shape ``[B, C, D]``.
    """

    if features.ndim != 3:
        raise ValueError("features must have shape [batch, points, channels]")
    if masks.ndim == 2:
        masks = masks.unsqueeze(1)
    if masks.ndim != 3 or masks.shape[0] != features.shape[0]:
        raise ValueError("masks must have shape [batch, classes, points]")
    if masks.shape[-1] != features.shape[1]:
        raise ValueError("masks and features must contain the same number of points")

    weights = masks.to(device=features.device, dtype=features.dtype).clamp_min(0.0)
    mass = weights.sum(dim=-1, keepdim=True)
    return torch.einsum("bcn,bnd->bcd", weights, features) / mass.clamp_min(eps)
