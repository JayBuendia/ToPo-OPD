"""Dependency-light DGCNN backbone for point-level feature extraction."""

from collections.abc import Sequence

import torch
from torch import Tensor, nn


def _knn_indices(features: Tensor, k: int) -> Tensor:
    if features.ndim != 3:
        raise ValueError("features must have shape [batch, points, channels]")
    num_points = features.shape[1]
    if num_points == 0:
        raise ValueError("DGCNN requires at least one point")

    k = min(int(k), num_points)
    squared_norm = (features * features).sum(dim=-1, keepdim=True)
    distance = (
        squared_norm
        + squared_norm.transpose(1, 2)
        - 2.0 * features @ features.transpose(1, 2)
    ).clamp_min_(0.0)
    return distance.topk(k=k, dim=-1, largest=False, sorted=False).indices


def _edge_features(features: Tensor, k: int) -> Tensor:
    batch_size, num_points, channels = features.shape
    indices = _knn_indices(features, k)
    k = indices.shape[-1]

    offsets = torch.arange(batch_size, device=features.device).view(-1, 1, 1)
    offsets = offsets * num_points
    flat_indices = (indices + offsets).reshape(-1)
    neighbors = features.reshape(batch_size * num_points, channels)[flat_indices]
    neighbors = neighbors.reshape(batch_size, num_points, k, channels)
    centers = features.unsqueeze(2).expand(-1, -1, k, -1)
    return torch.cat((neighbors - centers, centers), dim=-1)


class _EdgeConv(nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(input_dim * 2, output_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(output_dim),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

    def forward(self, features: Tensor, k: int) -> Tensor:
        edges = _edge_features(features, k).permute(0, 3, 1, 2).contiguous()
        return self.net(edges).amax(dim=-1).transpose(1, 2).contiguous()


class DGCNNBackbone(nn.Module):
    """Stacked EdgeConv blocks that return one feature vector per input point.

    Args:
        input_dim: Number of channels in each input point. Coordinates should be
            stored in the first three channels.
        edge_dims: Output widths of the EdgeConv blocks.
        output_dim: Width of the final point-level embedding.
        k: Number of dynamic neighbors used by each EdgeConv block.
    """

    def __init__(
        self,
        input_dim: int = 6,
        edge_dims: Sequence[int] = (64, 64, 128),
        output_dim: int = 192,
        k: int = 20,
    ) -> None:
        super().__init__()
        if not edge_dims:
            raise ValueError("edge_dims must contain at least one width")
        if k <= 0:
            raise ValueError("k must be positive")

        dimensions = [input_dim, *edge_dims]
        self.edge_convs = nn.ModuleList(
            _EdgeConv(dimensions[index], dimensions[index + 1])
            for index in range(len(edge_dims))
        )
        merged_dim = sum(edge_dims)
        self.projection = nn.Sequential(
            nn.Conv1d(merged_dim, output_dim, kernel_size=1, bias=False),
            nn.BatchNorm1d(output_dim),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.k = k

    def forward(self, points: Tensor) -> Tensor:
        if points.ndim != 3 or points.shape[-1] != self.input_dim:
            raise ValueError(
                f"points must have shape [batch, points, {self.input_dim}]"
            )

        features = points
        intermediate = []
        for edge_conv in self.edge_convs:
            features = edge_conv(features, self.k)
            intermediate.append(features)

        merged = torch.cat(intermediate, dim=-1).transpose(1, 2).contiguous()
        return self.projection(merged).transpose(1, 2).contiguous()
