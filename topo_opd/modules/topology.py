"""Topology-guided support prototype calibration."""

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass
class TopologyPrototypeOutput:
    calibrated: Tensor
    candidates: Tensor
    weights: Tensor
    topology: Tensor


def _gather_neighbors(values: Tensor, indices: Tensor) -> Tensor:
    batch_size, num_points, channels = values.shape
    k = indices.shape[-1]
    offsets = torch.arange(batch_size, device=values.device).view(-1, 1, 1)
    flat_indices = (indices + offsets * num_points).reshape(-1)
    gathered = values.reshape(batch_size * num_points, channels)[flat_indices]
    return gathered.reshape(batch_size, num_points, k, channels)


class _MaskedTopologyEncoder(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        topology_dim: int,
        k_neighbors: int,
        max_points: int | None,
    ) -> None:
        super().__init__()
        if k_neighbors <= 0:
            raise ValueError("k_neighbors must be positive")
        if max_points is not None and max_points <= 0:
            raise ValueError("max_points must be positive or None")
        edge_dim = feature_dim * 2 + 4
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_dim, topology_dim),
            nn.GELU(),
            nn.Linear(topology_dim, topology_dim),
        )
        self.point_encoder = nn.Sequential(
            nn.LayerNorm(feature_dim + topology_dim * 2),
            nn.Linear(feature_dim + topology_dim * 2, topology_dim),
            nn.GELU(),
            nn.Linear(topology_dim, topology_dim),
        )
        self.k_neighbors = k_neighbors
        self.max_points = max_points

    def _sample_foreground(
        self,
        features: Tensor,
        xyz: Tensor,
        masks: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        if self.max_points is None or features.shape[1] <= self.max_points:
            return features, xyz, masks

        sample_size = self.max_points
        sampled_indices = torch.zeros(
            features.shape[0],
            sample_size,
            device=features.device,
            dtype=torch.long,
        )
        sampled_masks = torch.zeros(
            features.shape[0],
            sample_size,
            device=features.device,
            dtype=torch.bool,
        )
        for batch_index in range(features.shape[0]):
            foreground = masks[batch_index].nonzero(as_tuple=False).flatten()
            count = min(foreground.numel(), sample_size)
            if count == 0:
                continue
            positions = (
                torch.linspace(
                    0,
                    foreground.numel() - 1,
                    steps=count,
                    device=features.device,
                )
                .round()
                .long()
            )
            sampled_indices[batch_index, :count] = foreground[positions]
            sampled_masks[batch_index, :count] = True

        feature_indices = sampled_indices.unsqueeze(-1).expand(
            -1, -1, features.shape[-1]
        )
        xyz_indices = sampled_indices.unsqueeze(-1).expand(-1, -1, 3)
        return (
            torch.gather(features, 1, feature_indices),
            torch.gather(xyz, 1, xyz_indices),
            sampled_masks,
        )

    def forward(self, features: Tensor, xyz: Tensor, masks: Tensor) -> Tensor:
        batch_size, num_classes, num_points = masks.shape
        feature_dim = features.shape[-1]
        flat_size = batch_size * num_classes

        flat_features = features[:, None].expand(-1, num_classes, -1, -1)
        flat_features = flat_features.reshape(flat_size, num_points, feature_dim)
        flat_xyz = xyz[:, None].expand(-1, num_classes, -1, -1)
        flat_xyz = flat_xyz.reshape(flat_size, num_points, 3)
        flat_mask = masks.reshape(flat_size, num_points).bool()
        flat_features, flat_xyz, flat_mask = self._sample_foreground(
            flat_features,
            flat_xyz,
            flat_mask,
        )
        num_points = flat_features.shape[1]

        distance = torch.cdist(flat_xyz, flat_xyz)
        valid_pairs = flat_mask.unsqueeze(1) & flat_mask.unsqueeze(2)
        eye = torch.eye(num_points, device=xyz.device, dtype=torch.bool).unsqueeze(0)
        distance = distance.masked_fill(~valid_pairs | eye, float("inf"))

        k = min(self.k_neighbors, num_points)
        neighbor_distance, indices = distance.topk(k=k, largest=False, sorted=False)
        neighbor_valid = neighbor_distance.isfinite()
        neighbor_features = _gather_neighbors(flat_features, indices)
        neighbor_xyz = _gather_neighbors(flat_xyz, indices)
        centers = flat_features.unsqueeze(2).expand(-1, -1, k, -1)
        center_xyz = flat_xyz.unsqueeze(2).expand(-1, -1, k, -1)
        delta_xyz = neighbor_xyz - center_xyz
        edge_input = torch.cat(
            (
                neighbor_features - centers,
                centers,
                delta_xyz,
                delta_xyz.norm(dim=-1, keepdim=True),
            ),
            dim=-1,
        )
        encoded_edges = self.edge_encoder(edge_input)

        valid = neighbor_valid.unsqueeze(-1)
        edge_mean = (encoded_edges * valid).sum(dim=2)
        edge_mean = edge_mean / valid.sum(dim=2).clamp_min(1)
        edge_max = encoded_edges.masked_fill(~valid, float("-inf")).amax(dim=2)
        edge_max = torch.where(
            torch.isfinite(edge_max), edge_max, torch.zeros_like(edge_max)
        )

        point_topology = self.point_encoder(
            torch.cat((flat_features, edge_mean, edge_max), dim=-1)
        )
        point_valid = flat_mask.unsqueeze(-1)
        topology_mean = (point_topology * point_valid).sum(dim=1)
        topology_mean = topology_mean / point_valid.sum(dim=1).clamp_min(1)
        topology_max = point_topology.masked_fill(~point_valid, float("-inf")).amax(
            dim=1
        )
        topology_max = torch.where(
            torch.isfinite(topology_max), topology_max, torch.zeros_like(topology_max)
        )
        topology = 0.5 * (topology_mean + topology_max)
        return topology.reshape(batch_size, num_classes, -1)


class TopologyGuidedPrototypeCalibration(nn.Module):
    """Expand and aggregate support prototypes using masked point topology."""

    def __init__(
        self,
        feature_dim: int,
        num_candidates: int = 4,
        topology_dim: int = 128,
        k_neighbors: int = 16,
        max_topology_points: int | None = 512,
    ) -> None:
        super().__init__()
        if num_candidates <= 0:
            raise ValueError("num_candidates must be positive")
        self.num_candidates = num_candidates
        self.feature_dim = feature_dim
        self.topology_encoder = _MaskedTopologyEncoder(
            feature_dim=feature_dim,
            topology_dim=topology_dim,
            k_neighbors=k_neighbors,
            max_points=max_topology_points,
        )
        self.candidate_expansion = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, feature_dim),
            nn.GELU(),
            nn.Linear(feature_dim, num_candidates * feature_dim),
        )
        self.scorer = nn.Sequential(
            nn.LayerNorm(feature_dim + topology_dim),
            nn.Linear(feature_dim + topology_dim, topology_dim),
            nn.GELU(),
            nn.Linear(topology_dim, 1),
        )
        nn.init.normal_(self.candidate_expansion[-1].weight, std=0.02)
        nn.init.zeros_(self.candidate_expansion[-1].bias)

    def forward(
        self,
        initial_prototypes: Tensor,
        support_features: Tensor,
        support_xyz: Tensor,
        support_masks: Tensor,
    ) -> TopologyPrototypeOutput:
        if support_masks.ndim == 2:
            support_masks = support_masks.unsqueeze(1)
        if initial_prototypes.ndim != 3:
            raise ValueError("initial_prototypes must have shape [B, C, D]")
        if support_features.ndim != 3 or support_features.shape[-1] != self.feature_dim:
            raise ValueError("support_features have an incompatible shape")
        if support_xyz.shape != (*support_features.shape[:2], 3):
            raise ValueError("support_xyz must have shape [B, N, 3]")
        expected_mask_shape = (
            support_features.shape[0],
            initial_prototypes.shape[1],
            support_features.shape[1],
        )
        if support_masks.shape != expected_mask_shape:
            raise ValueError(f"support_masks must have shape {expected_mask_shape}")

        masks = support_masks.to(device=support_features.device).bool()
        topology = self.topology_encoder(support_features, support_xyz, masks)
        offsets = self.candidate_expansion(initial_prototypes)
        offsets = offsets.reshape(
            *initial_prototypes.shape[:2], self.num_candidates, self.feature_dim
        )
        candidates = initial_prototypes.unsqueeze(2) + offsets
        expanded_topology = topology.unsqueeze(2).expand(
            -1, -1, self.num_candidates, -1
        )
        scores = self.scorer(torch.cat((candidates, expanded_topology), dim=-1))
        weights = scores.squeeze(-1).softmax(dim=-1)
        calibrated = (candidates * weights.unsqueeze(-1)).sum(dim=2)
        return TopologyPrototypeOutput(
            calibrated=calibrated,
            candidates=candidates,
            weights=weights,
            topology=topology,
        )
