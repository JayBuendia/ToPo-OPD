"""Composable ToPo-OPD model for episodic point-cloud segmentation."""

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .modules import (
    OnPolicyPrototypeDistillation,
    TopologyGuidedPrototypeCalibration,
    masked_average_pool,
)


class ToPoOPD(nn.Module):
    """Combine a point backbone, ToPo calibration, and training-only OPD."""

    def __init__(
        self,
        backbone: nn.Module,
        feature_dim: int,
        num_candidates: int = 4,
        topology_dim: int = 128,
        topology_neighbors: int = 16,
        max_topology_points: int | None = 512,
        opd_embedding_dim: int = 128,
        opd_temperature: float = 2.0,
        logit_scale: float = 10.0,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.topo = TopologyGuidedPrototypeCalibration(
            feature_dim=feature_dim,
            num_candidates=num_candidates,
            topology_dim=topology_dim,
            k_neighbors=topology_neighbors,
            max_topology_points=max_topology_points,
        )
        self.opd = OnPolicyPrototypeDistillation(
            feature_dim=feature_dim,
            embedding_dim=opd_embedding_dim,
            temperature=opd_temperature,
        )
        self.logit_scale = logit_scale

    def forward(
        self,
        support_points: Tensor,
        support_masks: Tensor,
        query_points: Tensor,
    ) -> dict[str, Tensor]:
        """Run an episode; see README for explicit support-cloud semantics.

        Legacy: points [B,N,D], masks [B,C,N]. Explicit clouds:
        points [B,S,N,D], masks [B,S,C,N]. S is the number of support
        clouds, not necessarily the per-class shot count.
        """
        explicit_shots = support_points.ndim == 4
        if not explicit_shots:
            if support_points.ndim != 3:
                raise ValueError("support_points must have shape [B,N,D] or [B,S,N,D]")
            if support_masks.ndim == 2:
                support_masks = support_masks.unsqueeze(1)
            support_points = support_points.unsqueeze(1)
            support_masks = support_masks.unsqueeze(1)
        b, shots, n, d = support_points.shape
        if min(b, shots, n) < 1 or d < 3:
            raise ValueError("support requires nonempty clouds and at least XYZ channels")
        if (support_masks.ndim != 4 or support_masks.shape[:2] != (b, shots)
                or support_masks.shape[-1] != n or support_masks.shape[2] < 1):
            raise ValueError("support_masks must align as [B,S,C,N]")
        if (query_points.ndim != 3 or query_points.shape[0] != b
                or query_points.shape[-1] != d or query_points.shape[1] < 1):
            raise ValueError("query_points must align as [B,Nq,D]")
        if not torch.isfinite(support_masks).all() or (support_masks < 0).any():
            raise ValueError("support_masks must be finite and nonnegative")
        valid = support_masks.sum(-1) > 0
        if explicit_shots and not valid.any(dim=1).all():
            raise ValueError("each episode class needs at least one nonempty support mask")

        # Flatten clouds into the batch axis, NEVER into the point axis.
        # BatchNorm may share batch statistics in training, but kNN is cloud-local.
        flat_points = support_points.reshape(b * shots, n, d)
        flat_masks = support_masks.reshape(b * shots, support_masks.shape[2], n)
        support_features = self.backbone(flat_points)
        query_features = self.backbone(query_points)
        shot_initial = masked_average_pool(support_features, flat_masks)
        topo_output = self.topo(
            initial_prototypes=shot_initial,
            support_features=support_features,
            support_xyz=flat_points[..., :3],
            support_masks=flat_masks,
        )
        def merge(value: Tensor) -> Tensor:
            value = value.reshape(b, shots, *value.shape[1:])
            weights = (valid if explicit_shots else torch.ones_like(valid)).to(value.dtype)
            while weights.ndim < value.ndim:
                weights = weights.unsqueeze(-1)
            return (value * weights).sum(1) / weights.sum(1).clamp_min(1)

        initial_prototypes = merge(shot_initial)
        calibrated = merge(topo_output.calibrated)
        normalized_query = F.normalize(query_features, dim=-1)
        normalized_prototypes = F.normalize(calibrated, dim=-1)
        logits = self.logit_scale * torch.einsum(
            "bnd,bcd->bnc", normalized_query, normalized_prototypes
        )
        output = {
            "logits": logits,
            "initial_prototypes": initial_prototypes,
            "calibrated_prototypes": calibrated,
            "candidate_prototypes": merge(topo_output.candidates),
            "candidate_weights": merge(topo_output.weights),
        }
        if explicit_shots:
            output["shot_initial_prototypes"] = shot_initial.reshape(b, shots, *shot_initial.shape[1:])
            output["shot_calibrated_prototypes"] = topo_output.calibrated.reshape(b, shots, *topo_output.calibrated.shape[1:])
            output["shot_candidate_prototypes"] = topo_output.candidates.reshape(b, shots, *topo_output.candidates.shape[1:])
            output["shot_candidate_weights"] = topo_output.weights.reshape(b, shots, *topo_output.weights.shape[1:])
            output["shot_valid"] = valid
        if self.training:
            opd_output = self.opd(
                query_features=query_features,
                coarse_logits=logits,
                calibrated_support=calibrated,
            )
            output["opd_loss"] = opd_output.loss
            output["query_prototypes"] = opd_output.query_prototypes
        return output
