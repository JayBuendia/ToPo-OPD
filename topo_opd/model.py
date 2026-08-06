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
        """Run one batched episode.

        ``support_masks`` has shape ``[B, C, Ns]``. Include a background mask
        as an additional channel when the episode classifier requires one.
        """

        support_features = self.backbone(support_points)
        query_features = self.backbone(query_points)
        initial_prototypes = masked_average_pool(support_features, support_masks)
        topo_output = self.topo(
            initial_prototypes=initial_prototypes,
            support_features=support_features,
            support_xyz=support_points[..., :3],
            support_masks=support_masks,
        )

        normalized_query = F.normalize(query_features, dim=-1)
        normalized_prototypes = F.normalize(topo_output.calibrated, dim=-1)
        logits = self.logit_scale * torch.einsum(
            "bnd,bcd->bnc", normalized_query, normalized_prototypes
        )
        output = {
            "logits": logits,
            "initial_prototypes": initial_prototypes,
            "calibrated_prototypes": topo_output.calibrated,
            "candidate_prototypes": topo_output.candidates,
            "candidate_weights": topo_output.weights,
        }
        if self.training:
            opd_output = self.opd(
                query_features=query_features,
                coarse_logits=logits,
                calibrated_support=topo_output.calibrated,
            )
            output["opd_loss"] = opd_output.loss
            output["query_prototypes"] = opd_output.query_prototypes
        return output
