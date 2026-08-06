"""On-policy prototype distillation."""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn


@dataclass
class PrototypeDistillationOutput:
    loss: Tensor
    query_prototypes: Tensor
    query_embeddings: Tensor
    support_embeddings: Tensor
    activation_weights: Tensor


class OnPolicyPrototypeDistillation(nn.Module):
    """Align query-side prototypes with calibrated support prototypes.

    Query prototypes are pooled from the model's current coarse predictions,
    making the student target on-policy. The calibrated support prototype acts
    as the teacher in a shared embedding space.
    """

    def __init__(
        self,
        feature_dim: int,
        embedding_dim: int = 128,
        temperature: float = 2.0,
        activation_temperature: float = 1.0,
        detach_teacher: bool = False,
    ) -> None:
        super().__init__()
        if temperature <= 0 or activation_temperature <= 0:
            raise ValueError("temperatures must be positive")
        self.query_head = self._embedding_head(feature_dim, embedding_dim)
        self.support_head = self._embedding_head(feature_dim, embedding_dim)
        self.temperature = temperature
        self.activation_temperature = activation_temperature
        self.detach_teacher = detach_teacher

    @staticmethod
    def _embedding_head(input_dim: int, output_dim: int) -> nn.Sequential:
        return nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, output_dim),
            nn.GELU(),
            nn.Linear(output_dim, output_dim),
        )

    def forward(
        self,
        query_features: Tensor,
        coarse_logits: Tensor,
        calibrated_support: Tensor,
    ) -> PrototypeDistillationOutput:
        if query_features.ndim != 3:
            raise ValueError("query_features must have shape [B, N, D]")
        if coarse_logits.shape[:2] != query_features.shape[:2]:
            raise ValueError("coarse_logits and query_features must align by point")
        if calibrated_support.shape[:2] != (
            query_features.shape[0],
            coarse_logits.shape[-1],
        ):
            raise ValueError("calibrated_support must have shape [B, C, D]")

        class_probability = F.softmax(
            coarse_logits / self.activation_temperature, dim=-1
        )
        activation = class_probability.transpose(1, 2)
        activation = activation / activation.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        query_prototypes = torch.bmm(activation, query_features)

        query_embeddings = self.query_head(query_prototypes)
        support_embeddings = self.support_head(calibrated_support)
        teacher = (
            support_embeddings.detach() if self.detach_teacher else support_embeddings
        )

        temperature = self.temperature
        student_log_probability = F.log_softmax(query_embeddings / temperature, dim=-1)
        teacher_probability = F.softmax(teacher / temperature, dim=-1)
        class_loss = F.kl_div(
            student_log_probability,
            teacher_probability,
            reduction="none",
        ).sum(dim=-1)
        loss = class_loss.mean() * (temperature * temperature)
        return PrototypeDistillationOutput(
            loss=loss,
            query_prototypes=query_prototypes,
            query_embeddings=query_embeddings,
            support_embeddings=support_embeddings,
            activation_weights=activation,
        )
