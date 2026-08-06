"""Adapter for Stratified Transformer point encoders."""

from collections.abc import Mapping

import torch
from torch import Tensor, nn


class StratifiedTransformerBackbone(nn.Module):
    """Normalize an external Stratified Transformer to the ToPo-OPD interface.

    The official Stratified Transformer implementations commonly consume
    flattened features, coordinates, and cumulative batch offsets. This adapter
    also supports encoders that directly accept a batched point tensor.

    Args:
        encoder: Instantiated Stratified Transformer encoder.
        encoder_dim: Width returned by ``encoder``.
        output_dim: Desired point feature width. A linear projection is added
            when this differs from ``encoder_dim``.
        input_mode: ``"offset"`` for ``encoder(feat, xyz, offset)`` or
            ``"batched"`` for ``encoder(points)``.
        include_xyz_in_features: Include XYZ in the flattened feature tensor
            used by ``offset`` mode.
    """

    def __init__(
        self,
        encoder: nn.Module,
        encoder_dim: int,
        output_dim: int | None = None,
        input_mode: str = "offset",
        include_xyz_in_features: bool = True,
    ) -> None:
        super().__init__()
        if input_mode not in {"offset", "batched"}:
            raise ValueError("input_mode must be 'offset' or 'batched'")
        self.encoder = encoder
        self.encoder_dim = encoder_dim
        self.output_dim = output_dim or encoder_dim
        self.input_mode = input_mode
        self.include_xyz_in_features = include_xyz_in_features
        self.projection = (
            nn.Identity()
            if self.output_dim == encoder_dim
            else nn.Linear(encoder_dim, self.output_dim)
        )

    @staticmethod
    def _extract_features(output: object) -> Tensor:
        if isinstance(output, Tensor):
            return output
        if isinstance(output, Mapping):
            for key in ("feat", "features", "point_features", "x"):
                value = output.get(key)
                if isinstance(value, Tensor):
                    return value
        if hasattr(output, "feat") and isinstance(output.feat, Tensor):
            return output.feat
        if isinstance(output, (tuple, list)):
            for value in output:
                if isinstance(value, Tensor) and value.ndim in {2, 3}:
                    return value
        raise TypeError("could not extract point features from encoder output")

    def forward(self, points: Tensor) -> Tensor:
        if points.ndim != 3 or points.shape[-1] < 3:
            raise ValueError("points must have shape [batch, points, channels>=3]")
        batch_size, num_points, _ = points.shape

        if self.input_mode == "batched":
            output = self.encoder(points)
        else:
            xyz = points[..., :3].reshape(batch_size * num_points, 3)
            if self.include_xyz_in_features or points.shape[-1] == 3:
                features = points.reshape(batch_size * num_points, -1)
            else:
                features = points[..., 3:].reshape(batch_size * num_points, -1)
            offset = (
                torch.arange(
                    1,
                    batch_size + 1,
                    device=points.device,
                    dtype=torch.long,
                )
                * num_points
            )
            output = self.encoder(features, xyz, offset)

        features = self._extract_features(output)
        if features.ndim == 2:
            if features.shape[0] != batch_size * num_points:
                raise ValueError(
                    "encoder changed the point count; provide an upsampled feature output"
                )
            features = features.reshape(batch_size, num_points, -1)
        elif features.shape[:2] != (batch_size, num_points):
            raise ValueError(
                "batched encoder output must preserve the input point count"
            )
        if features.shape[-1] != self.encoder_dim:
            raise ValueError(
                f"expected encoder width {self.encoder_dim}, got {features.shape[-1]}"
            )
        return self.projection(features)
