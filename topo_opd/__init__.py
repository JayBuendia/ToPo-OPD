"""Core components for Topology-Calibrated Prototype Distillation."""

from .backbones import DGCNNBackbone, StratifiedTransformerBackbone
from .model import ToPoOPD
from .modules import (
    OnPolicyPrototypeDistillation,
    PrototypeDistillationOutput,
    TopologyGuidedPrototypeCalibration,
    TopologyPrototypeOutput,
    masked_average_pool,
)

__all__ = [
    "DGCNNBackbone",
    "OnPolicyPrototypeDistillation",
    "PrototypeDistillationOutput",
    "StratifiedTransformerBackbone",
    "ToPoOPD",
    "TopologyGuidedPrototypeCalibration",
    "TopologyPrototypeOutput",
    "masked_average_pool",
]
