"""Prototype calibration and distillation modules."""

from .opd import OnPolicyPrototypeDistillation, PrototypeDistillationOutput
from .prototypes import masked_average_pool
from .topology import TopologyGuidedPrototypeCalibration, TopologyPrototypeOutput

__all__ = [
    "OnPolicyPrototypeDistillation",
    "PrototypeDistillationOutput",
    "TopologyGuidedPrototypeCalibration",
    "TopologyPrototypeOutput",
    "masked_average_pool",
]
