"""BC-DRCFAR: background-conditioned, distributionally robust CFAR detection."""

from .model import BCDRCFAR, count_trainable_parameters
from .opening import validate_calibration_opening, validate_locked_opening
from .baselines import BASELINE_NAMES, classical_cfar_outputs
from .evaluation import DecisionCounter, condition_rows, evaluate_synthetic_condition
from .protocol import build_w1d_manifest, load_protocol, validate_protocol
from .types import AcquisitionRecord, DetectionBatch, DetectionOutput

__all__ = [
    "AcquisitionRecord",
    "BASELINE_NAMES",
    "BCDRCFAR",
    "DecisionCounter",
    "DetectionBatch",
    "DetectionOutput",
    "build_w1d_manifest",
    "classical_cfar_outputs",
    "condition_rows",
    "count_trainable_parameters",
    "evaluate_synthetic_condition",
    "load_protocol",
    "validate_calibration_opening",
    "validate_locked_opening",
    "validate_protocol",
]
