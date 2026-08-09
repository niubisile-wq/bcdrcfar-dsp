"""Typed inputs and provenance records for BC-DRCFAR experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


VALID_ROLES = {"target", "guard", "clutter", "calibration"}
VALID_SPLITS = {"development", "calibration", "locked_test"}


@dataclass(frozen=True)
class AcquisitionRecord:
    acquisition_id: str
    session_id: str
    split: str
    role: str
    source: str
    file_sha256: str
    frequency_hz: float
    polarization: str
    range_bin: str
    start_index: int
    stop_index: int

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "AcquisitionRecord":
        record = cls(
            acquisition_id=str(value["acquisition_id"]),
            session_id=str(value["session_id"]),
            split=str(value["split"]),
            role=str(value["role"]),
            source=str(value["source"]),
            file_sha256=str(value["file_sha256"]),
            frequency_hz=float(value["frequency_hz"]),
            polarization=str(value["polarization"]),
            range_bin=str(value["range_bin"]),
            start_index=int(value["start_index"]),
            stop_index=int(value["stop_index"]),
        )
        record.validate()
        return record

    def validate(self) -> None:
        if not self.acquisition_id or not self.session_id or not self.source:
            raise ValueError("acquisition, session, and source identifiers are required")
        if self.split not in VALID_SPLITS:
            raise ValueError(f"unknown split: {self.split}")
        if self.role not in VALID_ROLES:
            raise ValueError(f"unknown range-bin role: {self.role}")
        if len(self.file_sha256) != 64 or any(character not in "0123456789abcdef" for character in self.file_sha256.lower()):
            raise ValueError("file_sha256 must be a 64-character hexadecimal digest")
        if not np.isfinite(self.frequency_hz) or self.frequency_hz <= 0:
            raise ValueError("frequency_hz must be finite and positive")
        if self.start_index < 0 or self.stop_index <= self.start_index:
            raise ValueError("sample interval must be non-empty and increasing")


@dataclass(frozen=True)
class DetectionBatch:
    """A batch of complex CUT and reference windows.

    Arrays use a final I/Q axis of length two. Reference shape is B x K x L x 2,
    CUT shape is B x L x 2, and reference_mask shape is B x K.
    """

    cut_iq: np.ndarray
    reference_iq: np.ndarray
    reference_mask: np.ndarray
    target_pfa: np.ndarray

    def validate(self) -> None:
        cut = np.asarray(self.cut_iq)
        reference = np.asarray(self.reference_iq)
        mask = np.asarray(self.reference_mask)
        pfa = np.asarray(self.target_pfa)
        if cut.ndim != 3 or cut.shape[-1] != 2:
            raise ValueError("cut_iq must have shape B x L x 2")
        if reference.ndim != 4 or reference.shape[-1] != 2:
            raise ValueError("reference_iq must have shape B x K x L x 2")
        if reference.shape[0] != cut.shape[0] or reference.shape[2] != cut.shape[1]:
            raise ValueError("CUT and reference batch/slow-time dimensions must match")
        if mask.shape != reference.shape[:2]:
            raise ValueError("reference_mask must have shape B x K")
        if pfa.shape not in {(cut.shape[0],), (cut.shape[0], 1)}:
            raise ValueError("target_pfa must contain one value per batch item")
        if not np.isfinite(cut).all() or not np.isfinite(reference).all():
            raise ValueError("I/Q inputs must be finite")
        if not np.asarray(mask, dtype=bool).any(axis=1).all():
            raise ValueError("every batch item requires at least one reference cell")
        if not np.isfinite(pfa).all() or np.any((pfa <= 0) | (pfa >= 1)):
            raise ValueError("target_pfa must be strictly between zero and one")


@dataclass(frozen=True)
class DetectionOutput:
    normalized_score: np.ndarray
    normalized_threshold: np.ndarray
    absolute_threshold: np.ndarray
    uncertainty: np.ndarray
    decision: np.ndarray
    anchor_weights: np.ndarray

