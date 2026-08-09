"""Fail-closed calibration and locked-test opening guards for BC-DRCFAR."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_calibration_opening(
    development_gate: Mapping[str, Any],
    freeze_manifest: Mapping[str, Any],
    *,
    protocol_path: str | Path,
    model_path: str | Path,
) -> dict[str, Any]:
    errors: list[str] = []
    if development_gate.get("status") != "BCDRCFAR_DEVELOPMENT_GO":
        errors.append("development gate is not GO")
    if freeze_manifest.get("status") != "BCDRCFAR_MODEL_FROZEN_BEFORE_CALIBRATION":
        errors.append("model is not frozen before calibration")
    if development_gate.get("calibration_opened") is not False or development_gate.get("locked_opened") is not False:
        errors.append("development gate records an earlier opening")
    if freeze_manifest.get("protocol_sha256") != sha256(protocol_path):
        errors.append("protocol hash mismatch")
    if freeze_manifest.get("model_sha256") != sha256(model_path):
        errors.append("model hash mismatch")
    for key in ("training_code_sha256", "model_code_sha256", "manifest_sha256", "development_predictions_sha256"):
        if not freeze_manifest.get(key):
            errors.append(f"missing frozen hash: {key}")
    if errors:
        raise ValueError("Calibration cannot be opened:\n- " + "\n- ".join(errors))
    payload = {
        "status": "BCDRCFAR_CALIBRATION_OPENABLE",
        "protocol_sha256": sha256(protocol_path),
        "model_sha256": sha256(model_path),
    }
    payload["opening_guard_sha256"] = hashlib.sha256(
        (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
    ).hexdigest()
    return payload


def validate_locked_opening(
    calibration_gate: Mapping[str, Any],
    calibration_freeze: Mapping[str, Any],
    *,
    protocol_path: str | Path,
    model_path: str | Path,
) -> dict[str, Any]:
    errors: list[str] = []
    if calibration_gate.get("status") != "BCDRCFAR_CALIBRATION_GO":
        errors.append("calibration gate is not GO")
    if calibration_freeze.get("status") != "BCDRCFAR_CALIBRATION_FROZEN_BEFORE_LOCKED":
        errors.append("calibration is not frozen before locked opening")
    if calibration_gate.get("locked_opened") is not False:
        errors.append("calibration gate records an earlier locked opening")
    if calibration_freeze.get("protocol_sha256") != sha256(protocol_path):
        errors.append("protocol hash mismatch")
    if calibration_freeze.get("model_sha256") != sha256(model_path):
        errors.append("model hash mismatch")
    for key in ("calibration_predictions_sha256", "threshold_corrections_sha256", "calibration_code_sha256"):
        if not calibration_freeze.get(key):
            errors.append(f"missing calibration hash: {key}")
    if errors:
        raise ValueError("Locked test cannot be opened:\n- " + "\n- ".join(errors))
    payload = {
        "status": "BCDRCFAR_LOCKED_OPENABLE",
        "protocol_sha256": sha256(protocol_path),
        "model_sha256": sha256(model_path),
    }
    payload["opening_guard_sha256"] = hashlib.sha256(
        (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
    ).hexdigest()
    return payload


__all__ = ["sha256", "validate_calibration_opening", "validate_locked_opening"]
