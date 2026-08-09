"""Fail-closed validation for one-time external confirmation opening."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_confirmation_design(
    design: dict[str, Any], *, root: str | Path
) -> dict[str, Any]:
    """Validate a frozen design without parsing any locked data payload."""

    root = Path(root)
    errors: list[str] = []
    if design.get("status") != "FROZEN_READY_TO_OPEN":
        errors.append("design status is not FROZEN_READY_TO_OPEN")
    if design.get("llm_policy") != "closed":
        errors.append("LLM policy must remain closed")

    adapter = set(design.get("adapter_only_files", []))
    locked = set(design.get("locked_files", []))
    if not adapter:
        errors.append("at least one adapter-only file is required")
    if not locked:
        errors.append("at least one locked file is required")
    if adapter & locked:
        errors.append("adapter-only and locked files overlap")

    hashes = design.get("locked_file_sha256", {})
    paths = design.get("locked_file_paths", {})
    exclusions = design.get("target_exclusion_manifest", {})
    cluster_by_file: dict[str, str] = {}
    for cluster in design.get("acquisition_clusters", []):
        cluster_id = cluster.get("cluster_id")
        for filename in cluster.get("files", []):
            if filename in cluster_by_file:
                errors.append(f"file appears in multiple clusters: {filename}")
            cluster_by_file[filename] = cluster_id

    for filename in sorted(locked):
        relative_path = paths.get(filename)
        path = root / design["data_root"] / str(relative_path or filename)
        if not path.is_file():
            errors.append(f"locked file missing: {filename}")
            continue
        expected_hash = hashes.get(filename)
        if not expected_hash or sha256(path) != expected_hash:
            errors.append(f"locked file hash missing or mismatched: {filename}")
        exclusion = exclusions.get(filename, {})
        if not exclusion.get("included_range_bins"):
            errors.append(f"included range bins not frozen: {filename}")
        if not exclusion.get("evidence_source"):
            errors.append(f"target-exclusion evidence missing: {filename}")
        if filename not in cluster_by_file:
            errors.append(f"acquisition cluster missing: {filename}")

    if set(hashes) != locked:
        errors.append("locked hash keys must exactly equal locked files")
    if set(paths) != locked:
        errors.append("locked path keys must exactly equal locked files")
    if set(exclusions) != locked:
        errors.append("target-exclusion keys must exactly equal locked files")
    if set(cluster_by_file) != locked:
        errors.append("clustered files must exactly equal locked files")
    if len(set(cluster_by_file.values())) < int(design.get("minimum_clusters", 1)):
        errors.append("too few independent acquisition clusters")

    for key in (
        "adapter_mapping_sha256",
        "parser_sha256",
        "observation_contract_sha256",
        "p2_model_sha256",
        "endpoint_protocol_sha256",
        "runner_sha256",
    ):
        if not design.get(key):
            errors.append(f"required frozen hash missing: {key}")

    if errors:
        raise ValueError("Confirmation design is not openable:\n- " + "\n- ".join(errors))
    return {
        "status": "OPENABLE",
        "locked_files": sorted(locked),
        "acquisition_clusters": sorted(set(cluster_by_file.values())),
        "design_sha256": hashlib.sha256(
            (json.dumps(design, sort_keys=True, separators=(",", ":")) + "\n").encode()
        ).hexdigest(),
    }
