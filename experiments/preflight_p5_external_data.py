"""Inventory P5 external files without parsing or scoring locked data."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "p5_external_data_acquisition.json"
OUTPUT = ROOT / "results" / "p5_external_data_preflight"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def matlab_container_hint(path: Path) -> str:
    header = path.read_bytes()[:128]
    if header.startswith(b"MATLAB 5.0 MAT-file"):
        return "mat_v5_to_v7"
    if header.startswith(b"\x89HDF\r\n\x1a\n"):
        return "mat_v7_3_hdf5"
    return "unknown"


def main() -> None:
    registry = json.loads(CONFIG.read_text(encoding="utf-8"))
    root = ROOT / registry["storage_root"]
    expected = []
    for source in registry["sources"]:
        for role_key in ("adapter_only_files", "locked_files"):
            for filename in source.get(role_key, []):
                expected.append((source["name"], role_key, filename))
        for filename in source.get("priority_files", []):
            expected.append((source["name"], "priority_files", filename))

    files = []
    for source, role, filename in expected:
        candidates = list(root.rglob(filename)) if root.exists() else []
        if len(candidates) > 1:
            raise RuntimeError(f"Duplicate external file name: {filename}")
        if not candidates:
            files.append(
                {
                    "source": source,
                    "role": role,
                    "filename": filename,
                    "status": "MISSING",
                }
            )
            continue
        path = candidates[0]
        files.append(
            {
                "source": source,
                "role": role,
                "filename": filename,
                "relative_path": str(path.relative_to(ROOT)).replace("\\", "/"),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
                "container_hint": matlab_container_hint(path),
                "status": "PRESENT_UNOPENED",
            }
        )

    missing = [row["filename"] for row in files if row["status"] == "MISSING"]
    summary = {
        "status": "WAITING_FOR_AUTHORIZED_DOWNLOAD" if missing else "HASH_INVENTORY_READY",
        "registry_sha256": sha256(CONFIG),
        "files": files,
        "missing_files": missing,
        "locked_data_parsed_or_scored": False,
        "next_action": (
            "complete authorized download and rerun preflight"
            if missing
            else "freeze parser tests and target-exclusion manifest before opening locked files"
        ),
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "preflight.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
