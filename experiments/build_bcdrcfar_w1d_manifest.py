from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.bcdrcfar.protocol import build_w1d_manifest, load_protocol  # noqa: E402


CONFIG = ROOT / "configs" / "bcdrcfar_protocol.json"
MANIFEST = ROOT / "data" / "manifests" / "bcdrcfar_w1d_cells.csv"
OUTPUT = ROOT / "results" / "bcdrcfar_w1d"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    protocol = load_protocol(CONFIG)
    frame = build_w1d_manifest(protocol)
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(MANIFEST, index=False, encoding="utf-8")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "BCDRCFAR_W1D_MANIFEST_BUILT",
        "protocol_version": protocol["protocol_version"],
        "cells": len(frame),
        "split_counts": frame["split"].value_counts().sort_index().to_dict(),
        "scenario_counts": frame["scenario"].value_counts().sort_index().to_dict(),
        "manifest_path": str(MANIFEST.resolve()),
        "manifest_sha256": sha256(MANIFEST),
        "config_sha256": sha256(CONFIG),
        "outcomes_opened": False,
        "calibration_opened": False,
        "locked_opened": False,
    }
    (OUTPUT / "manifest.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
