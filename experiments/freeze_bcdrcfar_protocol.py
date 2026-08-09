from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.bcdrcfar.model import BCDRCFAR, count_trainable_parameters  # noqa: E402
from src.bcdrcfar.protocol import load_protocol  # noqa: E402


CONFIG = ROOT / "configs" / "bcdrcfar_protocol.json"
OUTPUT = ROOT / "results" / "bcdrcfar_freeze"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    protocol = load_protocol(CONFIG)
    model = BCDRCFAR(
        hidden_channels=int(protocol["method"]["hidden_channels"]),
        distribution_pool_bins=int(protocol["method"]["distribution_pool_bins"]),
        dilations=tuple(int(value) for value in protocol["method"]["dilations"]),
        maximum_score_multiplier=float(protocol["method"]["maximum_score_multiplier"]),
        maximum_threshold_multiplier=float(protocol["method"]["maximum_threshold_multiplier"]),
    )
    parameters = count_trainable_parameters(model)
    if parameters > int(protocol["method"]["maximum_parameters"]):
        raise RuntimeError("model exceeds the frozen parameter budget")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "BCDRCFAR_PROTOCOL_FROZEN_FOR_DEVELOPMENT",
        "protocol_version": protocol["protocol_version"],
        "config_path": str(CONFIG.resolve()),
        "config_sha256": sha256(CONFIG),
        "model_code_sha256": sha256(ROOT / "src" / "bcdrcfar" / "model.py"),
        "protocol_code_sha256": sha256(ROOT / "src" / "bcdrcfar" / "protocol.py"),
        "simulation_code_sha256": sha256(ROOT / "src" / "bcdrcfar" / "simulation.py"),
        "trainable_parameters": parameters,
        "calibration_opened": False,
        "locked_opened": False,
    }
    (OUTPUT / "manifest.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
