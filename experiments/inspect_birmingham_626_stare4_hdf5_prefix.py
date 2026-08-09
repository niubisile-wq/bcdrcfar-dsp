"""Inspect the Birmingham 626 stare_4 HDF5 member from a virtual prefix."""

from __future__ import annotations

import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import h5py
import numpy as np
import zlib

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.birmingham_626 import fetch_remote_member_prefix_bytes  # noqa: E402


URL = "https://edata.bham.ac.uk/626/7/data.zip"
MANIFEST_PATH = ROOT / "results" / "dsp_v3_public_data" / "birmingham_626_remote_manifest.json"
OUTPUT_DIR = ROOT / "results" / "dsp_v3_public_data" / "birmingham_626"
REPORT_PATH = OUTPUT_DIR / "stare_4_hdf5_prefix.json"
MD_PATH = OUTPUT_DIR / "stare_4_hdf5_prefix.md"


class VirtualH5(io.RawIOBase):
    def __init__(self, prefix: bytes, size: int):
        self.prefix = prefix
        self.size = size
        self.pos = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset, whence=io.SEEK_SET):
        if whence == io.SEEK_SET:
            self.pos = max(0, int(offset))
        elif whence == io.SEEK_CUR:
            self.pos = max(0, self.pos + int(offset))
        elif whence == io.SEEK_END:
            self.pos = max(0, self.size + int(offset))
        else:
            raise ValueError("bad whence")
        return self.pos

    def readinto(self, b):
        if self.pos >= self.size:
            return 0
        n = min(len(b), self.size - self.pos)
        start = self.pos
        end = self.pos + n
        data = self.prefix[start:end]
        if len(data) < n:
            data = data + b"\x00" * (n - len(data))
        b[:n] = data
        self.pos = end
        return n


def main() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    entry = next(item for item in manifest["entries"] if item["name"] == "stare_4_radar.mat")
    prefix, _ = fetch_remote_member_prefix_bytes(URL, SimpleNamespace(**entry), prefix_bytes=8 * 1024 * 1024)
    member_prefix = zlib.decompressobj(-15).decompress(prefix, 2 * 1024 * 1024)
    vfile = VirtualH5(member_prefix, int(entry["uncompressed_size"]))

    payload: dict[str, object] = {
        "url": URL,
        "entry": entry,
        "zip_prefix_bytes": len(prefix),
        "member_prefix_bytes": len(member_prefix),
        "status": "ok",
        "root_keys": [],
        "datasets": [],
        "sample_errors": [],
    }
    md_lines = [
        "# Birmingham 626 stare_4 HDF5 prefix",
        "",
        f"- url: `{URL}`",
        f"- zip_prefix_bytes: `{len(prefix)}`",
        f"- member_prefix_bytes: `{len(member_prefix)}`",
        "",
    ]

    try:
        with h5py.File(vfile, "r") as handle:
            keys = list(handle.keys())
            payload["root_keys"] = keys
            md_lines.append("## Root Keys")
            md_lines.append("")
            md_lines.extend([f"- `{key}`" for key in keys])
            md_lines.append("")
            for key in keys:
                try:
                    ds = handle[key]
                    item = {
                        "name": key,
                        "kind": "dataset" if isinstance(ds, h5py.Dataset) else "group",
                        "shape": list(ds.shape) if isinstance(ds, h5py.Dataset) else None,
                        "dtype": str(ds.dtype) if isinstance(ds, h5py.Dataset) else None,
                    }
                    if isinstance(ds, h5py.Dataset) and key == "stare_4_rx1":
                        try:
                            sample = np.asarray(ds[0:2, 0:3])
                            item["sample_shape"] = list(sample.shape)
                            item["sample_preview"] = np.asarray(sample).tolist()
                        except Exception as sample_error:  # pragma: no cover - best effort
                            item["sample_error"] = repr(sample_error)
                            payload["sample_errors"].append({"name": key, "error": repr(sample_error)})
                    payload["datasets"].append(item)
                    md_lines.append(f"## `{key}`")
                    md_lines.append("")
                    md_lines.append(f"- kind: `{item['kind']}`")
                    if item["shape"] is not None:
                        md_lines.append(f"- shape: `{item['shape']}`")
                        md_lines.append(f"- dtype: `{item['dtype']}`")
                    if "sample_preview" in item:
                        md_lines.append(f"- sample_preview: `{item['sample_preview']}`")
                    if "sample_error" in item:
                        md_lines.append(f"- sample_error: `{item['sample_error']}`")
                    md_lines.append("")
                except Exception as error:
                    payload["sample_errors"].append({"name": key, "error": repr(error)})
                    md_lines.append(f"## `{key}`")
                    md_lines.append("")
                    md_lines.append(f"- error: `{error!r}`")
                    md_lines.append("")
    except Exception as error:
        payload["status"] = "error"
        payload["error"] = repr(error)
        md_lines.extend(["## Open Error", "", f"- error: `{error!r}`", ""])

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    MD_PATH.write_text("\n".join(md_lines), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
