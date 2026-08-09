"""Parse the suspected v1 object header for Birmingham 626 stare_4_rx1."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from types import SimpleNamespace
import zlib

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.birmingham_626 import fetch_remote_member_prefix_bytes  # noqa: E402


URL = "https://edata.bham.ac.uk/626/7/data.zip"
MANIFEST_PATH = ROOT / "results" / "dsp_v3_public_data" / "birmingham_626_remote_manifest.json"


def u16le(buf: bytes, off: int) -> int:
    return int.from_bytes(buf[off : off + 2], "little")


def u32le(buf: bytes, off: int) -> int:
    return int.from_bytes(buf[off : off + 4], "little")


def u64le(buf: bytes, off: int) -> int:
    return int.from_bytes(buf[off : off + 8], "little")


def parse_v1_header(buf: bytes, start: int) -> list[str]:
    lines: list[str] = []
    if start + 12 > len(buf):
        return [f"truncated header at {start:#x}"]

    version = buf[start]
    reserved = buf[start + 1]
    nmsgs = u16le(buf, start + 2)
    refcount = u32le(buf, start + 4)
    header_size = u32le(buf, start + 8)
    lines.append(
        f"header@{start:#x} version={version} reserved={reserved} nmsgs={nmsgs} refcount={refcount} header_size={header_size}"
    )

    cursor = start + 12
    for idx in range(nmsgs):
        if cursor + 8 > len(buf):
            lines.append(f"  msg[{idx}] truncated at {cursor:#x}")
            break
        msg_type = u16le(buf, cursor)
        msg_size = u16le(buf, cursor + 2)
        msg_flags = buf[cursor + 4]
        msg_reserved = buf[cursor + 5]
        data_start = cursor + 8
        data_end = data_start + msg_size
        data = buf[data_start:data_end]
        lines.append(
            f"  msg[{idx}] type=0x{msg_type:04x} size={msg_size} flags=0x{msg_flags:02x} reserved=0x{msg_reserved:02x} data@{data_start:#x}-{data_end:#x}"
        )
        lines.append(f"    data_hex={data[:64].hex()}")
        if msg_type == 0x0010 and len(data) >= 16:
            offset = u64le(data, 0)
            length = u64le(data, 8)
            lines.append(f"    continuation offset={offset:#x} length={length:#x}")
        cursor = (data_end + 7) & ~7

    return lines


def main() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    entry = next(item for item in manifest["entries"] if item["name"] == "stare_4_radar.mat")
    prefix, _ = fetch_remote_member_prefix_bytes(URL, SimpleNamespace(**entry), prefix_bytes=32 * 1024 * 1024)
    member_prefix = zlib.decompressobj(-15).decompress(prefix, 8 * 1024 * 1024)
    header_start = 0x520
    for line in parse_v1_header(member_prefix, header_start):
        print(line)


if __name__ == "__main__":
    main()
