"""Decode a small batch of stare_4 raw-data chunks from the cached prefix."""

from __future__ import annotations

import json
import os
import struct
import zlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PREFIX_PATH = ROOT / "results" / "dsp_v3_public_data" / "birmingham_626_stare4_member_prefix_16mb_chunked.bin"
OUTPUT_DIR = ROOT / "results" / "dsp_v3_public_data" / "birmingham_626"
REPORT_JSON = OUTPUT_DIR / "stare_4_chunk_batch.json"
REPORT_MD = OUTPUT_DIR / "stare_4_chunk_batch.md"

BASE_ADDRESS = 0x200
NODE_OFFSET = 0x37113A
MAX_RECORDS = int(os.environ.get("BIRMINGHAM_626_STARE4_MAX_RECORDS", "57"))


def parse_tree_node(data: bytes, offset: int) -> dict[str, object]:
    sig, node_type, node_level, entries_used, left_sibling, right_sibling = struct.unpack_from(
        "<4sBBHQQ", data, offset
    )
    if sig != b"TREE":
        raise ValueError(f"Unexpected node signature at 0x{offset:x}: {sig!r}")
    return {
        "offset": offset,
        "signature": sig.decode("ascii"),
        "node_type": int(node_type),
        "node_level": int(node_level),
        "entries_used": int(entries_used),
        "left_sibling": None if left_sibling == 0xFFFFFFFFFFFFFFFF else int(left_sibling),
        "right_sibling": None if right_sibling == 0xFFFFFFFFFFFFFFFF else int(right_sibling),
    }


def parse_record(data: bytes, offset: int) -> dict[str, int]:
    chunk_size, filter_mask, offset0, offset1, offset2, child = struct.unpack_from("<IIQQQQ", data, offset)
    return {
        "offset": offset,
        "chunk_size": int(chunk_size),
        "filter_mask": int(filter_mask),
        "offset0": int(offset0),
        "offset1": int(offset1),
        "offset2": int(offset2),
        "child": int(child),
        "physical_offset": int(BASE_ADDRESS + child),
    }


def decode_chunk(data: bytes, physical_offset: int, chunk_size: int) -> dict[str, object]:
    payload = data[physical_offset : physical_offset + chunk_size]
    decoded = zlib.decompress(payload, 15)
    values = struct.unpack("<" + "d" * (len(decoded) // 8), decoded)
    pairs = [
        (float(values[i]), float(values[i + 1]))
        for i in range(0, min(10, len(values)), 2)
    ]
    return {
        "decoded_bytes": len(decoded),
        "first_pairs": pairs,
    }


def main() -> None:
    data = PREFIX_PATH.read_bytes()
    node = parse_tree_node(data, NODE_OFFSET)
    record_offset = NODE_OFFSET + 24

    records: list[dict[str, object]] = []
    for idx in range(min(MAX_RECORDS, int(node["entries_used"]))):
        record = parse_record(data, record_offset + idx * 40)
        try:
            chunk_info = decode_chunk(data, record["physical_offset"], record["chunk_size"])
            record["decode_status"] = "ok"
            record["decoded_bytes"] = chunk_info["decoded_bytes"]
            record["first_pairs"] = chunk_info["first_pairs"]
        except Exception as error:
            record["decode_status"] = "error"
            record["decode_error"] = repr(error)
        records.append(record)

    report = {
        "prefix_path": str(PREFIX_PATH),
        "base_address": BASE_ADDRESS,
        "node": node,
        "records": records,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    md_lines = [
        "# Birmingham 626 stare_4 chunk batch",
        "",
        f"- prefix_path: `{PREFIX_PATH}`",
        f"- base_address: `0x{BASE_ADDRESS:x}`",
        f"- tree_node_offset: `0x{NODE_OFFSET:x}`",
        "",
        "## Node",
        "",
        f"- node_type: `{node['node_type']}`",
        f"- node_level: `{node['node_level']}`",
        f"- entries_used: `{node['entries_used']}`",
        "",
        "## Records",
        "",
        "| idx | chunk_size | filter_mask | offset0 | offset1 | offset2 | child | physical_offset | decode_status | decoded_bytes |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |",
    ]
    for idx, record in enumerate(records):
        md_lines.append(
            "| {idx} | {chunk_size} | {filter_mask} | {offset0} | {offset1} | {offset2} | `0x{child:x}` | `0x{physical_offset:x}` | {decode_status} | {decoded_bytes} |".format(
                idx=idx,
                chunk_size=record["chunk_size"],
                filter_mask=record["filter_mask"],
                offset0=record["offset0"],
                offset1=record["offset1"],
                offset2=record["offset2"],
                child=record["child"],
                physical_offset=record["physical_offset"],
                decode_status=record.get("decode_status", "unknown"),
                decoded_bytes=record.get("decoded_bytes", 0),
            )
        )
        if "first_pairs" in record:
            md_lines.append(
                f"- idx {idx} first pairs: {record['first_pairs']}"
            )
        if "decode_error" in record:
            md_lines.append(f"- idx {idx} error: `{record['decode_error']}`")
    REPORT_MD.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
