"""Remote ZIP inspection helpers for the Birmingham 626 archive.

The helpers are intentionally read-only. They use HTTP range requests to
inspect the archive tail, parse the central directory, and decode ZIP64
metadata when present.
"""

from __future__ import annotations

from dataclasses import dataclass
import io
import struct
from pathlib import Path
from typing import Any

import requests


ZIP_EOCD_SIG = b"PK\x05\x06"
ZIP64_LOCATOR_SIG = b"PK\x06\x07"
ZIP64_EOCD_SIG = b"PK\x06\x06"
ZIP_CENTRAL_DIR_SIG = b"PK\x01\x02"


_REMOTE_SESSION = requests.Session()
_REMOTE_SESSION.trust_env = False


def _remote_get(url: str, *, headers: dict[str, str], timeout: int) -> requests.Response:
    return _REMOTE_SESSION.get(url, headers=headers, timeout=timeout)


@dataclass(frozen=True)
class RemoteZipEntry:
    name: str
    compressed_size: int
    uncompressed_size: int
    local_header_offset: int
    compression_method: int
    flags: int
    crc32: int


@dataclass(frozen=True)
class RemoteZipLocalHeader:
    compression_method: int
    flags: int
    compressed_size: int
    uncompressed_size: int
    filename: str
    extra_len: int
    data_offset: int


@dataclass(frozen=True)
class RemoteMatV5Peek:
    member_name: str
    variable_name: str
    dimensions: tuple[int, ...]
    matlab_class: int
    complex_flag: bool
    text_header: str
    header_bytes: int


@dataclass(frozen=True)
class RemoteMatV5PrefixPeek:
    member_name: str
    variable_name: str
    dimensions: tuple[int, ...]
    matlab_class: int
    complex_flag: bool
    text_header: str
    header_bytes: int
    fetched_member_prefix_bytes: int
    decompressed_prefix_bytes: int


def http_range_get(url: str, *, suffix_bytes: int, timeout: int = 120) -> tuple[bytes, dict[str, str]]:
    headers = {"Range": f"bytes=-{int(suffix_bytes)}"}
    response = _remote_get(url, headers=headers, timeout=timeout)
    response.raise_for_status()
    return response.content, {k.lower(): v for k, v in response.headers.items()}


def _parse_content_range(value: str | None) -> int:
    if not value:
        raise ValueError("Missing Content-Range header")
    # Example: "bytes 8685984015-8690178318/8690178319"
    try:
        _, span = value.split(" ", 1)
        _, total = span.rsplit("/", 1)
        return int(total)
    except Exception as error:  # pragma: no cover - defensive parsing
        raise ValueError(f"Invalid Content-Range header: {value!r}") from error


def _find_eocd(tail: bytes) -> int:
    pos = tail.rfind(ZIP_EOCD_SIG)
    if pos < 0:
        raise ValueError("ZIP end-of-central-directory not found")
    return pos


def _read_zip64_locator(tail: bytes, eocd_pos: int) -> int:
    locator_pos = eocd_pos - 20
    if locator_pos < 0 or tail[locator_pos : locator_pos + 4] != ZIP64_LOCATOR_SIG:
        raise ValueError("ZIP64 locator not found")
    _, _, zip64_eocd_offset, _ = struct.unpack_from("<4sIQI", tail, locator_pos)
    return zip64_eocd_offset


def _read_zip64_eocd(tail: bytes, zip64_offset: int, tail_start: int) -> tuple[int, int, int]:
    relative = zip64_offset - tail_start
    if relative < 0 or relative + 56 > len(tail):
        raise ValueError("ZIP64 EOCD not fully contained in fetched tail")
    if tail[relative : relative + 4] != ZIP64_EOCD_SIG:
        raise ValueError("ZIP64 EOCD signature not found")
    # Size field begins at byte 4.
    fields = struct.unpack_from("<4sQHHIIQQQQ", tail, relative)
    _, _, _, _, _, _, total_entries_disk, total_entries, cd_size, cd_offset = fields
    return total_entries, cd_size, cd_offset


def _decode_zip64_extra(
    extra: bytes, *, need_csize: bool, need_usize: bool, need_offset: bool
) -> tuple[int | None, int | None, int | None]:
    cursor = 0
    csize = usize = offset = None
    while cursor + 4 <= len(extra):
        header_id, data_size = struct.unpack_from("<HH", extra, cursor)
        cursor += 4
        data = extra[cursor : cursor + data_size]
        cursor += data_size
        if header_id != 0x0001:
            continue
        stream = io.BytesIO(data)
        if need_usize:
            if len(data) < 8:
                break
            usize = struct.unpack("<Q", stream.read(8))[0]
        if need_csize:
            if len(data) < stream.tell() + 8:
                break
            csize = struct.unpack("<Q", stream.read(8))[0]
        if need_offset:
            if len(data) < stream.tell() + 8:
                break
            offset = struct.unpack("<Q", stream.read(8))[0]
        break
    return csize, usize, offset


def inspect_remote_zip(url: str, *, suffix_bytes: int = 4 * 1024 * 1024) -> dict[str, Any]:
    tail, headers = http_range_get(url, suffix_bytes=suffix_bytes)
    content_range = headers.get("content-range")
    archive_size = _parse_content_range(content_range)
    tail_start = archive_size - len(tail)

    eocd_pos = _find_eocd(tail)
    (
        _sig,
        _disk,
        _cd_disk,
        _disk_entries,
        total_entries_16,
        cd_size_16,
        cd_offset_16,
        _comment_len,
    ) = struct.unpack_from("<4sHHHHIIH", tail, eocd_pos)

    total_entries = total_entries_16
    cd_size = cd_size_16
    cd_offset = cd_offset_16
    if total_entries_16 == 0xFFFF or cd_size_16 == 0xFFFFFFFF or cd_offset_16 == 0xFFFFFFFF:
        zip64_offset = _read_zip64_locator(tail, eocd_pos)
        total_entries, cd_size, cd_offset = _read_zip64_eocd(tail, zip64_offset, tail_start)

    cd_relative = cd_offset - tail_start
    if cd_relative < 0 or cd_relative + cd_size > len(tail):
        raise ValueError("Central directory not fully contained in fetched tail")

    entries: list[RemoteZipEntry] = []
    cursor = cd_relative
    limit = cd_relative + cd_size
    while cursor < limit:
        if tail[cursor : cursor + 4] != ZIP_CENTRAL_DIR_SIG:
            raise ValueError("Central directory signature mismatch")
        (
            _sig,
            _ver_made,
            _ver_needed,
            flags,
            method,
            _mod_time,
            _mod_date,
            crc32,
            csize_32,
            usize_32,
            name_len,
            extra_len,
            comment_len,
            _disk_start,
            _int_attr,
            _ext_attr,
            rel_off_32,
        ) = struct.unpack_from("<4sHHHHHHIIIHHHHHII", tail, cursor)
        name_start = cursor + 46
        name_end = name_start + name_len
        extra_start = name_end
        extra_end = extra_start + extra_len
        comment_end = extra_end + comment_len
        if comment_end > len(tail):
            raise ValueError("Central directory entry extends past fetched tail")
        raw_name = tail[name_start:name_end]
        name = raw_name.decode("utf-8", errors="replace")
        extra = tail[extra_start:extra_end]
        need_csize = csize_32 == 0xFFFFFFFF
        need_usize = usize_32 == 0xFFFFFFFF
        need_offset = rel_off_32 == 0xFFFFFFFF
        csize = csize_32
        usize = usize_32
        rel_off = rel_off_32
        if need_csize or need_usize or need_offset:
            parsed_csize, parsed_usize, parsed_offset = _decode_zip64_extra(
                extra,
                need_csize=need_csize,
                need_usize=need_usize,
                need_offset=need_offset,
            )
            if need_csize and parsed_csize is not None:
                csize = parsed_csize
            if need_usize and parsed_usize is not None:
                usize = parsed_usize
            if need_offset and parsed_offset is not None:
                rel_off = parsed_offset
        entries.append(
            RemoteZipEntry(
                name=name,
                compressed_size=int(csize),
                uncompressed_size=int(usize),
                local_header_offset=int(rel_off),
                compression_method=int(method),
                flags=int(flags),
                crc32=int(crc32),
            )
        )
        cursor = comment_end

    return {
        "url": url,
        "archive_size": archive_size,
        "tail_bytes": len(tail),
        "central_directory_offset": int(cd_offset),
        "central_directory_size": int(cd_size),
        "total_entries": int(total_entries),
        "entries": [entry.__dict__ for entry in entries],
    }


def fetch_remote_member_bytes(
    url: str,
    entry: RemoteZipEntry,
    *,
    tail_probe: int = 64,
) -> tuple[bytes, RemoteZipLocalHeader]:
    """Fetch one archive member by range requests and return its payload bytes."""

    header_response = _remote_get(
        url,
        headers={"Range": f"bytes={entry.local_header_offset}-{entry.local_header_offset + max(0, tail_probe - 1)}"},
        timeout=120,
    )
    header_response.raise_for_status()
    header = header_response.content
    if len(header) < 30 or header[:4] != b"PK\x03\x04":
        raise ValueError("Remote local file header is missing or invalid")
    (
        _sig,
        _version,
        flags,
        method,
        _mtime,
        _mdate,
        _crc32,
        csize_32,
        usize_32,
        name_len,
        extra_len,
    ) = struct.unpack_from("<4sHHHHHIIIHH", header, 0)
    header_size = 30 + name_len + extra_len
    header_need = entry.local_header_offset + header_size
    if len(header) < header_size:
        header_response = _remote_get(
            url,
            headers={"Range": f"bytes={entry.local_header_offset}-{header_need - 1}"},
            timeout=120,
        )
        header_response.raise_for_status()
        header = header_response.content
        if len(header) < header_size:
            raise ValueError("Remote local file header truncated")
    filename = header[30 : 30 + name_len].decode("utf-8", errors="replace")
    data_offset = entry.local_header_offset + header_size
    if method == 0:
        size = entry.uncompressed_size
    else:
        size = entry.compressed_size
    data_response = requests.get(
        url,
        headers={"Range": f"bytes={data_offset}-{data_offset + size - 1}"},
        timeout=180,
    )
    data_response.raise_for_status()
    payload = data_response.content
    if len(payload) != size:
        raise ValueError(f"Expected {size} bytes, got {len(payload)}")
    if method == 0:
        member_bytes = payload
    elif method == 8:
        import zlib

        member_bytes = zlib.decompress(payload, -15)
    else:
        raise ValueError(f"Unsupported compression method: {method}")
    return (
        member_bytes,
        RemoteZipLocalHeader(
            compression_method=int(method),
            flags=int(flags),
            compressed_size=int(entry.compressed_size),
            uncompressed_size=int(entry.uncompressed_size),
            filename=filename,
            extra_len=int(extra_len),
            data_offset=int(data_offset),
        ),
    )


def fetch_remote_member_prefix_bytes(
    url: str,
    entry: RemoteZipEntry,
    *,
    prefix_bytes: int = 64 * 1024,
    tail_probe: int = 64,
) -> tuple[bytes, RemoteZipLocalHeader]:
    """Fetch only the beginning of a remote archive member.

    For stored members this returns the raw member prefix bytes.
    For deflated members this returns the raw compressed prefix bytes.
    """

    header_response = requests.get(
        url,
        headers={"Range": f"bytes={entry.local_header_offset}-{entry.local_header_offset + max(0, tail_probe - 1)}"},
        timeout=120,
    )
    header_response.raise_for_status()
    header = header_response.content
    if len(header) < 30 or header[:4] != b"PK\x03\x04":
        raise ValueError("Remote local file header is missing or invalid")
    (
        _sig,
        _version,
        flags,
        method,
        _mtime,
        _mdate,
        _crc32,
        csize_32,
        usize_32,
        name_len,
        extra_len,
    ) = struct.unpack_from("<4sHHHHHIIIHH", header, 0)
    header_size = 30 + name_len + extra_len
    header_need = entry.local_header_offset + header_size
    if len(header) < header_size:
        header_response = requests.get(
            url,
            headers={"Range": f"bytes={entry.local_header_offset}-{header_need - 1}"},
            timeout=120,
        )
        header_response.raise_for_status()
        header = header_response.content
        if len(header) < header_size:
            raise ValueError("Remote local file header truncated")
    filename = header[30 : 30 + name_len].decode("utf-8", errors="replace")
    data_offset = entry.local_header_offset + header_size
    if method == 0:
        size = min(int(prefix_bytes), int(entry.uncompressed_size))
    elif method == 8:
        size = min(int(prefix_bytes), int(entry.compressed_size))
    else:
        raise ValueError(f"Unsupported compression method: {method}")
    data_response = _remote_get(
        url,
        headers={"Range": f"bytes={data_offset}-{data_offset + size - 1}"},
        timeout=120,
    )
    data_response.raise_for_status()
    payload = data_response.content
    if len(payload) != size:
        raise ValueError(f"Expected {size} bytes, got {len(payload)}")
    return (
        payload,
        RemoteZipLocalHeader(
            compression_method=int(method),
            flags=int(flags),
            compressed_size=int(entry.compressed_size),
            uncompressed_size=int(entry.uncompressed_size),
            filename=filename,
            extra_len=int(extra_len),
            data_offset=int(data_offset),
        ),
    )


def fetch_remote_member_prefix_bytes_chunked(
    url: str,
    entry: RemoteZipEntry,
    *,
    prefix_bytes: int = 64 * 1024,
    tail_probe: int = 64,
    request_bytes: int = 256 * 1024,
) -> tuple[bytes, RemoteZipLocalHeader]:
    """Fetch a remote archive member prefix using smaller Range requests.

    This is slower than ``fetch_remote_member_prefix_bytes`` but is more robust
    for very large members served by a remote endpoint that times out on larger
    single requests.
    """

    header_response = _remote_get(
        url,
        headers={"Range": f"bytes={entry.local_header_offset}-{entry.local_header_offset + max(0, tail_probe - 1)}"},
        timeout=120,
    )
    header_response.raise_for_status()
    header = header_response.content
    if len(header) < 30 or header[:4] != b"PK\x03\x04":
        raise ValueError("Remote local file header is missing or invalid")
    (
        _sig,
        _version,
        flags,
        method,
        _mtime,
        _mdate,
        _crc32,
        csize_32,
        usize_32,
        name_len,
        extra_len,
    ) = struct.unpack_from("<4sHHHHHIIIHH", header, 0)
    header_size = 30 + name_len + extra_len
    header_need = entry.local_header_offset + header_size
    if len(header) < header_size:
        header_response = _remote_get(
            url,
            headers={"Range": f"bytes={entry.local_header_offset}-{header_need - 1}"},
            timeout=120,
        )
        header_response.raise_for_status()
        header = header_response.content
        if len(header) < header_size:
            raise ValueError("Remote local file header truncated")
    filename = header[30 : 30 + name_len].decode("utf-8", errors="replace")
    data_offset = entry.local_header_offset + header_size
    if method == 0:
        size = min(int(prefix_bytes), int(entry.uncompressed_size))
    elif method == 8:
        size = min(int(prefix_bytes), int(entry.compressed_size))
    else:
        raise ValueError(f"Unsupported compression method: {method}")

    payload_parts: list[bytes] = []
    remaining = size
    cursor = data_offset
    while remaining > 0:
        take = min(int(request_bytes), int(remaining))
        data_response = _remote_get(
            url,
            headers={"Range": f"bytes={cursor}-{cursor + take - 1}"},
            timeout=120,
        )
        data_response.raise_for_status()
        payload = data_response.content
        if not payload:
            raise ValueError("Remote member prefix request returned no data")
        payload_parts.append(payload)
        cursor += len(payload)
        remaining -= len(payload)
        if len(payload) < take:
            break

    combined = b"".join(payload_parts)
    if not combined:
        raise ValueError("Remote member prefix request returned empty payload")
    return (
        combined,
        RemoteZipLocalHeader(
            compression_method=int(method),
            flags=int(flags),
            compressed_size=int(entry.compressed_size),
            uncompressed_size=int(entry.uncompressed_size),
            filename=filename,
            extra_len=int(extra_len),
            data_offset=int(data_offset),
        ),
    )


def peek_remote_mat_v5(
    url: str,
    entry: RemoteZipEntry,
    *,
    fetch_bytes: int = 64 * 1024,
    decompress_budget: int = 4096,
) -> RemoteMatV5Peek:
    """Inspect the first variable in a remote MATLAB v5 member."""

    import zlib

    local, _ = fetch_remote_member_bytes(url, entry, tail_probe=fetch_bytes)
    if len(local) < 136:
        raise ValueError("Remote MAT payload is too small to inspect")
    if local[:128] and not local[:128].startswith(b"MATLAB 5.0 MAT-file"):
        raise ValueError("Remote member is not a MATLAB v5 container")
    text_header = local[:116].decode("latin-1", errors="replace").rstrip("\x00")
    byte_order = "<" if local[126:128] == b"IM" else ">"
    outer_type, outer_size = struct.unpack_from(byte_order + "II", local, 128)
    if outer_type != 15:
        raise ValueError(f"Expected miCOMPRESSED element, got {outer_type}")
    compressed = local[136:]
    decompressor = zlib.decompressobj()
    inner = decompressor.decompress(compressed, decompress_budget)
    if len(inner) < 64:
        raise ValueError("Could not decompress enough bytes to inspect MAT payload")
    inner_type, inner_size = struct.unpack_from(byte_order + "II", inner, 0)
    if inner_type != 14:
        raise ValueError(f"Expected miMATRIX element, got {inner_type}")
    cursor = 8
    sub_type, sub_size = struct.unpack_from(byte_order + "II", inner, cursor)
    cursor += 8
    if sub_type != 6 or sub_size < 8:
        raise ValueError("Invalid MAT array-flag subelement")
    flags = inner[cursor : cursor + sub_size]
    class_bits = struct.unpack_from(byte_order + "I", flags, 0)[0]
    matlab_class = class_bits & 0xFF
    complex_flag = bool(class_bits & 0x0800)
    cursor += (sub_size + 7) & ~7
    sub_type, sub_size = struct.unpack_from(byte_order + "II", inner, cursor)
    cursor += 8
    if sub_type != 5:
        raise ValueError("Invalid MAT dimensions subelement")
    dims = struct.unpack(byte_order + "I" * (sub_size // 4), inner[cursor : cursor + sub_size])
    cursor += (sub_size + 7) & ~7
    sub_type, sub_size = struct.unpack_from(byte_order + "II", inner, cursor)
    cursor += 8
    if sub_size == 0:
        raise ValueError("Empty MAT variable name")
    variable_name = inner[cursor : cursor + sub_size].decode("utf-8", errors="replace")
    return RemoteMatV5Peek(
        member_name=entry.name,
        variable_name=variable_name,
        dimensions=tuple(int(v) for v in dims),
        matlab_class=int(matlab_class),
        complex_flag=complex_flag,
        text_header=text_header,
        header_bytes=128,
    )


def peek_remote_mat_v5_prefix(
    url: str,
    entry: RemoteZipEntry,
    *,
    fetch_bytes: int = 128 * 1024,
    decompress_budget: int = 64 * 1024,
) -> RemoteMatV5PrefixPeek:
    """Inspect the first variable in a remote MATLAB v5 member from a prefix only."""

    import zlib

    prefix, _ = fetch_remote_member_prefix_bytes(url, entry, prefix_bytes=fetch_bytes)
    if len(prefix) < 30:
        raise ValueError("Remote MAT payload prefix is too small to inspect")
    if entry.compression_method == 0:
        member_prefix = prefix
    elif entry.compression_method == 8:
        zip_decompressor = zlib.decompressobj(-15)
        member_prefix = zip_decompressor.decompress(prefix, decompress_budget)
    else:
        raise ValueError(f"Unsupported compression method: {entry.compression_method}")
    if len(member_prefix) < 136:
        raise ValueError("Remote MAT member prefix is too small to inspect")
    if member_prefix[:128] and not member_prefix[:128].startswith(b"MATLAB 5.0 MAT-file"):
        raise ValueError("Remote member is not a MATLAB v5 container")
    text_header = member_prefix[:116].decode("latin-1", errors="replace").rstrip("\x00")
    byte_order = "<" if member_prefix[126:128] == b"IM" else ">"
    outer_type, outer_size = struct.unpack_from(byte_order + "II", member_prefix, 128)
    if outer_type != 15:
        inner = member_prefix[128:]
        decompressed_prefix_bytes = len(inner)
    else:
        compressed = member_prefix[136:]
        decompressor = zlib.decompressobj()
        inner = decompressor.decompress(compressed, decompress_budget)
        decompressed_prefix_bytes = len(inner)
    if len(inner) < 64:
        raise ValueError("Could not decompress enough bytes to inspect MAT payload")
    inner_type, inner_size = struct.unpack_from(byte_order + "II", inner, 0)
    if inner_type != 14:
        raise ValueError(f"Expected miMATRIX element, got {inner_type}")
    cursor = 8
    sub_type, sub_size = struct.unpack_from(byte_order + "II", inner, cursor)
    cursor += 8
    if sub_type != 6 or sub_size < 8:
        raise ValueError("Invalid MAT array-flag subelement")
    flags = inner[cursor : cursor + sub_size]
    class_bits = struct.unpack_from(byte_order + "I", flags, 0)[0]
    matlab_class = class_bits & 0xFF
    complex_flag = bool(class_bits & 0x0800)
    cursor += (sub_size + 7) & ~7
    sub_type, sub_size = struct.unpack_from(byte_order + "II", inner, cursor)
    cursor += 8
    if sub_type != 5:
        raise ValueError("Invalid MAT dimensions subelement")
    dims = struct.unpack(byte_order + "I" * (sub_size // 4), inner[cursor : cursor + sub_size])
    cursor += (sub_size + 7) & ~7
    sub_type, sub_size = struct.unpack_from(byte_order + "II", inner, cursor)
    cursor += 8
    if sub_size == 0:
        raise ValueError("Empty MAT variable name")
    variable_name = inner[cursor : cursor + sub_size].decode("utf-8", errors="replace")
    return RemoteMatV5PrefixPeek(
        member_name=entry.name,
        variable_name=variable_name,
        dimensions=tuple(int(v) for v in dims),
        matlab_class=int(matlab_class),
        complex_flag=complex_flag,
        text_header=text_header,
        header_bytes=128,
        fetched_member_prefix_bytes=len(prefix),
        decompressed_prefix_bytes=int(decompressed_prefix_bytes),
    )
