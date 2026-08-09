"""Explicit, non-heuristic readers for MATLAB radar containers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import numpy as np
from scipy.io import loadmat, whosmat


def matlab_container(path: str | Path) -> str:
    header = Path(path).read_bytes()[:128]
    if header.startswith(b"MATLAB 5.0 MAT-file"):
        return "mat_v5_to_v7"
    if header.startswith(b"\x89HDF\r\n\x1a\n"):
        return "mat_v7_3_hdf5"
    raise ValueError("Unsupported or invalid MATLAB container")


def inventory_mat(path: str | Path) -> list[dict[str, Any]]:
    """List variables or datasets without loading numeric payloads."""

    path = Path(path)
    kind = matlab_container(path)
    if kind == "mat_v5_to_v7":
        return [
            {"path": name, "shape": list(shape), "class": matlab_class}
            for name, shape, matlab_class in whosmat(path)
        ]

    rows: list[dict[str, Any]] = []
    with h5py.File(path, "r") as handle:
        def visitor(name: str, obj: h5py.Dataset | h5py.Group) -> None:
            if isinstance(obj, h5py.Dataset):
                rows.append(
                    {
                        "path": name,
                        "shape": list(obj.shape),
                        "dtype": str(obj.dtype),
                    }
                )

        handle.visititems(visitor)
    return rows


def _resolve(value: Any, path_parts: list[str]) -> Any:
    for part in path_parts:
        if isinstance(value, dict):
            if part not in value:
                raise KeyError(f"MAT path component not found: {part}")
            value = value[part]
        elif hasattr(value, part):
            value = getattr(value, part)
        elif isinstance(value, np.ndarray) and value.dtype.names and part in value.dtype.names:
            value = value[part]
        else:
            raise KeyError(f"MAT path component not found: {part}")
    return value


def load_explicit_array(path: str | Path, variable_path: str) -> np.ndarray:
    """Load exactly one named variable/dataset; never select by size or dtype."""

    path = Path(path)
    parts = [part for part in variable_path.strip("/").split("/") if part]
    if not parts:
        raise ValueError("variable_path must be explicit")
    kind = matlab_container(path)
    if kind == "mat_v5_to_v7":
        payload = loadmat(
            path,
            variable_names=[parts[0]],
            simplify_cells=True,
            chars_as_strings=True,
        )
        if parts[0] not in payload:
            raise KeyError(f"MAT variable not found: {parts[0]}")
        return np.asarray(_resolve(payload[parts[0]], parts[1:]))
    with h5py.File(path, "r") as handle:
        if variable_path.strip("/") not in handle:
            raise KeyError(f"HDF5 dataset not found: {variable_path}")
        return np.asarray(handle[variable_path.strip("/")])


def _decode_complex(array: np.ndarray, mapping: dict[str, Any]) -> np.ndarray:
    encoding = mapping["complex_encoding"]
    if encoding == "native_complex":
        if not np.iscomplexobj(array):
            raise ValueError("Expected a native complex array")
        return array.astype(np.complex128, copy=False)
    if encoding == "compound_real_imag":
        names = array.dtype.names or ()
        real_field = mapping.get("real_field", "real")
        imag_field = mapping.get("imag_field", "imag")
        if real_field not in names or imag_field not in names:
            raise ValueError("Expected compound real/imag fields")
        return np.asarray(array[real_field], dtype=float) + 1j * np.asarray(
            array[imag_field], dtype=float
        )
    if encoding == "split_axis":
        axis = int(mapping["complex_axis"])
        if array.shape[axis] != 2:
            raise ValueError("Complex split axis must have length two")
        real = np.take(array, 0, axis=axis)
        imag = np.take(array, 1, axis=axis)
        return np.asarray(real, dtype=float) + 1j * np.asarray(imag, dtype=float)
    raise ValueError(f"Unsupported complex encoding: {encoding}")


def load_complex_pulse_range(
    path: str | Path, mapping: dict[str, Any]
) -> np.ndarray:
    """Load one explicitly mapped echo matrix and return pulse x range."""

    path = Path(path)
    required = {"variable_path", "complex_encoding", "pulse_axis", "range_axis"}
    missing = required - mapping.keys()
    if missing:
        raise ValueError(f"Missing explicit MAT mapping fields: {sorted(missing)}")
    raw = load_explicit_array(path, str(mapping["variable_path"]))
    echo = _decode_complex(raw, mapping)
    if mapping.get("squeeze_singletons", False):
        echo = np.squeeze(echo)
    if echo.ndim != 2:
        raise ValueError(f"Mapped echo must be two-dimensional, got {echo.shape}")
    pulse_axis = int(mapping["pulse_axis"])
    range_axis = int(mapping["range_axis"])
    if pulse_axis == range_axis:
        raise ValueError("pulse_axis and range_axis must differ")
    echo = np.moveaxis(echo, (pulse_axis, range_axis), (0, 1))
    if echo.shape[0] < int(mapping.get("minimum_pulses", 256)):
        raise ValueError("Mapped echo contains too few pulses")
    if echo.shape[1] < int(mapping.get("minimum_range_bins", 1)):
        raise ValueError("Mapped echo contains too few range bins")
    if not np.isfinite(echo.real).all() or not np.isfinite(echo.imag).all():
        raise ValueError("Mapped echo contains non-finite values")
    return np.asarray(echo, dtype=np.complex128)
