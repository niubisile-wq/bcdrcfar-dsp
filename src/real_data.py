"""Readers and leakage-safe window helpers for public radar datasets.

The IPIX implementation follows the processing equations published with the
McMaster Dartmouth data.  Range-bin arguments are one-based to match the
official data table and MATLAB helper.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
from scipy.io import netcdf_file


IPIX_PILOT_METADATA = {
    "017": {"primary": 9, "secondary": tuple(range(8, 12))},
    "018": {"primary": 9, "secondary": tuple(range(8, 12))},
    "025": {"primary": 7, "secondary": tuple(range(6, 9))},
    "054": {"primary": 8, "secondary": tuple(range(7, 11))},
    "320": {"primary": 7, "secondary": tuple(range(6, 10))},
}


@dataclass(frozen=True)
class IpixSeries:
    samples: np.ndarray
    polarization: str
    range_bin: int
    mean_iq: tuple[float, float]
    std_iq: tuple[float, float]
    phase_imbalance_deg: float
    metadata: dict[str, object]


@dataclass(frozen=True)
class IpixReferenceTransform:
    """Common I/Q correction fitted only from designated clutter bins.

    Unlike per-range-bin standardization, this transform preserves amplitude
    contrasts between the CUT and reference cells.  That property is required
    for a scale-equivariant CFAR detector.
    """

    mean_iq: tuple[float, float]
    std_iq: tuple[float, float]
    sin_phase_imbalance: float


def fit_ipix_reference_transform(samples: Sequence[np.ndarray]) -> IpixReferenceTransform:
    """Fit one acquisition/polarization transform from clutter-only series."""

    values = [np.asarray(item) for item in samples]
    if not values or any(item.ndim != 1 or not np.iscomplexobj(item) for item in values):
        raise ValueError("samples must contain one or more complex one-dimensional series")
    pooled = np.concatenate(values)
    if not np.isfinite(pooled).all():
        raise ValueError("reference samples must be finite")
    mean_iq = (float(pooled.real.mean()), float(pooled.imag.mean()))
    std_iq = (float(pooled.real.std(ddof=0)), float(pooled.imag.std(ddof=0)))
    if min(std_iq) <= 1e-12:
        raise ValueError("Degenerate reference I/Q channel")
    i = (pooled.real - mean_iq[0]) / std_iq[0]
    q = (pooled.imag - mean_iq[1]) / std_iq[1]
    sin_beta = float(np.clip(np.mean(i * q), -0.999999, 0.999999))
    return IpixReferenceTransform(mean_iq, std_iq, sin_beta)


def apply_ipix_reference_transform(
    samples: np.ndarray, transform: IpixReferenceTransform
) -> np.ndarray:
    """Apply a clutter-fitted common transform without range-wise rescaling."""

    values = np.asarray(samples)
    if values.ndim != 1 or not np.iscomplexobj(values):
        raise ValueError("samples must be a one-dimensional complex series")
    i = (values.real.astype(float, copy=False) - transform.mean_iq[0]) / transform.std_iq[0]
    q = (values.imag.astype(float, copy=False) - transform.mean_iq[1]) / transform.std_iq[1]
    beta = transform.sin_phase_imbalance
    i = (i - q * beta) / np.sqrt(1.0 - beta**2)
    output = np.asarray(i + 1j * q, dtype=np.complex128)
    if not np.isfinite(output).all():
        raise ValueError("Non-finite values after common I/Q preprocessing")
    return output


def _scalar(variable) -> int:
    return int(np.asarray(variable.data).reshape(-1)[0])


def _decode(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("ascii", errors="replace").rstrip("\x00")
    return str(value)


def _frequency_ghz(variable) -> float:
    value = float(np.asarray(variable.data))
    units = _decode(variable._attributes.get("units", "")).strip().lower()
    if units == "ghz":
        return value
    if units == "mhz":
        return value / 1e3
    if units == "khz":
        return value / 1e6
    if units == "hz":
        return value / 1e9
    raise ValueError(f"Unsupported RF_frequency units: {units!r}")


def load_ipix_series(
    path: str | Path,
    polarization: str,
    range_bin: int,
    preprocess: str = "auto",
) -> IpixSeries:
    """Load one complex slow-time IPIX series.

    ``polarization`` is one of ``hh``, ``hv``, ``vv`` or ``vh``.  ``auto``
    removes the I/Q means and marginal scales and corrects phase imbalance,
    exactly as the public ``ipixload.m`` routine specifies for stare data.
    """

    path = Path(path)
    pol = polarization.lower()
    if pol not in {"hh", "hv", "vv", "vh"}:
        raise ValueError(f"Unsupported polarization: {polarization}")
    if preprocess not in {"raw", "auto"}:
        raise ValueError("preprocess must be 'raw' or 'auto'")

    with netcdf_file(path, "r", mmap=False) as nc:
        adc = nc.variables["adc_data"]
        shape = adc.shape
        nrange = int(shape[-2])
        if not 1 <= int(range_bin) <= nrange:
            raise ValueError(f"range_bin must be between 1 and {nrange}")

        like_i = _scalar(nc.variables["adc_like_I"])
        like_q = _scalar(nc.variables["adc_like_Q"])
        cross_i = _scalar(nc.variables["adc_cross_I"])
        cross_q = _scalar(nc.variables["adc_cross_Q"])
        same_pol = pol in {"hh", "vv"}
        i_chan, q_chan = (like_i, like_q) if same_pol else (cross_i, cross_q)

        tx_attr = _decode(nc._attributes.get("TX_polarization", ""))
        ridx = int(range_bin) - 1
        if len(shape) == 4:
            tx_idx = 0 if pol[0] == "h" else 1
            raw_i = np.asarray(adc.data[:, tx_idx, ridx, i_chan])
            raw_q = np.asarray(adc.data[:, tx_idx, ridx, q_chan])
        elif len(shape) == 3:
            if tx_attr.lower() and pol[0] != tx_attr.lower()[0]:
                raise ValueError(f"File contains transmit polarization {tx_attr}, not {pol[0]}")
            raw_i = np.asarray(adc.data[:, ridx, i_chan])
            raw_q = np.asarray(adc.data[:, ridx, q_chan])
        else:
            raise ValueError(f"Unexpected adc_data shape: {shape}")

        # Some files omit the unsigned flag.  Preserve the underlying byte.
        def to_unsigned_float(x: np.ndarray) -> np.ndarray:
            if x.dtype.itemsize == 1 and np.issubdtype(x.dtype, np.signedinteger):
                return x.view(np.uint8).astype(np.float64)
            return x.astype(np.float64, copy=False)

        i = to_unsigned_float(raw_i)
        q = to_unsigned_float(raw_q)
        metadata = {
            "file": path.name,
            "date": _decode(nc._attributes.get("Data_collection_date", "")),
            "site": _decode(nc._attributes.get("Site", "")),
            "tx_polarization": tx_attr,
            "prf_hz": float(np.asarray(nc.variables["PRF"].data)),
            "rf_frequency_ghz": _frequency_ghz(nc.variables["RF_frequency"]),
            "nrange": nrange,
            "nsweep": int(shape[0]),
        }

    if preprocess == "raw":
        mean_iq = (0.0, 0.0)
        std_iq = (1.0, 1.0)
        phase_deg = 0.0
    else:
        mean_iq = (float(i.mean()), float(q.mean()))
        std_iq = (float(i.std(ddof=0)), float(q.std(ddof=0)))
        if min(std_iq) <= 0:
            raise ValueError("Degenerate I/Q channel")
        i = (i - mean_iq[0]) / std_iq[0]
        q = (q - mean_iq[1]) / std_iq[1]
        sin_beta = float(np.clip(np.mean(i * q), -0.999999, 0.999999))
        phase_deg = float(np.degrees(np.arcsin(sin_beta)))
        i = (i - q * sin_beta) / np.sqrt(1.0 - sin_beta**2)

    return IpixSeries(
        samples=np.asarray(i + 1j * q, dtype=np.complex128),
        polarization=pol,
        range_bin=int(range_bin),
        mean_iq=mean_iq,
        std_iq=std_iq,
        phase_imbalance_deg=phase_deg,
        metadata=metadata,
    )


def nonoverlapping_windows(
    samples: np.ndarray,
    length: int,
    *,
    max_windows: int | None = None,
    offset: int = 0,
) -> np.ndarray:
    """Return leakage-safe, non-overlapping windows from a 1-D series."""

    x = np.asarray(samples)
    if x.ndim != 1 or length <= 0 or offset < 0:
        raise ValueError("samples must be 1-D and length/offset must be valid")
    count = max(0, (x.size - offset) // length)
    if max_windows is not None:
        count = min(count, int(max_windows))
    if count == 0:
        return np.empty((0, length), dtype=x.dtype)
    return x[offset : offset + count * length].reshape(count, length)


def iter_ipix_clutter_windows(
    path: str | Path,
    *,
    target_bins: Sequence[int],
    polarizations: Sequence[str] = ("hh", "vv", "hv", "vh"),
    length: int = 128,
    max_windows_per_series: int | None = None,
) -> Iterator[tuple[np.ndarray, dict[str, object]]]:
    """Yield amplitude windows from all non-target range bins in one file."""

    path = Path(path)
    with netcdf_file(path, "r", mmap=False) as nc:
        nrange = int(nc.variables["adc_data"].shape[-2])
    excluded = {int(v) for v in target_bins}
    for pol in polarizations:
        for rbin in range(1, nrange + 1):
            if rbin in excluded:
                continue
            series = load_ipix_series(path, pol, rbin, preprocess="auto")
            windows = nonoverlapping_windows(
                np.abs(series.samples), length, max_windows=max_windows_per_series
            )
            for index, window in enumerate(windows):
                yield window, {
                    "file": path.name,
                    "polarization": pol,
                    "range_bin": rbin,
                    "window": index,
                }


def target_and_clutter_amplitudes(
    path: str | Path,
    *,
    polarization: str,
    primary_bin: int,
    clutter_bins: Sequence[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Load one target-bin series and concatenate selected clutter bins."""

    target = np.abs(load_ipix_series(path, polarization, primary_bin).samples)
    clutter = np.concatenate(
        [np.abs(load_ipix_series(path, polarization, b).samples) for b in clutter_bins]
    )
    return target, clutter
