#!/usr/bin/env python3
"""Wafer-level spoke defect Fourier analysis.

The raw input is expected to have chip-level rows with:

    root_lot_id, wafer_id, chip_x_pos, chip_y_pos, bin_no

The requested bin_no value(s) are treated as defect bins. The selected raw
columns are read as strings first, so numeric wafer IDs and values such as W01
can coexist without Polars schema inference failures. All chip positions,
including non-defect bins, are used to normalize each wafer map to a unit-radius
disk. The defect indicator is averaged by theta bin. The spoke score rewards
low-frequency energy concentration and similarity to the sinc-shaped spectrum
of a localized angular pulse, while penalizing rough broadband energy.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import io
import math
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


EPSILON = 1e-12
TEXT_SEPARATORS = (",", "\t", ";", "|")
CORE_REQUIREMENTS = {
    "numpy": "numpy==1.26.4",
    "polars": "polars==1.14.0",
}


def _install_missing_runtime_packages() -> None:
    missing = [pip_spec for module_name, pip_spec in CORE_REQUIREMENTS.items() if importlib.util.find_spec(module_name) is None]
    if not missing:
        return
    print("Installing missing Python packages:", ", ".join(missing))
    subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])


_install_missing_runtime_packages()

import numpy as np
import polars as pl


@dataclass(frozen=True)
class Geometry:
    center_x: float
    center_y: float
    pitch_x: float
    pitch_y: float
    radius: float
    chip_count: int


@dataclass(frozen=True)
class WaferMapData:
    cells: pl.DataFrame
    chip_width: float
    chip_height: float
    wafer_radius: float


@dataclass(frozen=True)
class SpokeConfig:
    input_path: Path
    defect_bin_nos: object
    output_csv: Path = Path("spoke_fourier_output.csv")
    angular_bins: int = 360
    low_freq_max_harmonic: int | None = None
    broadband_min_harmonic: int | None = None
    sinc_width_min_deg: float = 1.0
    sinc_width_max_deg: float = 45.0
    sinc_width_step_deg: float = 0.5
    min_chips: int = 20
    group_cols: tuple[str, ...] = ("root_lot_id", "wafer_id")
    x_col: str = "chip_x_pos"
    y_col: str = "chip_y_pos"
    bin_col: str = "bin_no"
    sort_by_score: bool = True


@dataclass(frozen=True)
class SpokeRun:
    result: pl.DataFrame
    analysis_df: pl.DataFrame
    geometries: dict[object, Geometry]
    defect_bin_nos: tuple[str, ...]


def _split_columns(value: str | None) -> tuple[str, ...]:
    if not value:
        return ()
    return tuple(part.strip() for part in value.split(",") if part.strip())


def normalize_bin_no_list(value: object) -> tuple[str, ...]:
    """Normalize scalar/list bin_no input to a tuple of string values."""

    if value is None:
        raise ValueError("At least one defect bin_no must be provided.")
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",") if part.strip()]
    elif isinstance(value, Iterable):
        parts = [str(part).strip() for part in value if str(part).strip()]
    else:
        parts = [str(value).strip()]
    if not parts:
        raise ValueError("At least one defect bin_no must be provided.")
    return tuple(dict.fromkeys(parts))


def _unique_columns(columns: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(columns))


def _validate_columns(df: pl.DataFrame, columns: Iterable[str]) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"Input file is missing required columns: {', '.join(missing)}")


def _validate_config(config: SpokeConfig) -> None:
    if config.angular_bins < 4:
        raise ValueError("angular_bins must be >= 4.")
    max_harmonic = max_calculable_harmonic(config.angular_bins)
    low_freq_max, broadband_min = resolve_spectrum_bands(
        angular_bins=config.angular_bins,
        low_freq_max_harmonic=config.low_freq_max_harmonic,
        broadband_min_harmonic=config.broadband_min_harmonic,
    )
    if low_freq_max >= broadband_min:
        raise ValueError("low_freq_max_harmonic must be smaller than broadband_min_harmonic.")
    if config.sinc_width_min_deg <= 0:
        raise ValueError("sinc_width_min_deg must be > 0.")
    if config.sinc_width_max_deg < config.sinc_width_min_deg:
        raise ValueError("sinc_width_max_deg must be >= sinc_width_min_deg.")
    if config.sinc_width_step_deg <= 0:
        raise ValueError("sinc_width_step_deg must be > 0.")
    if config.sinc_width_max_deg >= 360:
        raise ValueError("sinc_width_max_deg must be < 360.")
    if max_harmonic < 2:
        raise ValueError("angular_bins must provide at least two calculable harmonics.")
    if config.min_chips < 1:
        raise ValueError("min_chips must be >= 1.")


def _normalize_column_name(column: str) -> str:
    return column.replace("\x00", "").strip().lstrip("\ufeff")


def _decode_text_bytes(raw: bytes) -> str:
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        return raw.decode("utf-16")
    if raw.startswith(b"\xef\xbb\xbf"):
        return raw.decode("utf-8-sig")

    sample = raw[:4096]
    if len(sample) > 2:
        even = sample[0::2]
        odd = sample[1::2]
        even_null_ratio = even.count(0) / max(len(even), 1)
        odd_null_ratio = odd.count(0) / max(len(odd), 1)
        if odd_null_ratio > 0.2 and odd_null_ratio > even_null_ratio * 2:
            return raw.decode("utf-16-le")
        if even_null_ratio > 0.2 and even_null_ratio > odd_null_ratio * 2:
            return raw.decode("utf-16-be")

    for encoding in ("utf-8-sig", "cp949", "euc-kr"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            pass
    return raw.decode("utf-8", errors="replace")


def _projection_columns(raw_columns: Iterable[str], selected: list[str]) -> list[str]:
    normalized_to_raw = {_normalize_column_name(column): column for column in raw_columns}
    missing = [column for column in selected if column not in normalized_to_raw]
    if missing:
        available = ", ".join(normalized_to_raw)
        raise ValueError(
            f"Input file is missing required columns: {', '.join(missing)}. "
            f"Available columns after normalization: {available}"
        )
    return [normalized_to_raw[column] for column in selected]


def _read_decoded_csv_with_separator(text: str, selected: list[str], separator: str) -> pl.DataFrame:
    reader = csv.DictReader(io.StringIO(text), delimiter=separator)
    if reader.fieldnames is None:
        raise ValueError("Input file has no header row.")

    projection = _projection_columns(reader.fieldnames, selected)
    raw_to_selected = {raw_column: _normalize_column_name(raw_column) for raw_column in projection}
    records: list[dict[str, str | None]] = []
    for row in reader:
        record = {selected_column: row.get(raw_column) for raw_column, selected_column in raw_to_selected.items()}
        if any(value not in (None, "") for value in record.values()):
            records.append(record)

    schema = {column: pl.Utf8 for column in selected}
    if not records:
        return pl.DataFrame(schema=schema)
    return pl.DataFrame(records, schema=schema).select(selected)


def _read_decoded_csv(path: Path, selected: list[str]) -> pl.DataFrame:
    text = _decode_text_bytes(path.read_bytes())
    errors: list[str] = []
    for separator in TEXT_SEPARATORS:
        try:
            return _read_decoded_csv_with_separator(text, selected, separator)
        except Exception as exc:
            label = "\\t" if separator == "\t" else separator
            errors.append(f"separator={label!r}: {exc}")
    raise ValueError("Decoded text could not be parsed with supported separators. " + " | ".join(errors))


def read_selected_csv(path: Path, columns: Iterable[str]) -> pl.DataFrame:
    """Read only the requested CSV columns as strings with encoding fallback."""

    path = Path(path)
    selected = _unique_columns(columns)
    try:
        return _read_decoded_csv(path, selected)
    except Exception as exc:
        error = str(exc)

    raise ValueError(
        "Could not read/select the required delimited text columns. "
        f"Required columns: {', '.join(selected)}. Error: {error}"
    )


def _partition_items(df: pl.DataFrame, cols: list[str]) -> Iterable[tuple[object, pl.DataFrame]]:
    if not cols:
        yield "__all__", df
        return
    yield from df.partition_by(cols, as_dict=True, maintain_order=True).items()


def _key_values(key: object, cols: list[str]) -> tuple[object, ...]:
    if not cols:
        return ()
    if isinstance(key, tuple):
        return key
    return (key,)


def _infer_geometry(frame: pl.DataFrame, *, x_col: str, y_col: str) -> Geometry:
    points = frame.select(
        pl.col(x_col).cast(pl.Float64, strict=False).alias(x_col),
        pl.col(y_col).cast(pl.Float64, strict=False).alias(y_col),
    ).unique()
    x = points.get_column(x_col).to_numpy()
    y = points.get_column(y_col).to_numpy()
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    if x.size == 0:
        raise ValueError("No finite chip coordinates are available for geometry inference.")

    min_x = float(np.min(x))
    max_x = float(np.max(x))
    min_y = float(np.min(y))
    max_y = float(np.max(y))
    center_x = (min_x + max_x) / 2.0
    center_y = (min_y + max_y) / 2.0
    span_x = max(max_x - min_x, 1.0)
    span_y = max(max_y - min_y, 1.0)
    pitch_x = 1.0
    pitch_y = span_x / span_y

    dx = (x - center_x) * pitch_x
    dy = (y - center_y) * pitch_y
    radius = float(np.max(np.hypot(dx, dy)))
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError("Could not infer a positive wafer radius.")

    return Geometry(
        center_x=center_x,
        center_y=center_y,
        pitch_x=pitch_x,
        pitch_y=pitch_y,
        radius=radius,
        chip_count=int(x.size),
    )


def attach_polar_coordinates(
    df: pl.DataFrame,
    *,
    group_cols: list[str],
    x_col: str,
    y_col: str,
) -> tuple[pl.DataFrame, dict[object, Geometry]]:
    """Normalize each wafer to a unit disk using all chip positions."""

    _validate_columns(df, [*group_cols, x_col, y_col])
    parts: list[pl.DataFrame] = []
    geometries: dict[object, Geometry] = {}

    for key, frame in _partition_items(df, group_cols):
        geometry = _infer_geometry(frame, x_col=x_col, y_col=y_col)
        geometries[key] = geometry
        part = (
            frame.with_columns(
                ((pl.col(x_col).cast(pl.Float64, strict=False) - geometry.center_x) * geometry.pitch_x).alias("_dx"),
                ((pl.col(y_col).cast(pl.Float64, strict=False) - geometry.center_y) * geometry.pitch_y).alias("_dy"),
            )
            .with_columns(
                pl.arctan2(pl.col("_dy"), pl.col("_dx")).alias("_theta"),
                ((pl.col("_dx").pow(2) + pl.col("_dy").pow(2)).sqrt() / geometry.radius).alias("_radius_norm"),
            )
            .drop("_dx", "_dy")
        )
        parts.append(part)

    if not parts:
        raise ValueError("No rows are available after input filtering.")
    return pl.concat(parts, how="vertical_relaxed"), geometries


def attach_defect_indicator(df: pl.DataFrame, *, bin_col: str, defect_bin_nos: tuple[str, ...]) -> pl.DataFrame:
    _validate_columns(df, [bin_col])
    return df.with_columns(
        pl.col(bin_col)
        .cast(pl.Utf8, strict=False)
        .str.strip_chars()
        .is_in(list(defect_bin_nos))
        .cast(pl.Int8)
        .alias("_is_defect")
    )


def max_calculable_harmonic(angular_bins: int) -> int:
    if angular_bins < 2:
        raise ValueError("angular_bins must be at least 2.")
    return angular_bins // 2


def resolve_spectrum_bands(
    *,
    angular_bins: int = 360,
    low_freq_max_harmonic: int | None = None,
    broadband_min_harmonic: int | None = None,
) -> tuple[int, int]:
    """Resolve low-frequency and broadband boundaries for the angular spectrum."""

    max_harmonic = max_calculable_harmonic(angular_bins)
    default_low_max = max(1, min(max_harmonic - 1, int(round(max_harmonic * 0.40))))
    low_max = default_low_max if low_freq_max_harmonic is None else int(low_freq_max_harmonic)
    default_broadband_min = max(low_max + 1, int(round(max_harmonic * 0.50)))
    broadband_min = default_broadband_min if broadband_min_harmonic is None else int(broadband_min_harmonic)

    if low_max < 1 or low_max >= max_harmonic:
        raise ValueError(f"low_freq_max_harmonic must be in 1..{max_harmonic - 1}.")
    if broadband_min <= low_max or broadband_min > max_harmonic:
        raise ValueError(f"broadband_min_harmonic must be in {low_max + 1}..{max_harmonic}.")
    return low_max, broadband_min


def high_frequency_harmonics(
    *,
    angular_bins: int = 360,
    min_harmonic: int = 8,
    max_harmonic: int | None = None,
) -> list[int]:
    resolved_max = max_calculable_harmonic(angular_bins) if max_harmonic is None else max_harmonic
    if min_harmonic < 1 or min_harmonic > resolved_max:
        raise ValueError("Invalid high-frequency harmonic range.")
    if resolved_max > max_calculable_harmonic(angular_bins):
        raise ValueError(f"max_harmonic cannot exceed angular_bins / 2 = {angular_bins // 2}.")
    return list(range(min_harmonic, resolved_max + 1))


def _angular_bin_average(theta: np.ndarray, defect: np.ndarray, *, bin_count: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    theta01 = (theta + math.pi) / (2.0 * math.pi)
    bins = np.floor(theta01 * bin_count).astype(int) % bin_count
    counts = np.bincount(bins, minlength=bin_count).astype(float)
    sums = np.bincount(bins, weights=defect, minlength=bin_count).astype(float)
    valid = counts > 0
    centers = -math.pi + (np.arange(bin_count, dtype=float) + 0.5) * (2.0 * math.pi / bin_count)
    defect_rate = np.divide(sums, counts, out=np.zeros_like(sums), where=valid)
    return centers[valid], defect_rate[valid], counts[valid], np.nonzero(valid)[0].astype(np.int64)


def _fourier_amplitudes(theta: np.ndarray, y: np.ndarray, harmonics: list[int]) -> dict[int, float]:
    if y.size == 0:
        return {harmonic: float("nan") for harmonic in harmonics}
    centered = y - float(np.mean(y))
    amplitudes: dict[int, float] = {}
    for harmonic in harmonics:
        coefficient = np.mean(centered * np.exp(-1j * harmonic * theta))
        amplitudes[harmonic] = float(2.0 * abs(coefficient))
    return amplitudes


def _fourier_phase(theta: np.ndarray, y: np.ndarray, harmonic: int) -> float:
    if y.size == 0:
        return float("nan")
    centered = y - float(np.mean(y))
    coefficient = np.mean(centered * np.exp(-1j * harmonic * theta))
    return float(np.angle(coefficient))


def _band_rms_from_amplitudes(amplitudes: Iterable[float]) -> float:
    values = np.array([value for value in amplitudes if np.isfinite(value)], dtype=float)
    if values.size == 0:
        return float("nan")
    return float(np.sqrt(np.sum(values**2) / 2.0))


def sinc_template_amplitudes(harmonics: Iterable[int], width_deg: float) -> np.ndarray:
    """Return the magnitude shape of a rectangular angular pulse spectrum."""

    harmonic_values = np.asarray(list(harmonics), dtype=float)
    if width_deg <= 0 or width_deg >= 360:
        raise ValueError("width_deg must be in (0, 360).")
    width_rad = math.radians(width_deg)
    return np.abs(np.sin(harmonic_values * width_rad / 2.0) / harmonic_values)


def _sinc_template_bank(
    harmonics: np.ndarray,
    *,
    min_width_deg: float,
    max_width_deg: float,
    width_step_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    widths = np.arange(min_width_deg, max_width_deg + width_step_deg * 0.5, width_step_deg, dtype=float)
    templates = np.vstack([sinc_template_amplitudes(harmonics, width) for width in widths])
    norms = np.linalg.norm(templates, axis=1, keepdims=True)
    normalized = np.divide(templates, norms, out=np.zeros_like(templates), where=norms > EPSILON)
    return widths, normalized


def _score_spoke_spectrum(
    harmonics: np.ndarray,
    amplitudes: np.ndarray,
    *,
    low_freq_max_harmonic: int,
    broadband_min_harmonic: int,
    sinc_widths_deg: np.ndarray,
    normalized_sinc_templates: np.ndarray,
) -> dict[str, object]:
    values = np.where(np.isfinite(amplitudes), np.maximum(amplitudes, 0.0), 0.0)
    low_mask = harmonics <= low_freq_max_harmonic
    broadband_mask = harmonics >= broadband_min_harmonic

    total_energy = float(np.sum(values**2))
    low_energy = float(np.sum(values[low_mask] ** 2))
    broadband_energy = float(np.sum(values[broadband_mask] ** 2))
    low_signal = float(math.sqrt(low_energy / 2.0))
    broadband_signal = float(math.sqrt(broadband_energy / 2.0))
    low_energy_ratio = low_energy / (total_energy + EPSILON)
    broadband_energy_ratio = broadband_energy / (total_energy + EPSILON)

    low_rms = float(np.sqrt(np.mean(values[low_mask] ** 2)))
    broadband_rms = float(np.sqrt(np.mean(values[broadband_mask] ** 2)))
    signal_to_noise = low_rms / (broadband_rms + EPSILON)
    noise_floor = float(np.median(values[broadband_mask]))
    excess = np.maximum(values - noise_floor, 0.0)
    excess_energy = float(np.sum(excess**2))

    if excess_energy > EPSILON:
        normalized_excess = excess / math.sqrt(excess_energy)
        similarities = normalized_sinc_templates @ normalized_excess
        best_index = int(np.argmax(similarities))
        sinc_similarity = float(np.clip(similarities[best_index], 0.0, 1.0))
        estimated_width_deg = float(sinc_widths_deg[best_index])
        roughness = float(np.sum(np.diff(excess) ** 2) / excess_energy)
        smoothness = float(1.0 / (1.0 + roughness))
        raw_template = sinc_template_amplitudes(harmonics, estimated_width_deg)
        template_scale = float(np.dot(excess, raw_template) / (np.dot(raw_template, raw_template) + EPSILON))
    else:
        sinc_similarity = 0.0
        estimated_width_deg = float("nan")
        roughness = float("nan")
        smoothness = 0.0
        template_scale = 0.0

    broadband_penalty = max(0.0, 1.0 - broadband_energy_ratio)
    spoke_signal = low_signal * low_energy_ratio * sinc_similarity * smoothness * broadband_penalty

    if total_energy > EPSILON:
        dominant_index = int(np.argmax(values))
        dominant_harmonic: int | None = int(harmonics[dominant_index])
        peak_amplitude = float(values[dominant_index])
    else:
        dominant_harmonic = None
        peak_amplitude = 0.0

    broadband_values = values[broadband_mask]
    broadband_peak_amplitude = float(np.max(broadband_values)) if broadband_values.size else 0.0
    return {
        "spoke_fourier_signal": float(spoke_signal),
        "low_freq_fourier_signal": low_signal,
        "broadband_fourier_signal": broadband_signal,
        "low_freq_energy_ratio": float(low_energy_ratio),
        "broadband_energy_ratio": float(broadband_energy_ratio),
        "sinc_similarity": sinc_similarity,
        "estimated_spoke_width_deg": estimated_width_deg,
        "sinc_template_scale": template_scale,
        "spectral_noise_floor": noise_floor,
        "spectral_roughness": roughness,
        "spectral_smoothness": smoothness,
        "signal_to_noise": float(signal_to_noise),
        "dominant_harmonic": dominant_harmonic,
        "peak_harmonic_amplitude": peak_amplitude,
        "broadband_peak_amplitude": broadband_peak_amplitude,
        # Compatibility aliases retained as diagnostic broadband values.
        "high_freq_fourier_signal": broadband_signal,
        "peak_high_freq_amplitude": broadband_peak_amplitude,
    }


def build_wafer_theta_signal(
    analysis_df: pl.DataFrame,
    wafer_key: tuple[object, ...],
    *,
    group_cols: tuple[str, ...] | list[str] = ("root_lot_id", "wafer_id"),
    angular_bins: int = 360,
) -> pl.DataFrame:
    """Return theta-bin defect rate for one wafer."""

    group_cols = list(group_cols)
    _validate_columns(analysis_df, [*group_cols, "_theta", "_is_defect"])
    wafer_df = filter_wafer_rows(analysis_df, wafer_key, group_cols=group_cols)
    numeric = wafer_df.select(
        pl.col("_theta").cast(pl.Float64, strict=False),
        pl.col("_is_defect").cast(pl.Float64, strict=False),
    )
    theta_all = numeric.get_column("_theta").to_numpy()
    defect_all = numeric.get_column("_is_defect").to_numpy()
    finite = np.isfinite(theta_all) & np.isfinite(defect_all)

    if not np.any(finite):
        return pl.DataFrame(
            schema={
                "theta_bin": pl.Int64,
                "theta_rad": pl.Float64,
                "defect_rate": pl.Float64,
                "chip_count": pl.Int64,
            }
        )

    theta, defect_rate, chip_count, theta_bin = _angular_bin_average(
        theta_all[finite],
        defect_all[finite],
        bin_count=angular_bins,
    )
    return pl.DataFrame(
        {
            "theta_bin": theta_bin,
            "theta_rad": theta,
            "defect_rate": defect_rate,
            "chip_count": chip_count.astype(np.int64),
        }
    )


def compute_harmonic_spectrum(
    theta_signal_df: pl.DataFrame,
    *,
    angular_bins: int = 360,
    max_harmonic: int | None = None,
    theta_col: str = "theta_rad",
    value_col: str = "defect_rate",
) -> pl.DataFrame:
    """Compute harmonic amplitudes for one theta signal."""

    if max_harmonic is None:
        max_harmonic = max_calculable_harmonic(angular_bins)
    if max_harmonic < 1 or max_harmonic > max_calculable_harmonic(angular_bins):
        raise ValueError(f"max_harmonic must be in 1..{angular_bins // 2}.")

    _validate_columns(theta_signal_df, [theta_col, value_col])
    theta = theta_signal_df.get_column(theta_col).to_numpy()
    y = theta_signal_df.get_column(value_col).to_numpy()
    harmonics = list(range(1, max_harmonic + 1))
    amplitudes = _fourier_amplitudes(theta, y, harmonics)
    phases = {harmonic: _fourier_phase(theta, y, harmonic) for harmonic in harmonics}

    return pl.DataFrame(
        {
            "harmonic": harmonics,
            "frequency_cycles_per_revolution": harmonics,
            "amplitude": [amplitudes[harmonic] for harmonic in harmonics],
            "phase_rad": [phases[harmonic] for harmonic in harmonics],
        }
    )


def filter_wafer_rows(
    df: pl.DataFrame,
    wafer_key: tuple[object, ...],
    *,
    group_cols: tuple[str, ...] | list[str] = ("root_lot_id", "wafer_id"),
) -> pl.DataFrame:
    group_cols = list(group_cols)
    if len(wafer_key) != len(group_cols):
        raise ValueError(f"wafer_key must have {len(group_cols)} values matching {group_cols}.")
    _validate_columns(df, group_cols)

    condition = pl.lit(True)
    for column, value in zip(group_cols, wafer_key):
        condition = condition & (
            pl.col(column).cast(pl.Utf8, strict=False).str.strip_chars() == str(value).strip()
        )
    selected = df.filter(condition)
    if selected.height:
        return selected

    available = df.select(group_cols).unique().head(20)
    raise ValueError(f"No rows found for wafer_key={wafer_key!r}. Available examples:\n{available}")


def _minimum_positive_step(values: np.ndarray) -> float:
    unique_values = np.unique(values[np.isfinite(values)])
    if unique_values.size < 2:
        return 1.0
    differences = np.diff(unique_values)
    positive = differences[differences > np.finfo(float).eps]
    return float(np.min(positive)) if positive.size else 1.0


def build_wafer_map_data(
    analysis_df: pl.DataFrame,
    wafer_key: tuple[object, ...],
    *,
    group_cols: tuple[str, ...] | list[str] = ("root_lot_id", "wafer_id"),
    x_col: str = "chip_x_pos",
    y_col: str = "chip_y_pos",
) -> WaferMapData:
    """Return one normalized rectangular cell per chip for wafer-map plotting."""

    group_cols = list(group_cols)
    _validate_columns(analysis_df, [*group_cols, x_col, y_col, "_is_defect"])
    wafer_df = filter_wafer_rows(analysis_df, wafer_key, group_cols=group_cols)
    geometry = _infer_geometry(wafer_df, x_col=x_col, y_col=y_col)

    cells = (
        wafer_df.select(
            pl.col(x_col).cast(pl.Float64, strict=False).alias("chip_x"),
            pl.col(y_col).cast(pl.Float64, strict=False).alias("chip_y"),
            pl.col("_is_defect").cast(pl.Int8, strict=False).alias("is_defect"),
        )
        .filter(pl.col("chip_x").is_finite() & pl.col("chip_y").is_finite())
        .group_by(["chip_x", "chip_y"], maintain_order=True)
        .agg(pl.col("is_defect").max())
        .with_columns(
            (((pl.col("chip_x") - geometry.center_x) * geometry.pitch_x) / geometry.radius).alias("map_x"),
            (((pl.col("chip_y") - geometry.center_y) * geometry.pitch_y) / geometry.radius).alias("map_y"),
        )
        .select("chip_x", "chip_y", "map_x", "map_y", "is_defect")
    )
    if cells.is_empty():
        raise ValueError(f"No finite chip coordinates found for wafer_key={wafer_key!r}.")

    chip_width = _minimum_positive_step(cells.get_column("chip_x").to_numpy()) * geometry.pitch_x / geometry.radius
    chip_height = _minimum_positive_step(cells.get_column("chip_y").to_numpy()) * geometry.pitch_y / geometry.radius
    map_x = cells.get_column("map_x").to_numpy()
    map_y = cells.get_column("map_y").to_numpy()
    half_diagonal = 0.5 * math.hypot(chip_width, chip_height)
    wafer_radius = float(np.max(np.hypot(map_x, map_y)) + half_diagonal)

    return WaferMapData(
        cells=cells,
        chip_width=chip_width,
        chip_height=chip_height,
        wafer_radius=wafer_radius,
    )


def compute_spoke_signals(
    analysis_df: pl.DataFrame,
    *,
    group_cols: list[str],
    angular_bins: int,
    low_freq_max_harmonic: int | None = None,
    broadband_min_harmonic: int | None = None,
    sinc_width_min_deg: float = 1.0,
    sinc_width_max_deg: float = 45.0,
    sinc_width_step_deg: float = 0.5,
    min_chips: int = 20,
) -> pl.DataFrame:
    _validate_columns(analysis_df, [*group_cols, "_theta", "_is_defect"])
    if sinc_width_min_deg <= 0 or sinc_width_max_deg < sinc_width_min_deg or sinc_width_max_deg >= 360:
        raise ValueError("sinc template widths must satisfy 0 < min <= max < 360 degrees.")
    if sinc_width_step_deg <= 0:
        raise ValueError("sinc_width_step_deg must be > 0.")
    if min_chips < 1:
        raise ValueError("min_chips must be >= 1.")

    low_freq_max, broadband_min = resolve_spectrum_bands(
        angular_bins=angular_bins,
        low_freq_max_harmonic=low_freq_max_harmonic,
        broadband_min_harmonic=broadband_min_harmonic,
    )
    max_harmonic = max_calculable_harmonic(angular_bins)
    harmonics = np.arange(1, max_harmonic + 1, dtype=int)
    harmonic_list = harmonics.tolist()
    sinc_widths, sinc_templates = _sinc_template_bank(
        harmonics,
        min_width_deg=sinc_width_min_deg,
        max_width_deg=sinc_width_max_deg,
        width_step_deg=sinc_width_step_deg,
    )
    records: list[dict[str, object]] = []

    for key, group in _partition_items(analysis_df, group_cols):
        record: dict[str, object] = dict(zip(group_cols, _key_values(key, group_cols)))
        numeric = group.select(
            pl.col("_theta").cast(pl.Float64, strict=False),
            pl.col("_is_defect").cast(pl.Float64, strict=False),
        )
        theta_all = numeric.get_column("_theta").to_numpy()
        defect_all = numeric.get_column("_is_defect").to_numpy()
        finite = np.isfinite(theta_all) & np.isfinite(defect_all)
        total_chip_count = int(np.count_nonzero(finite))
        defect_chip_count = int(np.sum(defect_all[finite])) if total_chip_count else 0

        record["total_chip_count"] = total_chip_count
        record["defect_chip_count"] = defect_chip_count
        record["defect_rate"] = float(defect_chip_count / total_chip_count) if total_chip_count else 0.0
        record["angular_bins"] = angular_bins
        record["max_calculable_harmonic"] = max_harmonic
        record["low_freq_max_harmonic"] = low_freq_max
        record["broadband_min_harmonic"] = broadband_min
        record["broadband_max_harmonic"] = max_harmonic
        record["high_freq_min_harmonic"] = broadband_min
        record["high_freq_max_harmonic"] = max_harmonic

        if total_chip_count < min_chips:
            record["theta_bins_with_chips"] = 0
            record["theta_coverage"] = 0.0
            record["mean_theta_defect_rate"] = float("nan")
            record["std_theta_defect_rate"] = float("nan")
            record["spoke_fourier_signal"] = float("nan")
            record["low_freq_fourier_signal"] = float("nan")
            record["broadband_fourier_signal"] = float("nan")
            record["low_freq_energy_ratio"] = float("nan")
            record["broadband_energy_ratio"] = float("nan")
            record["sinc_similarity"] = float("nan")
            record["estimated_spoke_width_deg"] = float("nan")
            record["sinc_template_scale"] = float("nan")
            record["spectral_noise_floor"] = float("nan")
            record["spectral_roughness"] = float("nan")
            record["spectral_smoothness"] = float("nan")
            record["high_freq_fourier_signal"] = float("nan")
            record["peak_high_freq_amplitude"] = float("nan")
            record["peak_harmonic_amplitude"] = float("nan")
            record["broadband_peak_amplitude"] = float("nan")
            record["dominant_harmonic"] = None
            record["dominant_phase_rad"] = float("nan")
            record["signal_to_noise"] = float("nan")
            records.append(record)
            continue

        theta, defect_rate, chip_count, _ = _angular_bin_average(
            theta_all[finite],
            defect_all[finite],
            bin_count=angular_bins,
        )
        record["theta_bins_with_chips"] = int(chip_count.size)
        record["theta_coverage"] = float(chip_count.size / angular_bins)
        record["mean_theta_defect_rate"] = float(np.mean(defect_rate)) if defect_rate.size else float("nan")
        record["std_theta_defect_rate"] = float(np.std(defect_rate)) if defect_rate.size else float("nan")

        amplitude_map = _fourier_amplitudes(theta, defect_rate, harmonic_list)
        amplitude_values = np.array([amplitude_map[harmonic] for harmonic in harmonic_list], dtype=float)
        spectrum_metrics = _score_spoke_spectrum(
            harmonics,
            amplitude_values,
            low_freq_max_harmonic=low_freq_max,
            broadband_min_harmonic=broadband_min,
            sinc_widths_deg=sinc_widths,
            normalized_sinc_templates=sinc_templates,
        )
        record.update(spectrum_metrics)
        dominant_harmonic = spectrum_metrics["dominant_harmonic"]
        record["dominant_phase_rad"] = (
            _fourier_phase(theta, defect_rate, int(dominant_harmonic))
            if dominant_harmonic is not None
            else float("nan")
        )
        records.append(record)

    return pl.DataFrame(records)


def run_spoke_fourier(config: SpokeConfig, *, write_csv: bool = True) -> SpokeRun:
    _validate_config(config)
    defect_bin_nos = normalize_bin_no_list(config.defect_bin_nos)
    group_cols = list(config.group_cols)
    requested_columns = _unique_columns([*group_cols, config.x_col, config.y_col, config.bin_col])

    df = read_selected_csv(Path(config.input_path), requested_columns)
    df = attach_defect_indicator(df, bin_col=config.bin_col, defect_bin_nos=defect_bin_nos)
    analysis_df, geometries = attach_polar_coordinates(
        df,
        group_cols=group_cols,
        x_col=config.x_col,
        y_col=config.y_col,
    )
    result = compute_spoke_signals(
        analysis_df,
        group_cols=group_cols,
        angular_bins=config.angular_bins,
        low_freq_max_harmonic=config.low_freq_max_harmonic,
        broadband_min_harmonic=config.broadband_min_harmonic,
        sinc_width_min_deg=config.sinc_width_min_deg,
        sinc_width_max_deg=config.sinc_width_max_deg,
        sinc_width_step_deg=config.sinc_width_step_deg,
        min_chips=config.min_chips,
    )
    if config.sort_by_score and result.height and "spoke_fourier_signal" in result.columns:
        result = result.sort("spoke_fourier_signal", descending=True, nulls_last=True)
    if write_csv:
        config.output_csv.parent.mkdir(parents=True, exist_ok=True)
        result.write_csv(config.output_csv)
    return SpokeRun(result=result, analysis_df=analysis_df, geometries=geometries, defect_bin_nos=defect_bin_nos)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compute low-frequency coherent spoke Fourier signals from bin_no wafer maps.")
    parser.add_argument("input_csv", type=Path, help="Input chip-level .txt or .csv path. Comma and tab delimiters are both tried.")
    parser.add_argument("--defect-bin-nos", required=True, help="Comma-separated bin_no values treated as defects.")
    parser.add_argument("-o", "--output-csv", type=Path, default=Path("spoke_fourier_output.csv"))
    parser.add_argument("--angular-bins", type=int, default=360)
    parser.add_argument("--low-freq-max-harmonic", type=int, default=None)
    parser.add_argument("--broadband-min-harmonic", type=int, default=None)
    parser.add_argument("--sinc-width-min-deg", type=float, default=1.0)
    parser.add_argument("--sinc-width-max-deg", type=float, default=45.0)
    parser.add_argument("--sinc-width-step-deg", type=float, default=0.5)
    parser.add_argument("--min-chips", type=int, default=20)
    parser.add_argument("--group-cols", default="root_lot_id,wafer_id")
    parser.add_argument("--x-col", default="chip_x_pos")
    parser.add_argument("--y-col", default="chip_y_pos")
    parser.add_argument("--bin-col", default="bin_no")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    config = SpokeConfig(
        input_path=args.input_csv,
        defect_bin_nos=_split_columns(args.defect_bin_nos),
        output_csv=args.output_csv,
        angular_bins=args.angular_bins,
        low_freq_max_harmonic=args.low_freq_max_harmonic,
        broadband_min_harmonic=args.broadband_min_harmonic,
        sinc_width_min_deg=args.sinc_width_min_deg,
        sinc_width_max_deg=args.sinc_width_max_deg,
        sinc_width_step_deg=args.sinc_width_step_deg,
        min_chips=args.min_chips,
        group_cols=_split_columns(args.group_cols),
        x_col=args.x_col,
        y_col=args.y_col,
        bin_col=args.bin_col,
    )
    run_spoke_fourier(config, write_csv=True)


if __name__ == "__main__":
    main()
