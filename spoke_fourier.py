#!/usr/bin/env python3
"""Wafer-level spoke defect Fourier analysis.

The raw input is expected to have chip-level rows with:

    root_lot_id, wafer_id, chip_x_pos, chip_y_pos, bin_no

The requested bin_no value(s) are treated as defect bins. The selected raw
columns are read as strings first, so numeric wafer IDs and values such as W01
can coexist without Polars schema inference failures. All chip positions,
including non-defect bins, are used to normalize each wafer map to a unit-radius
disk. The defect indicator is averaged by theta bin, then high-frequency
Fourier energy is used as the spoke signal score.
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
class SpokeConfig:
    input_path: Path
    defect_bin_nos: object
    output_csv: Path = Path("spoke_fourier_output.csv")
    angular_bins: int = 360
    high_freq_min_harmonic: int = 8
    high_freq_max_harmonic: int | None = None
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
    if config.high_freq_min_harmonic < 1:
        raise ValueError("high_freq_min_harmonic must be >= 1.")
    if config.high_freq_min_harmonic > max_harmonic:
        raise ValueError(f"high_freq_min_harmonic cannot exceed {max_harmonic}.")
    if config.high_freq_max_harmonic is not None:
        if config.high_freq_max_harmonic < config.high_freq_min_harmonic:
            raise ValueError("high_freq_max_harmonic must be >= high_freq_min_harmonic.")
        if config.high_freq_max_harmonic > max_harmonic:
            raise ValueError(f"high_freq_max_harmonic cannot exceed {max_harmonic}.")
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


def _read_csv_projection(source: Path | io.StringIO, selected: list[str], projection: list[str]) -> pl.DataFrame:
    # Read selected raw columns as strings first. This avoids schema inference
    # failures when wafer_id/bin_no contains mixed numeric and string values.
    df = pl.read_csv(
        source,
        has_header=True,
        separator=",",
        columns=projection,
        schema_overrides={column: pl.Utf8 for column in projection},
        truncate_ragged_lines=True,
    )
    return df.rename({column: _normalize_column_name(column) for column in df.columns}).select(selected)


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


def _read_plain_csv(path: Path, selected: list[str]) -> pl.DataFrame:
    header = pl.read_csv(path, has_header=True, separator=",", n_rows=0, truncate_ragged_lines=True)
    if any("\x00" in column for column in header.columns):
        raise ValueError("NUL-padded column names detected; retrying with decoded text.")
    projection = _projection_columns(header.columns, selected)
    return _read_csv_projection(path, selected, projection)


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
    errors: list[str] = []
    for reader_name, reader in (("plain", _read_plain_csv), ("decoded", _read_decoded_csv)):
        try:
            return reader(path, selected)
        except Exception as exc:
            errors.append(f"{reader_name}: {exc}")
    raise ValueError(
        "Could not read/select the required delimited text columns. "
        f"Required columns: {', '.join(selected)}. Attempts: {' | '.join(errors)}"
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


def compute_spoke_signals(
    analysis_df: pl.DataFrame,
    *,
    group_cols: list[str],
    angular_bins: int,
    high_freq_min_harmonic: int,
    high_freq_max_harmonic: int | None,
    min_chips: int,
) -> pl.DataFrame:
    _validate_columns(analysis_df, [*group_cols, "_theta", "_is_defect"])

    high_harmonics = high_frequency_harmonics(
        angular_bins=angular_bins,
        min_harmonic=high_freq_min_harmonic,
        max_harmonic=high_freq_max_harmonic,
    )
    max_harmonic = max_calculable_harmonic(angular_bins)
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
        record["high_freq_min_harmonic"] = high_harmonics[0]
        record["high_freq_max_harmonic"] = high_harmonics[-1]

        if total_chip_count < min_chips:
            record["theta_bins_with_chips"] = 0
            record["theta_coverage"] = 0.0
            record["mean_theta_defect_rate"] = float("nan")
            record["std_theta_defect_rate"] = float("nan")
            record["high_freq_fourier_signal"] = float("nan")
            record["peak_high_freq_amplitude"] = float("nan")
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

        amplitudes = _fourier_amplitudes(theta, defect_rate, high_harmonics)
        dominant_harmonic = max(high_harmonics, key=lambda harmonic: amplitudes[harmonic])
        high_freq_signal = _band_rms_from_amplitudes(amplitudes.values())
        peak_amplitude = amplitudes[dominant_harmonic]

        record["high_freq_fourier_signal"] = high_freq_signal
        record["peak_high_freq_amplitude"] = peak_amplitude
        record["dominant_harmonic"] = int(dominant_harmonic)
        record["dominant_phase_rad"] = _fourier_phase(theta, defect_rate, dominant_harmonic)
        record["signal_to_noise"] = float(high_freq_signal / (record["std_theta_defect_rate"] + EPSILON))
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
        high_freq_min_harmonic=config.high_freq_min_harmonic,
        high_freq_max_harmonic=config.high_freq_max_harmonic,
        min_chips=config.min_chips,
    )
    if config.sort_by_score and result.height and "high_freq_fourier_signal" in result.columns:
        result = result.sort("high_freq_fourier_signal", descending=True, nulls_last=True)
    if write_csv:
        config.output_csv.parent.mkdir(parents=True, exist_ok=True)
        result.write_csv(config.output_csv)
    return SpokeRun(result=result, analysis_df=analysis_df, geometries=geometries, defect_bin_nos=defect_bin_nos)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compute high-frequency spoke Fourier signals from bin_no wafer maps.")
    parser.add_argument("input_csv", type=Path, help="Input chip-level comma-separated .txt or .csv path.")
    parser.add_argument("--defect-bin-nos", required=True, help="Comma-separated bin_no values treated as defects.")
    parser.add_argument("-o", "--output-csv", type=Path, default=Path("spoke_fourier_output.csv"))
    parser.add_argument("--angular-bins", type=int, default=360)
    parser.add_argument("--high-freq-min-harmonic", type=int, default=8)
    parser.add_argument("--high-freq-max-harmonic", type=int, default=None)
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
        high_freq_min_harmonic=args.high_freq_min_harmonic,
        high_freq_max_harmonic=args.high_freq_max_harmonic,
        min_chips=args.min_chips,
        group_cols=_split_columns(args.group_cols),
        x_col=args.x_col,
        y_col=args.y_col,
        bin_col=args.bin_col,
    )
    run_spoke_fourier(config, write_csv=True)


if __name__ == "__main__":
    main()
