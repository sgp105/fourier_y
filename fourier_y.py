#!/usr/bin/env python3
"""Wafer-level annular Fourier y-value extraction.

This module reads chip-level comma-separated text/CSV exports, normalizes each
wafer map to polar coordinates, builds an annular angular y(theta) signal, and
extracts the requested Fourier harmonic as a wafer-level score.
"""

from __future__ import annotations

import argparse
import importlib.util
import io
import math
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


EPSILON = 1e-12
CORE_REQUIREMENTS = {
    "numpy": "numpy==1.26.4",
    "polars": "polars==1.14.0",
}


def _install_missing_runtime_packages() -> None:
    """Install required runtime packages when they are missing."""

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
class FourierConfig:
    input_path: Path
    output_csv: Path = Path("fourier_y_output.csv")
    inner_radius: float = 0.6
    outer_radius: float = 1.0
    target_harmonic: int | None = 16
    harmonics: tuple[int, ...] | None = None
    angular_bins: int = 384
    min_ring_chips: int = 20
    group_cols: tuple[str, ...] = ("root_lot_id", "wafer_id")
    geometry_cols: tuple[str, ...] = ("item_id",)
    item_id_col: str = "item_id"
    item_id_value: str | None = None
    x_col: str = "chip_x_pos"
    y_col: str = "chip_y_pos"
    value_col: str = "y_value"
    update_col: str | None = "last_update_time"
    pitch_x: float | None = None
    pitch_y: float | None = None
    auto_aspect: bool = True
    sort_by_score: bool = True


@dataclass(frozen=True)
class FourierRun:
    result: pl.DataFrame
    polar_df: pl.DataFrame
    geometries: dict[object, Geometry]


def _split_columns(value: str | None) -> tuple[str, ...]:
    if not value:
        return ()
    return tuple(part.strip() for part in value.split(",") if part.strip())


def _unique_columns(columns: Iterable[str | None]) -> list[str]:
    return list(dict.fromkeys(column for column in columns if column))


def _validate_columns(df: pl.DataFrame, columns: Iterable[str]) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"Input file is missing required columns: {', '.join(missing)}")


def _validate_config(config: FourierConfig) -> None:
    if not (0.0 <= config.inner_radius < config.outer_radius):
        raise ValueError("inner_radius must be non-negative and smaller than outer_radius.")
    if config.outer_radius > 1.2:
        raise ValueError("outer_radius is normalized to wafer radius and should usually be <= 1.0.")
    if config.angular_bins < 0:
        raise ValueError("angular_bins must be >= 0. Use 0 for direct chip-level coefficients.")
    if config.min_ring_chips < 1:
        raise ValueError("min_ring_chips must be >= 1.")
    if config.target_harmonic is not None and config.target_harmonic < 1:
        raise ValueError("target_harmonic must be a positive integer or None.")


def _resolve_harmonics(config: FourierConfig) -> list[int]:
    harmonics = list(config.harmonics or ())
    if config.target_harmonic is not None:
        harmonics.append(config.target_harmonic)
    harmonics = sorted(set(harmonics))
    if not harmonics:
        raise ValueError("At least one harmonic is required.")
    if min(harmonics) < 1:
        raise ValueError("Fourier harmonics must be positive integers.")
    return harmonics


def _normalize_column_name(column: str) -> str:
    return column.replace("\x00", "").strip().lstrip("\ufeff")


def _decode_text_bytes(raw: bytes) -> str:
    """Decode CSV-like text, including UTF-16 exports with NUL-padded headers."""

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


def _read_plain_csv(path: Path) -> pl.DataFrame:
    return pl.read_csv(path, has_header=True, separator=",")


def _read_decoded_csv(path: Path) -> pl.DataFrame:
    text = _decode_text_bytes(path.read_bytes())
    return pl.read_csv(io.StringIO(text), has_header=True, separator=",")


def read_selected_csv(path: Path, columns: Iterable[str]) -> pl.DataFrame:
    """Read comma-separated .txt/.csv and keep only selected columns."""

    selected = _unique_columns(columns)
    errors: list[str] = []
    for reader_name, reader in (("plain", _read_plain_csv), ("decoded", _read_decoded_csv)):
        try:
            df = reader(path)
            if reader_name == "plain" and any("\x00" in column for column in df.columns):
                raise ValueError("NUL-padded column names detected; retrying with decoded text.")
            df = df.rename({column: _normalize_column_name(column) for column in df.columns})
            _validate_columns(df, selected)
            return df.select(selected)
        except Exception as exc:
            errors.append(f"{reader_name}: {exc}")

    raise ValueError(
        "Could not read/select the required comma-separated columns with Polars. "
        f"Required columns: {', '.join(selected)}. Attempts: {' | '.join(errors)}"
    )


def filter_latest_update_rows(df: pl.DataFrame, *, group_cols: list[str], update_col: str | None) -> pl.DataFrame:
    """Keep only the latest update snapshot per wafer."""

    if not update_col:
        return df
    _validate_columns(df, [*group_cols, update_col])
    working = df.with_columns(
        pl.col(update_col).cast(pl.Utf8, strict=False).str.strip_chars().alias("_update_text")
    ).with_columns(
        pl.col("_update_text").str.to_datetime(strict=False, exact=False).alias("_update_dt")
    )
    parsed_count = working.select(pl.col("_update_dt").is_not_null().sum()).item()
    update_key = "_update_dt" if parsed_count else "_update_text"
    latest = working.group_by(group_cols).agg(pl.col(update_key).max().alias("_latest_update_key"))
    return (
        working.join(latest, on=group_cols, how="left")
        .filter(pl.col(update_key) == pl.col("_latest_update_key"))
        .drop("_update_text", "_update_dt", "_latest_update_key")
    )


def filter_item_id_rows(df: pl.DataFrame, *, item_id_col: str, item_id_value: str | None) -> pl.DataFrame:
    """Keep only rows matching the requested item_id string value."""

    if item_id_value is None or str(item_id_value).strip() == "":
        return df

    _validate_columns(df, [item_id_col])
    requested_item_id = str(item_id_value).strip()
    filtered = df.filter(
        pl.col(item_id_col).cast(pl.Utf8, strict=False).str.strip_chars() == requested_item_id
    )
    if filtered.height:
        return filtered

    available = (
        df.select(pl.col(item_id_col).cast(pl.Utf8, strict=False).str.strip_chars().alias(item_id_col))
        .get_column(item_id_col)
        .drop_nulls()
        .unique()
        .sort()
        .head(20)
        .to_list()
    )
    raise ValueError(
        f"No rows remain after filtering {item_id_col} == {requested_item_id!r}. "
        f"Available examples: {available}"
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


def _infer_geometry(
    frame: pl.DataFrame,
    *,
    x_col: str,
    y_col: str,
    pitch_x: float | None,
    pitch_y: float | None,
    auto_aspect: bool,
) -> Geometry:
    """Infer center, aspect scaling, and wafer radius from occupied chips."""

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

    if pitch_x is None and pitch_y is None:
        resolved_pitch_x = 1.0
        resolved_pitch_y = span_x / span_y if auto_aspect else 1.0
    elif pitch_x is None:
        resolved_pitch_y = float(pitch_y)
        resolved_pitch_x = resolved_pitch_y * span_y / span_x if auto_aspect else resolved_pitch_y
    elif pitch_y is None:
        resolved_pitch_x = float(pitch_x)
        resolved_pitch_y = resolved_pitch_x * span_x / span_y if auto_aspect else resolved_pitch_x
    else:
        resolved_pitch_x = float(pitch_x)
        resolved_pitch_y = float(pitch_y)

    dx = (x - center_x) * resolved_pitch_x
    dy = (y - center_y) * resolved_pitch_y
    radius = float(np.max(np.hypot(dx, dy)))
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError("Could not infer a positive wafer radius.")

    return Geometry(
        center_x=center_x,
        center_y=center_y,
        pitch_x=resolved_pitch_x,
        pitch_y=resolved_pitch_y,
        radius=radius,
        chip_count=int(x.size),
    )


def attach_polar_coordinates(
    df: pl.DataFrame,
    *,
    geometry_cols: list[str],
    x_col: str,
    y_col: str,
    pitch_x: float | None,
    pitch_y: float | None,
    auto_aspect: bool,
) -> tuple[pl.DataFrame, dict[object, Geometry]]:
    _validate_columns(df, [x_col, y_col, *geometry_cols])

    parts: list[pl.DataFrame] = []
    geometries: dict[object, Geometry] = {}
    for key, frame in _partition_items(df, geometry_cols):
        geometry = _infer_geometry(
            frame,
            x_col=x_col,
            y_col=y_col,
            pitch_x=pitch_x,
            pitch_y=pitch_y,
            auto_aspect=auto_aspect,
        )
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


def _angular_bin_average(theta: np.ndarray, y: np.ndarray, *, bin_count: int) -> tuple[np.ndarray, np.ndarray]:
    if bin_count <= 0:
        return theta, y

    theta01 = (theta + math.pi) / (2.0 * math.pi)
    bins = np.floor(theta01 * bin_count).astype(int) % bin_count
    counts = np.bincount(bins, minlength=bin_count).astype(float)
    sums = np.bincount(bins, weights=y, minlength=bin_count).astype(float)
    valid = counts > 0
    if not np.any(valid):
        return np.array([], dtype=float), np.array([], dtype=float)

    centers = -math.pi + (np.arange(bin_count, dtype=float) + 0.5) * (2.0 * math.pi / bin_count)
    averaged = np.divide(sums, counts, out=np.zeros_like(sums), where=valid)
    return centers[valid], averaged[valid]


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
    centered = y - float(np.mean(y))
    coefficient = np.mean(centered * np.exp(-1j * harmonic * theta))
    return float(np.angle(coefficient))


def compute_wafer_signals(
    df: pl.DataFrame,
    *,
    group_cols: list[str],
    y_col: str,
    update_col: str | None,
    inner_radius: float,
    outer_radius: float,
    harmonics: list[int],
    target_harmonic: int | None,
    angular_bins: int,
    min_ring_chips: int,
) -> pl.DataFrame:
    _validate_columns(df, [*group_cols, y_col, "_theta", "_radius_norm"])

    records: list[dict[str, object]] = []
    for key, group in _partition_items(df, group_cols):
        record: dict[str, object] = dict(zip(group_cols, _key_values(key, group_cols)))

        numeric = group.select(
            pl.col(y_col).cast(pl.Float64, strict=False).alias("_y"),
            pl.col("_theta").cast(pl.Float64, strict=False),
            pl.col("_radius_norm").cast(pl.Float64, strict=False),
        )
        y_all = numeric.get_column("_y").to_numpy()
        theta_all = numeric.get_column("_theta").to_numpy()
        radius_all = numeric.get_column("_radius_norm").to_numpy()
        finite = np.isfinite(y_all) & np.isfinite(theta_all) & np.isfinite(radius_all)
        total_chip_count = int(np.count_nonzero(finite))

        ring = finite & (radius_all >= inner_radius) & (radius_all <= outer_radius)
        ring_chip_count = int(np.count_nonzero(ring))
        record["total_chip_count"] = total_chip_count
        record["ring_chip_count"] = ring_chip_count
        record["ring_coverage"] = float(ring_chip_count / total_chip_count) if total_chip_count else 0.0

        if "item_id" in group.columns:
            item_ids = sorted(str(value) for value in group.get_column("item_id").drop_nulls().unique().to_list())
            record["item_id"] = item_ids[0] if len(item_ids) == 1 else ";".join(item_ids[:8])
        if update_col and update_col in group.columns:
            update_values = sorted(str(value) for value in group.get_column(update_col).drop_nulls().unique().to_list())
            record[update_col] = update_values[-1] if update_values else None

        if ring_chip_count < min_ring_chips:
            record["fourier_y_value"] = float("nan")
            record["dominant_harmonic"] = None
            record["dominant_phase_rad"] = float("nan")
            record["ring_mean_y"] = float("nan")
            record["ring_std_y"] = float("nan")
            record["signal_to_noise"] = float("nan")
            record["snr_weighted_fourier_y"] = float("nan")
            for harmonic in harmonics:
                record[f"harmonic_{harmonic}"] = float("nan")
            records.append(record)
            continue

        theta = theta_all[ring]
        y = y_all[ring]
        theta_signal, y_signal = _angular_bin_average(theta, y, bin_count=angular_bins)
        if y_signal.size < min(len(harmonics) * 2 + 1, min_ring_chips):
            theta_signal, y_signal = theta, y

        amplitudes = _fourier_amplitudes(theta_signal, y_signal, harmonics)
        for harmonic, amplitude in amplitudes.items():
            record[f"harmonic_{harmonic}"] = amplitude

        if target_harmonic is not None:
            if target_harmonic not in amplitudes:
                raise ValueError("target_harmonic must be included in harmonics.")
            selected_harmonic = target_harmonic
        else:
            selected_harmonic = max(amplitudes, key=lambda harmonic: amplitudes[harmonic])

        selected_amplitude = amplitudes[selected_harmonic]
        ring_std_y = float(np.std(y_signal))
        signal_to_noise = float(selected_amplitude / (ring_std_y + EPSILON))
        record["fourier_y_value"] = selected_amplitude
        record["dominant_harmonic"] = int(selected_harmonic)
        record["dominant_phase_rad"] = _fourier_phase(theta_signal, y_signal, selected_harmonic)
        record["ring_mean_y"] = float(np.mean(y_signal))
        record["ring_std_y"] = ring_std_y
        record["signal_to_noise"] = signal_to_noise
        record["snr_weighted_fourier_y"] = float(selected_amplitude * signal_to_noise)
        records.append(record)

    return pl.DataFrame(records)


def run_fourier_y(config: FourierConfig, *, write_csv: bool = True) -> FourierRun:
    """Run the full wafer Fourier workflow from input file to result table."""

    _validate_config(config)
    harmonics = _resolve_harmonics(config)
    group_cols = list(config.group_cols)
    geometry_cols = list(config.geometry_cols)
    item_id_filter_col = (
        config.item_id_col
        if config.item_id_value is not None and str(config.item_id_value).strip() != ""
        else None
    )
    requested_columns = _unique_columns(
        [
            *group_cols,
            *geometry_cols,
            item_id_filter_col,
            config.x_col,
            config.y_col,
            config.value_col,
            config.update_col,
        ]
    )

    df = read_selected_csv(Path(config.input_path), requested_columns)
    df = filter_item_id_rows(df, item_id_col=config.item_id_col, item_id_value=config.item_id_value)
    df = filter_latest_update_rows(df, group_cols=group_cols, update_col=config.update_col)
    polar_df, geometries = attach_polar_coordinates(
        df,
        geometry_cols=geometry_cols,
        x_col=config.x_col,
        y_col=config.y_col,
        pitch_x=config.pitch_x,
        pitch_y=config.pitch_y,
        auto_aspect=config.auto_aspect,
    )
    result = compute_wafer_signals(
        polar_df,
        group_cols=group_cols,
        y_col=config.value_col,
        update_col=config.update_col,
        inner_radius=config.inner_radius,
        outer_radius=config.outer_radius,
        harmonics=harmonics,
        target_harmonic=config.target_harmonic,
        angular_bins=config.angular_bins,
        min_ring_chips=config.min_ring_chips,
    )
    if config.sort_by_score and result.height and "snr_weighted_fourier_y" in result.columns:
        result = result.sort("snr_weighted_fourier_y", descending=True, nulls_last=True)
    if write_csv:
        config.output_csv.parent.mkdir(parents=True, exist_ok=True)
        result.write_csv(config.output_csv)
    return FourierRun(result=result, polar_df=polar_df, geometries=geometries)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compute wafer-level annular Fourier y-value signals.")
    parser.add_argument("input_csv", type=Path, help="Input chip-level comma-separated .txt or .csv path.")
    parser.add_argument("-o", "--output-csv", type=Path, default=Path("fourier_y_output.csv"))
    parser.add_argument("--inner-radius", type=float, default=0.6)
    parser.add_argument("--outer-radius", type=float, default=1.0)
    parser.add_argument("--target-harmonic", type=int, default=16)
    parser.add_argument("--harmonics", default=None, help="Optional comma-separated extra harmonics to output.")
    parser.add_argument("--angular-bins", type=int, default=384)
    parser.add_argument("--min-ring-chips", type=int, default=20)
    parser.add_argument("--group-cols", default="root_lot_id,wafer_id")
    parser.add_argument("--geometry-cols", default="item_id")
    parser.add_argument("--item-id-col", default="item_id")
    parser.add_argument("--item-id", default=None, help="String item_id value to keep, for example MSR0022.")
    parser.add_argument("--x-col", default="chip_x_pos")
    parser.add_argument("--y-col", default="chip_y_pos")
    parser.add_argument("--value-col", default="y_value")
    parser.add_argument("--last-update-col", default="last_update_time", help="Use '' to disable latest filtering.")
    parser.add_argument("--pitch-x", type=float, default=None)
    parser.add_argument("--pitch-y", type=float, default=None)
    parser.add_argument("--no-auto-aspect", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    update_col = args.last_update_col.strip() or None
    harmonics = _split_columns(args.harmonics)
    config = FourierConfig(
        input_path=args.input_csv,
        output_csv=args.output_csv,
        inner_radius=args.inner_radius,
        outer_radius=args.outer_radius,
        target_harmonic=args.target_harmonic,
        harmonics=tuple(int(harmonic) for harmonic in harmonics) if harmonics else None,
        angular_bins=args.angular_bins,
        min_ring_chips=args.min_ring_chips,
        group_cols=_split_columns(args.group_cols),
        geometry_cols=_split_columns(args.geometry_cols),
        item_id_col=args.item_id_col,
        item_id_value=args.item_id,
        x_col=args.x_col,
        y_col=args.y_col,
        value_col=args.value_col,
        update_col=update_col,
        pitch_x=args.pitch_x,
        pitch_y=args.pitch_y,
        auto_aspect=not args.no_auto_aspect,
    )
    run_fourier_y(config, write_csv=True)


if __name__ == "__main__":
    main()
