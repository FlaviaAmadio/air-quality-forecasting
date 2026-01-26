# src/transform.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import pandas as pd


@dataclass(frozen=True)
class AirQualityConfig:
    station_col: str = "ESTACIO"
    pollutant_col: str = "CODI_CONTAMINANT"
    year_col: str = "ANY"
    month_col: str = "MES"
    day_col: str = "DIA"


def _hour_cols(prefix: str, hours: Iterable[int] = range(1, 25)) -> list[str]:
    """Generate hourly columns like H01..H24 or V01..V24."""
    return [f"{prefix}{h:02d}" for h in hours]


def wide_to_long_hourly(
    df: pd.DataFrame,
    *,
    id_cols: Sequence[str],
    value_prefix: str = "H",
    flag_prefix: str = "V",
    value_name: str = "value",
    flag_name: str = "valid_flag",
) -> pd.DataFrame:
    """
    Convert a wide hourly table (H01..H24 and V01..V24) into long format with aligned flags.
    """
    value_cols = _hour_cols(value_prefix)
    flag_cols = _hour_cols(flag_prefix)

    missing_values = [c for c in value_cols if c not in df.columns]
    missing_flags = [c for c in flag_cols if c not in df.columns]
    if missing_values:
        raise KeyError(f"Missing value columns: {missing_values}")
    if missing_flags:
        raise KeyError(f"Missing flag columns: {missing_flags}")

    df_values = df.melt(
        id_vars=list(id_cols),
        value_vars=value_cols,
        var_name="hour_col",
        value_name=value_name,
    )
    df_values["hour"] = df_values["hour_col"].str[1:].astype(int) - 1
    df_values = df_values.drop(columns=["hour_col"])

    df_flags = df.melt(
        id_vars=list(id_cols),
        value_vars=flag_cols,
        var_name="hour_col",
        value_name=flag_name,
    )
    df_flags["hour"] = df_flags["hour_col"].str[1:].astype(int) - 1
    df_flags = df_flags.drop(columns=["hour_col"])

    df_long = df_values.merge(
        df_flags,
        on=list(id_cols) + ["hour"],
        how="left",
        validate="one_to_one",
    )

    return df_long


def build_datetime_index(
    df_long: pd.DataFrame,
    *,
    year_col: str,
    month_col: str,
    day_col: str,
    hour_col: str = "hour",
    datetime_col: str = "datetime",
) -> pd.DataFrame:
    """Create a datetime column from year/month/day/hour."""
    out = df_long.copy()
    out[datetime_col] = pd.to_datetime(
        dict(
            year=out[year_col],
            month=out[month_col],
            day=out[day_col],
            hour=out[hour_col],
        ),
        errors="coerce",
    )
    return out


def make_clean_no2_timeseries(
    df: pd.DataFrame,
    *,
    station_id: int,
    pollutant_code: int,
    cfg: AirQualityConfig = AirQualityConfig(),
    valid_values: Optional[Sequence[str]] = ("V",),
    value_col_name: str = "NO2",
    clean_col_name: str = "NO2_clean",
    set_hourly_freq: bool = True,
) -> pd.DataFrame:
    """
    Filter to one station + pollutant, reshape to long, construct datetime, and create NO2_clean.

    Returns a DataFrame indexed by datetime with:
      - station
      - NO2 (raw)
      - valid_flag
      - NO2_clean
    """
    # Filter early (reduces memory + complexity)
    df_filt = df.loc[
        (df[cfg.station_col] == station_id) & (df[cfg.pollutant_col] == pollutant_code)
    ].copy()

    if df_filt.empty:
        raise ValueError(
            f"No rows found for station_id={station_id} and pollutant_code={pollutant_code}"
        )

    id_cols = [cfg.year_col, cfg.month_col, cfg.day_col, cfg.station_col]

    df_long = wide_to_long_hourly(
        df_filt,
        id_cols=id_cols,
        value_prefix="H",
        flag_prefix="V",
        value_name=value_col_name,
        flag_name="valid_flag",
    )

    df_long = build_datetime_index(
        df_long,
        year_col=cfg.year_col,
        month_col=cfg.month_col,
        day_col=cfg.day_col,
    )

    # Clean using validation flags
    if valid_values is None:
        # If you later decide not to filter by flags, keep raw as clean
        df_long[clean_col_name] = df_long[value_col_name]
    else:
        df_long[clean_col_name] = df_long[value_col_name].where(
            df_long["valid_flag"].isin(valid_values)
        )

    # Final tidy output
    df_out = (
        df_long[["datetime", cfg.station_col, value_col_name, "valid_flag", clean_col_name]]
        .rename(columns={cfg.station_col: "station"})
        .sort_values("datetime")
        .set_index("datetime")
    )

    if set_hourly_freq:
        df_out = df_out.asfreq("H")

    return df_out
