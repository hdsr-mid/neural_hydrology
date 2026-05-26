"""Export grouped ensemble NetCDF inference output to a long-format Unity Catalog table."""

from typing import Literal


from __future__ import annotations

import logging
from pathlib import Path

import netCDF4
import pandas as pd
import xarray as xr

LOGGER = logging.getLogger(__name__)

ENSEMBLE_NETCDF_NAME = "polders_hdsr_1h.nc"
TARGET_VARIABLE = "afvoer"
SIM_PREFIX = f"{TARGET_VARIABLE}_sim_"

OUTPUT_COLUMNS = ("datetime", "ensemble_id", "afvoergeb_id", "value")


def _decode_datetime(nc_path: Path, basin_id: str) -> pd.DatetimeIndex:
    with netCDF4.Dataset(nc_path, mode="r") as root:
        if basin_id not in root.groups:
            raise KeyError(f"Basin group {basin_id!r} not found in {nc_path}")
        tvar = root.groups[basin_id].variables["datetime"]
        values = tvar[:]
        units = getattr(tvar, "units", "") or ""
        if "since 1970" in units:
            return pd.to_datetime(values, unit="s", origin="unix")
        calendar = getattr(tvar, "calendar", "standard")
        decoded = netCDF4.num2date(
            values,
            units=units,
            calendar=calendar,
            only_use_cftime_datetimes=False,
        )
        return pd.DatetimeIndex(pd.to_datetime(decoded))


def load_ensemble_netcdf_long(nc_path: Path) -> pd.DataFrame:
    """Read grouped ensemble NetCDF and return a long-format DataFrame."""
    nc_path = Path(nc_path).resolve()
    if not nc_path.is_file():
        raise FileNotFoundError(nc_path)

    with netCDF4.Dataset(nc_path, mode="r") as root:
        basin_ids = sorted(root.groups.keys())

    if not basin_ids:
        raise ValueError(f"No basin groups found in {nc_path}")

    parts: list[pd.DataFrame] = []
    for basin_id in basin_ids:
        datetime_index = _decode_datetime(nc_path, basin_id)
        ds = xr.open_dataset(nc_path, group=basin_id)
        try:
            sim_vars = sorted(
                (v for v in ds.data_vars if str(v).startswith(SIM_PREFIX)),
                key=lambda name: int(str(name).split("_")[-1]),
            )
            if not sim_vars:
                raise KeyError(
                    f"No variables with prefix {SIM_PREFIX!r} in group {basin_id}."
                )
            for var_name in sim_vars:
                ensemble_id = int(str(var_name).split("_")[-1])
                values = ds[var_name].values.astype(float, copy=False)
                parts.append(
                    pd.DataFrame(
                        {
                            "datetime": datetime_index,
                            "ensemble_id": ensemble_id,
                            "afvoergeb_id": basin_id,
                            "value": values,
                        }
                    )
                )
        finally:
            ds.close()

    df = pd.concat(parts, ignore_index=True)
    df = df[list[Literal['datetime', 'ensemble_id', 'afvoergeb_id', 'value']](OUTPUT_COLUMNS)].sort_values(
        ["afvoergeb_id", "ensemble_id", "datetime"],
        ignore_index=True,
    )
    LOGGER.info(
        "Loaded %s rows from %s (%s basins)",
        len(df),
        nc_path.name,
        df["afvoergeb_id"].nunique(),
    )
    return df


def write_ensemble_table(spark, df: pd.DataFrame, table_name: str) -> None:
    """Truncate the Unity Catalog table (if it exists) and append fresh NetCDF data."""
    try:
        from pyspark.sql import types as T
    except ImportError as exc:
        raise ImportError(
            "PySpark is required to write ensemble forecasts to Unity Catalog. "
            "Run this module on a Databricks cluster."
        ) from exc

    table_name = table_name.strip()
    if not table_name:
        raise ValueError("table_name must be a non-empty Unity Catalog name.")

    missing = set(OUTPUT_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame missing columns: {sorted(missing)}")

    schema = T.StructType(
        [
            T.StructField("datetime", T.TimestampType(), nullable=False),
            T.StructField("ensemble_id", T.IntegerType(), nullable=False),
            T.StructField("afvoergeb_id", T.StringType(), nullable=False),
            T.StructField("value", T.DoubleType(), nullable=True),
        ]
    )

    out = df[list[Literal['datetime', 'ensemble_id', 'afvoergeb_id', 'value']](OUTPUT_COLUMNS)].copy()
    out["datetime"] = pd.to_datetime(out["datetime"])

    sdf = spark.createDataFrame(out, schema=schema)

    if spark.catalog.tableExists(table_name):
        LOGGER.info("Truncating table %s", table_name)
        spark.sql(f"TRUNCATE TABLE {table_name}")
    else:
        LOGGER.info("Table %s does not exist yet; will be created on write", table_name)

    sdf.write.format("delta").mode("append").saveAsTable(table_name)

    dt_min = out["datetime"].min()
    dt_max = out["datetime"].max()
    LOGGER.info(
        "Wrote %s rows to %s (datetime %s .. %s)",
        len(out),
        table_name,
        dt_min,
        dt_max,
    )
