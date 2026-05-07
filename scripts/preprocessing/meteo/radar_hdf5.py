from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import geopandas as gpd
import h5py
import numpy as np
import pandas as pd
from rasterio.transform import from_origin
from rasterstats import zonal_stats

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class RadarGrid:
    precip_mm: np.ndarray  # (y, x)
    affine: object  # rasterio Affine
    crs_proj4: str
    timestamp_utc: datetime
    nodata_masked: bool = True


def _normalize_radar_proj4_and_scale(proj4_params: str) -> tuple[str, float]:
    """
    Normalize KNMI radar PROJ.4 strings so PROJ can build CRS transformers reliably.

    Some KNMI radar datasets publish PROJ.4 with ellipsoid axes in km (e.g. +a=6378.137),
    which makes PROJ think it is a different celestial body than common meter-based CRS.
    We convert axes to meters and switch units to meters, and return a scale factor for
    x/y grid coordinates (km -> m).
    """
    s = proj4_params.strip()

    # Detect km-based ellipsoid (values around 6378) and convert to meters.
    a_m = re.search(r"\+a=(\d+(\.\d+)?)", s)
    b_m = re.search(r"\+b=(\d+(\.\d+)?)", s)
    scale = 1.0
    if a_m and b_m:
        a = float(a_m.group(1))
        b = float(b_m.group(1))
        if a < 100000 and b < 100000:
            s = re.sub(r"\+a=\d+(\.\d+)?", f"+a={a*1000.0:.3f}", s)
            s = re.sub(r"\+b=\d+(\.\d+)?", f"+b={b*1000.0:.3f}", s)
            scale = 1000.0

    # Normalize units if needed
    if "+units=km" in s and scale == 1000.0:
        s = s.replace("+units=km", "+units=m")

    return s, scale


def _timestamp_from_overview_attrs(f: h5py.File) -> datetime | None:
    try:
        raw = f["/overview"].attrs.get("product_datetime_end")
        if raw is None:
            return None
        s = raw.tobytes().split(b"\x00")[0].decode("ascii")
        s = s.strip()
        # Try common KNMI formats, with special handling for 24:00 endtimes.
        for fmt in ("%Y%m%d%H%M%S", "%Y%m%d%H%M", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"):
            try:
                dt = datetime.strptime(s.rstrip("Z"), fmt).replace(tzinfo=timezone.utc)
                return dt
            except ValueError:
                pass
        # Handle ISO-like "...T24:00:00Z"
        m = re.search(r"^(\d{4})-(\d{2})-(\d{2})T24:00:00Z?$", s)
        if m:
            dt0 = datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)), 0, 0, 0, tzinfo=timezone.utc)
            return dt0 + timedelta(days=1)
    except Exception:
        return None
    return None


def _timestamp_from_filename(path: Path) -> datetime | None:
    name = path.name
    # Prefer explicit ...T... patterns in MFBS style
    m = re.search(r"(\d{8}T\d{6})", name)
    if m:
        try:
            return datetime.strptime(m.group(1), "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc)
        except ValueError:
            return None
    # Then try 14 digits (YYYYMMDDHHMMSS)
    m = re.search(r"(\d{14})", name)
    if m:
        try:
            return datetime.strptime(m.group(1), "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
        except ValueError:
            return None
    # Then try 12 digits (YYYYMMDDHHMM)
    m = re.search(r"(\d{12})", name)
    if m:
        raw = m.group(1)
        ymd = raw[:8]
        hh = int(raw[8:10])
        mm = int(raw[10:12])
        if hh == 24 and mm == 0:
            dt0 = datetime.strptime(ymd, "%Y%m%d").replace(tzinfo=timezone.utc)
            return dt0 + timedelta(days=1)
        try:
            return datetime.strptime(raw, "%Y%m%d%H%M").replace(tzinfo=timezone.utc)
        except ValueError:
            return None
    return None


def read_knmi_radar_h5(path: Path) -> RadarGrid:
    """
    Read KNMI radar HDF5 to precipitation grid in mm (accumulation for product interval).

    Uses calibration formula stored in /image1/calibration and geographic grid metadata
    as shown in scripts/preprocessing/proces_h5.py.
    """
    with h5py.File(path, mode="r") as f:
        timestamp = _timestamp_from_filename(path) or _timestamp_from_overview_attrs(f)
        if timestamp is None:
            raise ValueError(f"Cannot parse timestamp from filename or /overview: {path.name}")

        cal = f["/image1/calibration"].attrs
        calibration_formula = cal["calibration_formulas"].decode("ascii").replace(" ", "")
        multiplier = float(
            calibration_formula[
                calibration_formula.index("GEO=") + len("GEO=") : calibration_formula.index("*PV+")
            ]
        )
        offset = float(calibration_formula[calibration_formula.index("*PV+") + len("*PV+") :])
        out_of_image = int(cal["calibration_out_of_image"][0])
        missing_data = int(cal["calibration_missing_data"][0])

        geo = f["geographic"].attrs
        ncols = int(geo["geo_number_columns"][0])
        nrows = int(geo["geo_number_rows"][0])
        dx = float(geo["geo_pixel_size_x"][0])
        dy = float(geo["geo_pixel_size_y"][0])
        col_off = float(geo["geo_column_offset"][0])
        row_off = float(geo["geo_row_offset"][0])
        proj4_params = (
            f["geographic"]["map_projection"]
            .attrs["projection_proj4_params"]
            .tobytes()
            .split(b"\x00")[0]
            .decode("ascii")
        )
        proj4_params_norm, coord_scale = _normalize_radar_proj4_and_scale(proj4_params)

        # Build affine from KNMI geographic metadata.
        # Note: some products use negative pixel_size_y (north-up grids). rasterio's
        # from_origin expects positive pixel sizes and will encode the north-up
        # convention by storing a negative y scale in the affine.
        x_ul = (-col_off) * coord_scale
        y_ul = ((-row_off) if dy < 0 else (-row_off + nrows * dy)) * coord_scale
        affine = from_origin(x_ul, y_ul, abs(dx) * coord_scale, abs(dy) * coord_scale)

        pixel = f["/image1/image_data"][:].astype(np.float64)
        precip = offset + multiplier * pixel
        precip[pixel == out_of_image] = np.nan
        precip[pixel == missing_data] = np.nan

        return RadarGrid(
            precip_mm=precip,
            affine=affine,
            crs_proj4=proj4_params_norm,
            timestamp_utc=timestamp,
        )


def read_polders() -> gpd.GeoDataFrame:
    nh_root = Path(__file__).resolve().parents[3]
    attr_path = nh_root / "data_ens" / "attributes" / "polders_data_aangevuld.csv"
    df = pd.read_csv(attr_path)
    geom = gpd.GeoSeries.from_wkt(df["geom_simple"])
    gdf = gpd.GeoDataFrame(df[["SHAPE_ID"]].copy(), geometry=geom, crs="EPSG:28992")
    return gdf


def zonal_mean_precip_mm(grid: RadarGrid, polders: gpd.GeoDataFrame) -> pd.Series:
    """
    Compute mean precipitation over each polygon.

    Returns a Series indexed by SHAPE_ID with float values (mm).
    """
    radar_crs = grid.crs_proj4
    p = polders.to_crs(radar_crs)
    # rasterstats expects array oriented with row 0 as top row; from_origin matches that.
    stats = zonal_stats(
        p,
        grid.precip_mm,
        affine=grid.affine,
        stats=["mean"],
        all_touched=True,
        nodata=np.nan,
    )
    vals = np.array([d.get("mean", np.nan) for d in stats], dtype=float)
    return pd.Series(vals, index=p["SHAPE_ID"].astype(str).to_list(), name="neerslag_mm")

