# Calculate the variability of streefpeil within each polder
# Source data: Peilgebied praktijk, Geopackage download from https://data-hdsr.opendata.arcgis.com/datasets/HDSR::peilgebied-praktijk/about
# Downloaded on 2026-03-19 10:00
import math
from pathlib import Path
from typing import List

import geopandas as gpd  # not added to requirements.txt to avoid a large dependency tree just for this one-off analysis
import numpy as np
import pandas as pd

LAYER_NAME = "Peilgebied"


def read_peilgebied(path: Path) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(path, layer=LAYER_NAME)
    return gdf


def read_polders(source_crs="EPSG:4326", target_crs: str = "EPSG:28992") -> gpd.GeoDataFrame:

    path = Path(__file__).parent.parent.parent / "data" / "attributes" / "polders_data_aangevuld.csv"
    df = pd.read_csv(path)
    geom_series = gpd.GeoSeries.from_wkt(df.geom_simple)
    gdf = gpd.GeoDataFrame(data=df, geometry=geom_series, crs=source_crs)
    gdf = gdf.to_crs(target_crs)
    return gdf


def write_polders(polders_gdf: gpd.GeoDataFrame, crs: str = "EPSG:28992") -> gpd.GeoDataFrame:

    path = Path(__file__).parent.parent.parent / "data" / "attributes" / "polders_data_aangevuld.csv"
    polders_gdf = polders_gdf.to_crs("EPSG:28992")
    polders_gdf.assign(geom_simple=polders_gdf.geometry.apply(lambda g: g.wkt)) \
        .drop(columns="geometry") \
        .to_csv(path, index=False)


def laag_en_hoog_peil(
        vast_peil: float | None,
        zomerpeil: float | None,
        winterpeil: float | None,
        flexibel_bovenpeil: float | None,
        flexibel_onderpeil: float | None,
        nodatavalues: List[float] = [-999]
):
    vast_peil = math.nan if vast_peil in nodatavalues else vast_peil
    zomerpeil = math.nan if zomerpeil in nodatavalues else zomerpeil
    winterpeil = math.nan if winterpeil in nodatavalues else winterpeil
    flexibel_bovenpeil = math.nan if flexibel_bovenpeil in nodatavalues else flexibel_bovenpeil
    flexibel_onderpeil = math.nan if flexibel_onderpeil in nodatavalues else flexibel_onderpeil
    if not math.isnan(vast_peil):
        return vast_peil, vast_peil
    elif not math.isnan(zomerpeil) or not math.isnan(winterpeil):
        return np.nanmin([zomerpeil, winterpeil]), np.nanmax([zomerpeil, winterpeil])
    elif not math.isnan(flexibel_bovenpeil) or not math.isnan(flexibel_onderpeil):
        return np.nanmin([flexibel_onderpeil, flexibel_bovenpeil]), np.nanmax([flexibel_onderpeil, flexibel_bovenpeil])
    else:
        return None, None


def calculate_low_avg_high_peil(peilgebied: gpd.GeoDataFrame) -> None:
    """
    Elk peilvak heeft een eigen regime. Soms is dat zomerpeil/winterpeil, soms vast peil, soms flexibel peil.
    Deze functie voegt kolommen toe die onafhankelijk van deze regimes voor elk peilvak een hoog en laag peil bevatten
    en het gemiddelde daarvan.

    Edits `peilgebied` in-place
    """
    peilgebied[["laag_peil", "hoog_peil"]] = peilgebied.apply(
        lambda row: pd.Series(
            laag_en_hoog_peil(
                row["VASTPEIL"],
                row["ZOMERPEIL"],
                row["WINTERPEIL"],
                row["FLEXIBEL_BOVENPEIL"],
                row["FLEXIBEL_ONDERPEIL"]
            )
        ),
        axis=1
    )
    peilgebied["gemiddeld_peil"] = peilgebied["laag_peil"] + peilgebied["hoog_peil"] / 2


def assign_polder_to_peilgebied(polders_gdf: gpd.GeoDataFrame, peilgebied_gdf: gpd.GeoDataFrame) -> None:
    """
    Add a polder_id to the peilgebied_gdf based on a spatial join (peilgebied `point_on_surface` is in polder polygon)

    Edits peilgebied_gdf in-place
    """

    # 1. Make a copy of peilgebied and compute point_on_surface
    points = peilgebied_gdf.copy()
    points["geometry"] = points.geometry.apply(lambda geom: geom.representative_point())

    # 2. Spatial join: which polder contains each point?
    joined = gpd.sjoin(
        points,
        polders_gdf,  # [["SHAPE_ID", "geom_simple"]],
        how="left",
        predicate="intersects"
    )

    # 3. Add the SHAPE_ID to the original peilgebied_gdf
    peilgebied_gdf["polder_id"] = joined["SHAPE_ID"].values


def peil_stats_to_polder(polders_gdf: gpd.GeoDataFrame, peilgebied_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Voeg statistieken over de verdeling van peilen binnen de polder toe aan de polders_gdf
    """
    summary = (
        peilgebied_gdf
        .groupby("polder_id")["gemiddeld_peil"]
        .agg(["min", "max", "mean", "std"])
    )

    summary = summary.rename(columns={
        "min": "peil_min",
        "max": "peil_max",
        "mean": "peil_mean",
        "std": "peil_std",
    })
    polders_gdf = polders_gdf.merge(
        summary,
        left_on="SHAPE_ID",
        right_index=True,
        how="left"
    )
    polders_gdf["peil_range"] = polders_gdf["peil_max"] - polders_gdf["peil_min"]
    return polders_gdf


if __name__ == "__main__":
    # Calculate the variability of streefpeil within each polder
    # Source data: Peilgebied praktijk, Geopackage download from https://data-hdsr.opendata.arcgis.com/datasets/HDSR::peilgebied-praktijk/about
    # Downloaded on 2026-03-19 10:00
    
    peilgebied_gpkg_path = Path("C:/Users/leendert.vanwolfswin/Documents/hdsr/Peilgebied_damo_786486471653385609.gpkg")
    peilgebied_gdf = read_peilgebied(peilgebied_gpkg_path)
    polders_gdf = read_polders()

    calculate_low_avg_high_peil(peilgebied_gdf)
    assign_polder_to_peilgebied(polders_gdf=polders_gdf, peilgebied_gdf=peilgebied_gdf)
    polders_gdf = peil_stats_to_polder(polders_gdf=polders_gdf, peilgebied_gdf=peilgebied_gdf)

    write_polders(polders_gdf)
