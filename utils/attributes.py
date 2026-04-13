from pathlib import Path

import pandas as pd
import geopandas as gpd


ATTRIBUTES_PATH = Path(__file__).parent.parent / "data" / "attributes" / "polders_data_aangevuld.csv"

def get_area(basin: str) -> float:
    """Get the area of given `basin` from the static attributes csv"""
    df = pd.read_csv(ATTRIBUTES_PATH)
    row = df.loc[df["SHAPE_ID"] == basin, "oppervlak"]
    if row.empty:
        return None
    return row.iloc[0]


def read_attributes() -> gpd.GeoDataFrame:
    """Get the area of given `basin` from the static attributes csv"""

    df = pd.read_csv(ATTRIBUTES_PATH)
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.GeoSeries.from_wkt(df["geom_simple"]),
        crs="EPSG:28992"
    )
    return gdf
