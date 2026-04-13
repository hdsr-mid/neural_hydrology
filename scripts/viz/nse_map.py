# Get dataframe of afvoergebied code, geometry, nse_1d, nse_1h, nse_avg
from pathlib import Path
from utils.results import evaluate, get_nse_values_gdf

import plotly.express as px
import geopandas as gpd


def plot_nse(
        gdf: gpd.GeoDataFrame,
        time_resolution: str,
        title: str
):
    gdf = gdf.to_crs(epsg=4326)
    nse_field = f"nse_{time_resolution}"

    fig = px.choropleth_map(
        gdf,
        geojson=gdf,
        locations="SHAPE_ID",
        featureidkey="properties.SHAPE_ID",
        color=nse_field,
        color_continuous_scale="RdYlGn",
        map_style="carto-positron",   # background map
        range_color=(-1, 1),
        zoom=7,
        center={"lat": 52.1, "lon": 5.3},  # Netherlands center
        opacity=0.7,
        labels={nse_field: f"NSE ({time_resolution})"}
    )

    minx, miny, maxx, maxy = gdf.total_bounds
    pad = 0.05
    bounds = dict(
        west=minx - pad,
        south=miny - pad,
        east=maxx + pad,
        north=maxy + pad,
    )

    fig.update_layout(
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        map=dict(bounds=bounds)
    )

    fig.update_layout(
        title=title,
    )

    fig.show()


if __name__ == "__main__":

    run_dir = "C:/Users/leendert.vanwolfswin/Documents/hdsr/runs/runs/development_run_23_2503_122253"
    netcdf_output_dir = Path(run_dir) / "netcdf"
    netcdf_output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(__file__).parent.parent.parent / "data"
    basins_file = data_dir / "hdsr_polders.txt"
    with basins_file.open("r") as f:
        lines = f.readlines()
        basins = [line.strip() for line in lines]
    # basins = ["AFVG41"]
    print(basins)

    # Results berekenen per polder
    results_dict = evaluate(
        run_dir=run_dir,
        period="test",
        basins=basins,
        config_overrides={
            "device": "cpu",
            "data_dir": str(data_dir),
        }
    )

    nse_values_gdf = get_nse_values_gdf(
        results_dict=results_dict,
        basins=basins,
    )
    plot_nse(nse_values_gdf, time_resolution="avg", title=f"Nash-Sutcliffe Efficiency (NSE) per afvoergebied, gemiddelde van 1h en 1D")


print("Klaar")
