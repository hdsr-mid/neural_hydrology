"""Visualize hourly hydrographs from evaluated NetCDF results using Dash."""

import sys
import threading
import time
import webbrowser
from pathlib import Path

import dash
import numpy as np
import pandas as pd
import plotly.express as px
import xarray as xr
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go

# Add the neural_hydrology src to the path so we can import from it
sys.path.insert(0, str(Path(__file__).parent.parent / "neural_hydrology" / "src"))

from neural_hydrology.utils.attributes import get_area
from neural_hydrology.utils.results import compute_nse

# --- Configuration ---
# adapt these code to make it point to the right folders
# List of (label, eval_dir) tuples — one per model
EVAL_DIRS = [
    ("Model 1", Path(__file__).parent / "eval_results" / "development_run_23"),
    
    ("Model 2", Path(__file__).parent / "eval_results" / "development_run_23"),

    # ("Model 2", Path(__file__).parent / "eval_results" / "run_seed_2"),
    # ("Model 3", Path(__file__).parent / "eval_results" / "run_seed_3"),
]
PERIODS = ["train", "validation", "test"]
BASIN_FILE = Path(__file__).parent / "hdsr_polders.txt"




def get_basins() -> list[str]:
    return [line.strip() for line in BASIN_FILE.read_text().splitlines() if line.strip()]


def load_hydrograph_data(eval_dirs: list[tuple[str, Path]], basins: list[str], periods: list[str]) -> dict:
    """
    Load hourly NetCDF results for multiple models and periods, convert from m/s to m³/s.

    Returns a dict keyed by basin:
        {basin: {(label, period): {"datetime": ndarray, "observed": ndarray, "simulated": ndarray, "nse": float}}}
    """
    data = {}
    for basin in basins:
        area = get_area(basin)
        basin_entry = {}

        for label, eval_dir in eval_dirs:
            for period in periods:
                nc_path = eval_dir / period / f"{basin}_1h.nc"
                if not nc_path.exists():
                    continue

                ds = xr.open_dataset(nc_path)
                obs = ds["afvoer_obs"].values.flatten()
                sim = ds["afvoer_sim"].values.flatten()
                datetime_index = pd.to_datetime(ds["datetime"].values)
                ds.close()

                # Convert m/s -> m³/s
                if area is not None:
                    obs = obs * area
                    sim = sim * area

                nse = compute_nse(obs, sim)
                basin_entry[(label, period)] = {
                    "datetime": datetime_index,
                    "observed": obs,
                    "simulated": sim,
                    "nse": nse,
                }

        if basin_entry:
            data[basin] = basin_entry

    return data


def plot(data: dict, model_labels: list[str], periods: list[str]):
    """Launch a Dash app to interactively view hydrographs per basin."""
    basins = list(data.keys())
    color_seq = px.colors.qualitative.Plotly

    app = dash.Dash(__name__)

    app.layout = html.Div([
        html.H2("Hydrograph Visualisatie (uurlijks)"),
        html.Div([
            html.Div(
                dcc.Dropdown(
                    id="basin-dropdown",
                    options=[{"label": b, "value": b} for b in basins],
                    value=basins[0],
                    clearable=False,
                    style={"width": "300px"},
                ),
            ),
            html.Div(
                dcc.Checklist(
                    id="model-checklist",
                    options=[{"label": lbl, "value": lbl} for lbl in model_labels],
                    value=model_labels,
                    labelStyle={"display": "inline-block", "margin-right": "15px"},
                ),
            ),
            html.Div(
                dcc.Checklist(
                    id="period-checklist",
                    options=[{"label": p.capitalize(), "value": p} for p in periods],
                    value=["test"],
                    labelStyle={"display": "inline-block", "margin-right": "15px"},
                ),
            ),
        ], style={"display": "flex", "gap": "30px", "align-items": "center", "margin-bottom": "20px"}),
        dcc.Graph(id="hydrograph"),
    ])

    @app.callback(
        Output("hydrograph", "figure"),
        Input("basin-dropdown", "value"),
        Input("model-checklist", "value"),
        Input("period-checklist", "value"),
    )
    def update_figure(selected_basin, selected_models, selected_periods):
        basin_data = data[selected_basin]

        fig = go.Figure()

        # Plot observed and simulated traces per model per period
        for i, label in enumerate(model_labels):
            if label not in selected_models:
                continue
            color = color_seq[i % len(color_seq)]

            for period in periods:
                if period not in selected_periods:
                    continue
                key = (label, period)
                if key not in basin_data:
                    continue

                entry = basin_data[key]
                nse = entry["nse"]

                # Observed trace (grey, one per period)
                fig.add_trace(go.Scatter(
                    x=entry["datetime"],
                    y=entry["observed"],
                    mode="lines",
                    name=f"Geobserveerd ({period})",
                    line=dict(color="grey"),
                    legendgroup=f"obs_{period}",
                    showlegend=(i == 0),
                ))

                # Simulated trace
                fig.add_trace(go.Scatter(
                    x=entry["datetime"],
                    y=entry["simulated"],
                    mode="lines",
                    name=f"{label} — {period} (NSE: {nse:.3f})",
                    line=dict(color=color),
                ))

        # Vertical lines at period boundaries
        if "train" in selected_periods:
            for label in model_labels:
                key = (label, "train")
                if key in basin_data:
                    train_end = pd.Timestamp(basin_data[key]["datetime"].max())
                    fig.add_vline(x=train_end, line_dash="dash", line_color="black")
                    break

        if "validation" in selected_periods:
            for label in model_labels:
                key = (label, "validation")
                if key in basin_data:
                    val_end = pd.Timestamp(basin_data[key]["datetime"].max())
                    fig.add_vline(x=val_end, line_dash="dash", line_color="black")
                    break

        fig.update_layout(
            title=f"Hydrograph {selected_basin}",
            xaxis_title="Datum",
            yaxis_title="Debiet (m³/s)",
            legend=dict(orientation="h", y=-0.15, x=0.5, xanchor="center"),
            margin=dict(b=80),
        )

        return fig

    def _open_browser():
        time.sleep(1.0)
        webbrowser.open("http://127.0.0.1:8050/")

    threading.Thread(target=_open_browser, daemon=True).start()
    print("Dash is running at http://127.0.0.1:8050/")
    app.run(host="127.0.0.1", port=8050, debug=False, use_reloader=False)


if __name__ == "__main__":
    basins = get_basins()
    model_labels = [label for label, _ in EVAL_DIRS]
    data = load_hydrograph_data(eval_dirs=EVAL_DIRS, basins=basins, periods=PERIODS)
    print(f"Loaded {len(data)} basins across {len(EVAL_DIRS)} model(s)")
    plot(data, model_labels=model_labels, periods=PERIODS)
