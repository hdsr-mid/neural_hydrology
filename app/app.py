from functools import lru_cache
from pathlib import Path
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, callback, dcc, html, no_update


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = DATA_DIR / "models"

ROB_EXPORT_PATH = DATA_DIR / "rob_export_2022.csv.gz"
MANIFEST_PATH = DATA_DIR / "manifest.csv"
NSE_PATH = DATA_DIR / "nse_1h.csv"


def empty_figure(message: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
        font={"size": 16},
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(
        template="plotly_white",
        margin={"l": 30, "r": 30, "t": 50, "b": 30},
    )
    return fig


@lru_cache(maxsize=1)
def load_rob_export() -> pd.DataFrame:
    if not ROB_EXPORT_PATH.exists():
        return pd.DataFrame(
            columns=["datetime", "discharge_m3_s", "afvoergebied_id"]
        )

    df = pd.read_csv(
        ROB_EXPORT_PATH,
        compression="gzip",
        parse_dates=["datetime"],
    )
    df["discharge_m3_s"] = pd.to_numeric(
        df["discharge_m3_s"],
        errors="coerce",
    )

    return (
        df.dropna(subset=["datetime", "afvoergebied_id"])
        .sort_values(["afvoergebied_id", "datetime"])
        .reset_index(drop=True)
    )


@lru_cache(maxsize=1)
def load_manifest() -> pd.DataFrame:
    if not MANIFEST_PATH.exists():
        return pd.DataFrame(
            columns=["afvoergebied_id", "bestand", "rijen", "bestand_mb"]
        )
    return pd.read_csv(MANIFEST_PATH)


@lru_cache(maxsize=1)
def load_nse() -> pd.DataFrame:
    if not NSE_PATH.exists():
        return pd.DataFrame(
            columns=["afvoergebied_id", "dataset", "model", "nse_1h"]
        )

    df = pd.read_csv(NSE_PATH)
    df["nse_1h"] = pd.to_numeric(df["nse_1h"], errors="coerce")
    return df


@lru_cache(maxsize=8)
def load_basin_model_data(basin: str) -> pd.DataFrame:
    path = MODEL_DIR / f"{basin}.csv.gz"

    if not path.exists():
        return pd.DataFrame(
            columns=[
                "datetime",
                "observed_m3_s",
                "model_1_m3_s",
                "model_2_m3_s",
                "dataset",
            ]
        )

    df = pd.read_csv(
        path,
        compression="gzip",
        parse_dates=["datetime"],
    )

    for col in ["observed_m3_s", "model_1_m3_s", "model_2_m3_s"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df.sort_values(["dataset", "datetime"]).reset_index(drop=True)


rob_df = load_rob_export()
manifest_df = load_manifest()
nse_df = load_nse()

rob_basins = sorted(rob_df["afvoergebied_id"].dropna().unique().tolist())
model_basins = sorted(
    manifest_df["afvoergebied_id"].dropna().unique().tolist()
)

default_rob_basin = (
    "AFVG50"
    if "AFVG50" in rob_basins
    else (rob_basins[0] if rob_basins else None)
)
default_model_basin = (
    "AFVG50"
    if "AFVG50" in model_basins
    else (model_basins[0] if model_basins else None)
)

rob_min_date = (
    rob_df["datetime"].min().date()
    if not rob_df.empty
    else None
)
rob_max_date = (
    rob_df["datetime"].max().date()
    if not rob_df.empty
    else None
)


def card(title, value_id):
    return html.Div(
        [
            html.Div(title, className="metric-label"),
            html.Div(id=value_id, className="metric-value"),
        ],
        className="metric-card",
    )


app = Dash(__name__)
server = app.server

model_warning = None
if manifest_df.empty:
    model_warning = html.Div(
        [
            html.Strong("Modelvergelijking nog niet beschikbaar. "),
            html.Span(
                "Run in de visualisatie-notebook eerst sectie 10."
            ),
        ],
        className="warning-box",
    )

rob_warning = None
if rob_df.empty:
    rob_warning = html.Div(
        [
            html.Strong("Rob-export nog niet beschikbaar. "),
            html.Span(
                "Run in de visualisatie-notebook eerst sectie 10."
            ),
        ],
        className="warning-box",
    )


model_tab = html.Div(
    [
        model_warning,
        html.Div(
            [
                html.Div(
                    [
                        html.Label("Afvoergebied"),
                        dcc.Dropdown(
                            id="model-basin",
                            options=[
                                {"label": x, "value": x}
                                for x in model_basins
                            ],
                            value=default_model_basin,
                            clearable=False,
                        ),
                    ],
                    className="control",
                ),
                html.Div(
                    [
                        html.Label("Dataset"),
                        dcc.Dropdown(
                            id="model-dataset",
                            options=[
                                {"label": "Validatie", "value": "validation"},
                                {"label": "Test", "value": "test"},
                            ],
                            value="test",
                            clearable=False,
                        ),
                    ],
                    className="control",
                ),
                html.Div(
                    [
                        html.Label("Dag"),
                        dcc.DatePickerSingle(
                            id="model-date",
                            display_format="DD-MM-YYYY",
                            first_day_of_week=1,
                        ),
                    ],
                    className="control",
                ),
            ],
            className="controls-grid",
        ),
        html.Div(
            [
                card("NSE 1h — model 1", "nse-model-1"),
                card("NSE 1h — model 2", "nse-model-2"),
                card("Uren op geselecteerde dag", "model-hours"),
            ],
            className="metrics-grid",
        ),
        dcc.Graph(
            id="model-graph",
            figure=empty_figure(
                "Kies een afvoergebied, dataset en dag."
            ),
        ),
    ],
    className="tab-body",
)


rob_tab = html.Div(
    [
        rob_warning,
        html.Div(
            className="info-box",
            children=[
                html.Strong("Voorlopige 2022-export. "),
                html.Span(
                    "AFVG1, AFVG42 en AFVG65 zijn tijdelijk uitgesloten "
                    "door onvolledige historische brondata."
                ),
            ],
        ),
        html.Div(
            [
                html.Div(
                    [
                        html.Label("Afvoergebied"),
                        dcc.Dropdown(
                            id="rob-basin",
                            options=[
                                {"label": x, "value": x}
                                for x in rob_basins
                            ],
                            value=default_rob_basin,
                            clearable=False,
                        ),
                    ],
                    className="control",
                ),
                html.Div(
                    [
                        html.Label("Aggregatie"),
                        dcc.Dropdown(
                            id="rob-frequency",
                            options=[
                                {"label": "Uur", "value": "h"},
                                {"label": "Dag", "value": "D"},
                                {"label": "Week", "value": "W"},
                            ],
                            value="D",
                            clearable=False,
                        ),
                    ],
                    className="control",
                ),
                html.Div(
                    [
                        html.Label("Periode"),
                        dcc.DatePickerRange(
                            id="rob-dates",
                            min_date_allowed=rob_min_date,
                            max_date_allowed=rob_max_date,
                            start_date=rob_min_date,
                            end_date=rob_max_date,
                            display_format="DD-MM-YYYY",
                            first_day_of_week=1,
                        ),
                    ],
                    className="control",
                ),
            ],
            className="controls-grid",
        ),
        html.Div(
            [
                card("Gemiddeld debiet", "rob-mean"),
                card("Maximum debiet", "rob-max"),
                card("Volume geselecteerde periode", "rob-volume"),
            ],
            className="metrics-grid",
        ),
        html.Div(
            [
                html.Button(
                    "Download geselecteerde data",
                    id="rob-download-button",
                    n_clicks=0,
                    className="button",
                ),
                dcc.Download(id="rob-download"),
            ],
            className="download-row",
        ),
        dcc.Graph(
            id="rob-timeseries",
            figure=empty_figure("Geen exportdata beschikbaar."),
        ),
        dcc.Graph(
            id="rob-ranking",
            figure=empty_figure("Geen exportdata beschikbaar."),
        ),
    ],
    className="tab-body",
)


app.layout = html.Div(
    [
        html.Div(
            [
                html.H1("NeuralHydrology — HDSR"),
                html.P(
                    "Interactieve controle van modelresultaten en de "
                    "2022-export."
                ),
            ],
            className="header",
        ),
        dcc.Tabs(
            value="model",
            children=[
                dcc.Tab(
                    label="Modelvergelijking",
                    value="model",
                    children=model_tab,
                ),
                dcc.Tab(
                    label="Export voor Rob",
                    value="rob",
                    children=rob_tab,
                ),
            ],
        ),
    ],
    className="page",
)


@callback(
    Output("model-date", "min_date_allowed"),
    Output("model-date", "max_date_allowed"),
    Output("model-date", "date"),
    Input("model-basin", "value"),
    Input("model-dataset", "value"),
)
def update_model_date_bounds(basin, dataset):
    if basin is None or dataset is None:
        return None, None, None

    d = load_basin_model_data(basin)
    d = d[d["dataset"] == dataset]

    if d.empty:
        return None, None, None

    min_date = d["datetime"].min().date()
    max_date = d["datetime"].max().date()

    preferred = pd.Timestamp("2022-01-01").date()
    selected = (
        preferred
        if min_date <= preferred <= max_date
        else min_date
    )

    return min_date, max_date, selected


@callback(
    Output("model-graph", "figure"),
    Output("nse-model-1", "children"),
    Output("nse-model-2", "children"),
    Output("model-hours", "children"),
    Input("model-basin", "value"),
    Input("model-dataset", "value"),
    Input("model-date", "date"),
)
def update_model_graph(basin, dataset, selected_date):
    if basin is None or dataset is None or selected_date is None:
        return (
            empty_figure("Modeldata is niet beschikbaar."),
            "—",
            "—",
            "—",
        )

    d = load_basin_model_data(basin)

    start = pd.Timestamp(selected_date)
    end = start + pd.Timedelta(days=1)

    d = d[
        (d["dataset"] == dataset)
        & (d["datetime"] >= start)
        & (d["datetime"] < end)
    ].copy()

    if d.empty:
        return (
            empty_figure(
                f"Geen data voor {basin} op {start.date()}."
            ),
            "—",
            "—",
            "0",
        )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=d["datetime"],
            y=d["observed_m3_s"],
            mode="lines",
            name="Gemeten",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=d["datetime"],
            y=d["model_1_m3_s"],
            mode="lines",
            name="Model 1",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=d["datetime"],
            y=d["model_2_m3_s"],
            mode="lines",
            name="Model 2",
        )
    )

    fig.update_layout(
        template="plotly_white",
        title=(
            f"{basin} — "
            f"{'validatie' if dataset == 'validation' else 'test'} — "
            f"{start.strftime('%d-%m-%Y')}"
        ),
        xaxis_title="Datum/tijd",
        yaxis_title="Debiet [m³/s]",
        hovermode="x unified",
        legend={"orientation": "h", "y": 1.12},
        margin={"l": 60, "r": 30, "t": 90, "b": 55},
    )

    def nse_for(model_name):
        n = nse_df[
            (nse_df["afvoergebied_id"] == basin)
            & (nse_df["dataset"] == dataset)
            & (nse_df["model"] == model_name)
        ]["nse_1h"]

        if n.empty or not np.isfinite(n.iloc[0]):
            return "NaN"

        return f"{n.iloc[0]:.3f}"

    return (
        fig,
        nse_for("Model 1"),
        nse_for("Model 2"),
        f"{len(d):,}",
    )


@callback(
    Output("rob-timeseries", "figure"),
    Output("rob-mean", "children"),
    Output("rob-max", "children"),
    Output("rob-volume", "children"),
    Input("rob-basin", "value"),
    Input("rob-frequency", "value"),
    Input("rob-dates", "start_date"),
    Input("rob-dates", "end_date"),
)
def update_rob_graph(basin, frequency, start_date, end_date):
    if (
        rob_df.empty
        or basin is None
        or start_date is None
        or end_date is None
    ):
        return (
            empty_figure("Exportdata is niet beschikbaar."),
            "—",
            "—",
            "—",
        )

    start = pd.Timestamp(start_date)
    end_exclusive = pd.Timestamp(end_date) + pd.Timedelta(days=1)

    d = rob_df[
        (rob_df["afvoergebied_id"] == basin)
        & (rob_df["datetime"] >= start)
        & (rob_df["datetime"] < end_exclusive)
    ].copy()

    if d.empty:
        return (
            empty_figure("Geen data in de gekozen periode."),
            "—",
            "—",
            "—",
        )

    d = d.set_index("datetime").sort_index()

    rule_map = {"h": "1h", "D": "1D", "W": "1W"}
    rule = rule_map.get(frequency, "1D")

    plotted = (
        d["discharge_m3_s"]
        .resample(rule)
        .mean()
        .dropna()
    )

    fig = go.Figure(
        go.Scatter(
            x=plotted.index,
            y=plotted.values,
            mode="lines",
            name="Voorspeld debiet",
        )
    )

    fig.update_layout(
        template="plotly_white",
        title=f"{basin} — voorspeld debiet",
        xaxis_title="Datum/tijd",
        yaxis_title="Debiet [m³/s]",
        hovermode="x unified",
        margin={"l": 60, "r": 30, "t": 70, "b": 55},
    )

    mean_q = d["discharge_m3_s"].mean()
    max_q = d["discharge_m3_s"].max()
    volume = d["discharge_m3_s"].sum() * 3600

    return (
        fig,
        f"{mean_q:.3f} m³/s",
        f"{max_q:.3f} m³/s",
        f"{volume / 1_000_000:.2f} mln m³",
    )


@callback(
    Output("rob-ranking", "figure"),
    Input("rob-basin", "value"),
)
def update_ranking(_):
    if rob_df.empty:
        return empty_figure("Exportdata is niet beschikbaar.")

    ranking = (
        rob_df.groupby("afvoergebied_id")["discharge_m3_s"]
        .sum()
        .mul(3600)
        .div(1_000_000)
        .sort_values()
    )

    fig = go.Figure(
        go.Bar(
            x=ranking.values,
            y=ranking.index,
            orientation="h",
        )
    )
    fig.update_layout(
        template="plotly_white",
        title="Voorspeld jaarvolume 2022 per afvoergebied",
        xaxis_title="Jaarvolume [miljoen m³]",
        yaxis_title="Afvoergebied",
        height=max(650, 22 * len(ranking)),
        margin={"l": 75, "r": 30, "t": 70, "b": 55},
    )
    return fig


@callback(
    Output("rob-download", "data"),
    Input("rob-download-button", "n_clicks"),
    State("rob-basin", "value"),
    State("rob-dates", "start_date"),
    State("rob-dates", "end_date"),
    prevent_initial_call=True,
)
def download_rob_selection(n_clicks, basin, start_date, end_date):
    if (
        not n_clicks
        or rob_df.empty
        or basin is None
        or start_date is None
        or end_date is None
    ):
        return no_update

    start = pd.Timestamp(start_date)
    end_exclusive = pd.Timestamp(end_date) + pd.Timedelta(days=1)

    d = rob_df[
        (rob_df["afvoergebied_id"] == basin)
        & (rob_df["datetime"] >= start)
        & (rob_df["datetime"] < end_exclusive)
    ][
        ["datetime", "discharge_m3_s", "afvoergebied_id"]
    ].copy()

    filename = (
        f"{basin}_{start.date()}_{pd.Timestamp(end_date).date()}.csv"
    )

    return dcc.send_data_frame(
        d.to_csv,
        filename,
        index=False,
    )


if __name__ == "__main__":
    port = int(
        os.getenv(
            "DATABRICKS_APP_PORT",
            os.getenv("PORT", "8000"),
        )
    )

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False,
    )
