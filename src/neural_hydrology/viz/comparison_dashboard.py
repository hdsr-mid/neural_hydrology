import re

import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go

from src.neural_hydrology.utils.metrics import nse, pbias
from src.neural_hydrology.viz.color import hex_to_rgba, PALETTE

PLOT_ROWS = {
    "raw": 1,
    "cumulative": 2,
    "weekly": 3,
}


SERIES_COLORS = {
    "DHYDRO": PALETTE[0],
    "Measured": PALETTE[1],
    "LSTM": PALETTE[2],
}


def calculate_metrics(afvg_data: dict) -> dict:
    """Calculate all metrics required for the dashboard."""

    def metrics(
        measured: pd.DataFrame,
        simulated: pd.DataFrame,
    ) -> dict[str, float]:
        """Compute performance metrics for measured and simulated time series.

        The first column of each DataFrame is used for the comparison. Missing
        values are handled within the individual metric functions.

        Args:
            measured: DataFrame containing measured values.
            simulated: DataFrame containing simulated values.

        Returns:
            A dictionary containing the calculated performance metrics.
        """
        return {
            "NSE": nse(
                measured.iloc[:, 0],
                simulated.iloc[:, 0],
            ),
            "Bias (%)": pbias(
                measured.iloc[:, 0],
                simulated.iloc[:, 0],
            ),
        }

    measured_raw = afvg_data["raw"]["Measured"]
    measured_weekly = afvg_data["weekly"]["Measured"]

    return {
        "raw": {
            "DHYDRO": metrics(
                measured_raw,
                afvg_data["raw"]["DHYDRO"],
            ),
            "LSTM": metrics(
                measured_raw,
                afvg_data["raw"]["LSTM"]["median"],
            ),
        },
        "weekly": {
            "DHYDRO": metrics(
                measured_weekly,
                afvg_data["weekly"]["DHYDRO"],
            ),
            "LSTM": metrics(
                measured_weekly,
                afvg_data["weekly"]["LSTM"],
            ),
        },
        "cumulative": {
            "DHYDRO": pbias(
                measured_raw.iloc[:, 0],
                afvg_data["raw"]["DHYDRO"].iloc[:, 0],
            ),
            "LSTM": pbias(
                measured_raw.iloc[:, 0],
                afvg_data["raw"]["LSTM"]["median"].iloc[:, 0],
            ),
        },
    }


def add_nse_bias_table(
    fig: go.Figure,
    row: int,
    metrics: dict,
    visible: bool,
) -> None:
    """Add an NSE/Bias table."""

    fig.add_trace(
        go.Table(
            header=dict(
                values=["Model", "NSE", "Bias (%)"],
            ),
            cells=dict(
                values=[
                    ["DHYDRO", "LSTM"],
                    [
                        f"{metrics['DHYDRO']['NSE']:.2f}",
                        f"{metrics['LSTM']['NSE']:.2f}",
                    ],
                    [
                        f"{metrics['DHYDRO']['Bias (%)']:+.1f}%",
                        f"{metrics['LSTM']['Bias (%)']:+.1f}%",
                    ],
                ],
            ),
            visible=visible,
        ),
        row=row,
        col=2,
    )


def add_volume_error_table(
    fig: go.Figure,
    metrics: dict,
    visible: bool,
) -> None:
    """Add a cumulative volume error table."""

    fig.add_trace(
        go.Table(
            header=dict(
                values=["Model", "Volume Error"],
            ),
            cells=dict(
                values=[
                    ["DHYDRO", "LSTM"],
                    [
                        f"{metrics['DHYDRO']:+.1f}%",
                        f"{metrics['LSTM']:+.1f}%",
                    ],
                ],
            ),
            visible=visible,
        ),
        row=2,
        col=2,
    )

def add_metric_tables(
    fig: go.Figure,
    metrics: dict,
    visible: bool,
) -> int:
    """Add all metric tables."""

    add_nse_bias_table(
        fig,
        row=1,
        metrics=metrics["raw"],
        visible=visible,
    )

    add_volume_error_table(
        fig,
        metrics=metrics["cumulative"],
        visible=visible,
    )

    add_nse_bias_table(
        fig,
        row=3,
        metrics=metrics["weekly"],
        visible=visible,
    )

    return 3


def add_ensemble_traces(
    fig: go.Figure,
    row: int,
    label: str,
    series: dict,
    color: str,
    visible: bool,
    show_legend: bool,
) -> int:
    """Add median and uncertainty band traces."""

    lower = series["lower"]
    upper = series["upper"]
    median = series["median"]

    fill_color = hex_to_rgba(color, 0.2)

    fig.add_trace(
        go.Scatter(
            x=lower.index,
            y=lower.iloc[:, 0],
            mode="lines",
            line=dict(width=0),
            hoverinfo="skip",
            showlegend=False,
            legendgroup=label,
            visible=visible,
        ),
        row=row,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=upper.index,
            y=upper.iloc[:, 0],
            mode="lines",
            line=dict(width=0),
            fill="tonexty",
            fillcolor=fill_color,
            hoverinfo="skip",
            showlegend=False,
            legendgroup=label,
            visible=visible,
        ),
        row=row,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=median.index,
            y=median.iloc[:, 0],
            mode="lines",
            name=label,
            legendgroup=label,
            showlegend=show_legend,
            line=dict(
                color=color,
                width=2,
            ),
            visible=visible,
        ),
        row=row,
        col=1,
    )

    return 3


def add_dataframe_trace(
    fig: go.Figure,
    row: int,
    plot_type: str,
    label: str,
    series,
    color: str,
    visible: bool,
    show_legend: bool,
) -> int:
    """Add a regular dataframe trace."""

    if plot_type == "weekly":
        trace = go.Bar(
            x=series.index,
            y=series.iloc[:, 0],
            name=label,
            legendgroup=label,
            showlegend=show_legend,
            marker_color=color,
            visible=visible,
        )
    else:
        trace = go.Scatter(
            x=series.index,
            y=series.iloc[:, 0],
            mode="lines",
            name=label,
            legendgroup=label,
            showlegend=show_legend,
            line=dict(color=color),
            visible=visible,
        )

    fig.add_trace(
        trace,
        row=row,
        col=1,
    )

    return 1


def add_plot_type(
    fig: go.Figure,
    row: int,
    plot_type: str,
    series_dict: dict,
    visible: bool,
    show_legend: bool,
) -> int:
    """Add all traces for a single plot type."""

    trace_count = 0
    ordered_items = ordered_series_items(series_dict)
    for i, (label, series) in enumerate(ordered_items):
        color = SERIES_COLORS.get(
            label,
            PALETTE[len(SERIES_COLORS) % len(PALETTE)],
        )
        if isinstance(series, dict):
            trace_count += add_ensemble_traces(
                fig=fig,
                row=row,
                label=label,
                series=series,
                color=color,
                visible=visible,
                show_legend=show_legend,
            )
        else:
            trace_count += add_dataframe_trace(
                fig=fig,
                row=row,
                plot_type=plot_type,
                label=label,
                series=series,
                color=color,
                visible=visible,
                show_legend=show_legend,
            )

    return trace_count


def add_series_traces(
    fig: go.Figure,
    afvg_data: dict,
    visible: bool,
    show_legend: bool,
) -> int:
    """Add all timeseries traces."""

    trace_count = 0

    for plot_type, row in PLOT_ROWS.items():

        trace_count += add_plot_type(
            fig=fig,
            row=row,
            plot_type=plot_type,
            series_dict=afvg_data[plot_type],
            visible=visible,
            show_legend=show_legend and row == 1,
        )

    return trace_count


def add_afvg_to_figure(
    fig: go.Figure,
    afvg_data: dict,
    visible: bool,
    show_legend: bool,
) -> int:
    """Add all traces belonging to a single AFGV."""

    metrics = calculate_metrics(afvg_data)

    trace_count = 0

    trace_count += add_metric_tables(
        fig=fig,
        metrics=metrics,
        visible=visible,
    )

    trace_count += add_series_traces(
        fig=fig,
        afvg_data=afvg_data,
        visible=visible,
        show_legend=show_legend,
    )

    return trace_count


def add_dropdown(
    fig: go.Figure,
    afvg_ids: list[str],
    trace_counts: list[int],
    title: str,
) -> None:
    """Add the AFGV selector dropdown."""

    buttons = []

    start_trace = 0

    for afvg_id, n_traces in zip(afvg_ids, trace_counts):

        visible = [False] * len(fig.data)

        for i in range(start_trace, start_trace + n_traces):
            visible[i] = True

        buttons.append(
            dict(
                label=afvg_id,
                method="update",
                args=[
                    {"visible": visible},
                    {"title": f"{title} - {afvg_id}"},
                ],
            )
        )

        start_trace += n_traces

    fig.update_layout(
        updatemenus=[
            dict(
                buttons=buttons,
                direction="down",
                x=0,
                y=1.15,
            )
        ]
    )

def configure_layout(
    fig: go.Figure,
    title: str,
) -> None:
    """Apply final layout settings."""

    fig.update_layout(
        title=title,
        height=1200,
        hovermode="x unified",
        barmode="group",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
        ),
    )

    fig.update_yaxes(title_text="Raw [m³]", row=1, col=1)
    fig.update_yaxes(title_text="Cumulative [m³]", row=2, col=1)
    fig.update_yaxes(title_text="Weekly total [m³]", row=3, col=1)





def sort_afvg_data(data: dict[str, dict]) -> dict[str, dict]:
    """Sort AFGV codes by numeric identifier."""

    return dict(
        sorted(
            data.items(),
            key=lambda item: int(re.search(r"\d+", item[0]).group()),
        )
    )


def ordered_series_items(
    series_dict: dict,
) -> list[tuple[str, object]]:
    """Return series with Measured plotted first."""

    items = list(series_dict.items())

    items.sort(
        key=lambda item: item[0] != "Measured"
    )

    return items


def create_comparison_figure() -> go.Figure:
    """Create the base subplot layout."""

    return make_subplots(
        rows=3,
        cols=2,
        column_widths=[0.8, 0.2],
        specs=[
            [{}, {"type": "domain"}],
            [{}, {"type": "domain"}],
            [{}, {"type": "domain"}],
        ],
        subplot_titles=[
            "Raw discharge", "",
            "Cumulative discharge", "",
            "Weekly totals", "",
        ],
        vertical_spacing=0.08,
    )

def afvg_comparison_dashboard(
    data: dict[str, dict],
    title: str = (
        "DHYDRO en LSTM debieten bij eindkunstwerk "
        "vergeleken met metingen"
    ),
) -> None:
    """Create an interactive dashboard comparing modelled and measured discharges."""

    if not data:
        raise ValueError("No AFGV data provided")

    data = sort_afvg_data(data)
    fig = create_comparison_figure()
    trace_counts: list[int] = []
    for afvg_idx, (_, afvg_data) in enumerate(data.items()):
        n_traces = add_afvg_to_figure(
            fig=fig,
            afvg_data=afvg_data,
            visible=afvg_idx == 0,
            show_legend=afvg_idx == 0,
        )

        trace_counts.append(n_traces)

    add_dropdown(
        fig=fig,
        afvg_ids=list(data.keys()),
        trace_counts=trace_counts,
        title=title,
    )

    configure_layout(
        fig=fig,
        title=f"{title} - {next(iter(data.keys()))}",
    )

    fig.show()
