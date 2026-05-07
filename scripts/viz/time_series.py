import pandas as pd
import plotly.graph_objects as go
from _plotly_utils.colors import qualitative
from plotly.subplots import make_subplots


PALETTE: list[str] = qualitative.Plotly


def plot_time_series(
    series: dict[str, pd.DataFrame],
    title: str | None = None,
) -> None:
    """
    Plot multiple time series in a single interactive plot

    :param series: {label: timeseries} dict
    :param title: optional title for the plot
    """
    fig = go.Figure()

    for label, df in series.items():
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df.iloc[:, 0],
                mode="lines",
                name=label,
            )
        )

    fig.update_layout(
        title=dict(
            text=title or "",
            x=0.5,
            xanchor="center",
        ),
        xaxis_title="Time",
        yaxis_title="Value",
        legend_title="Series",
    )

    fig.show()


def plot_time_series_by_area(
    data: dict[str, dict[str, pd.DataFrame]],
    title: str | None = None,
) -> None:
    """
    Plot multiple time series for multiple areas in a single interactive plot

    :param data: {area_name: {label: time_series}} dict
    :param title: optional title for the plot
    """

    n_rows: int = len(data)

    fig = make_subplots(
        rows=n_rows,
        cols=1,
        shared_xaxes=True,
        subplot_titles=list(data.keys()),
        vertical_spacing=0.02,
    )

    for row, (area, series_dict) in enumerate(data.items(), start=1):
        for i, (label, df) in enumerate(series_dict.items()):
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df.iloc[:, 0],
                    mode="lines",
                    name=label,
                    legendgroup=label,
                    showlegend=row == 1,
                    line=dict(color=PALETTE[i % len(PALETTE)])
                ),
                row=row,
                col=1,
            )

        fig.update_yaxes(
            title_text="Value",
            showticklabels=True,
            row=row,
            col=1,
        )

    fig.update_layout(
        title=dict(
            text=title or "",
            x=0.5,
            xanchor="center",
        ),
        height=500 * n_rows,
        xaxis_title="Time",
        yaxis_title="Value",
    )

    fig.show()

