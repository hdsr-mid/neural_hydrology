import pandas as pd

import plotly.graph_objects as go


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


