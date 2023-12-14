import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..config import DEFAULT_AXIS, DEFAULT_COLORSCHEME, DEFAULT_LAYOUT


def pairsplot(df: pd.DataFrame, var_names: list[str], thinning: float = 1.0, **kwargs) -> go.Figure:
    """Grid plot with variables from a dataframe.

    Args:
        df (panda.Dataframe): dataframe.
        var_names (list of str): names of variables to be ploted on the horizontal axis.
        thinning (float): fraction of the data to be used for plotting.
        **kwargs: additional arguments passed to `plotly.express.scatter_matrix`.


    Returns:
        plotly figure
    """
    if thinning < 1.0:
        df = df.sample(frac=thinning)

    fig = px.scatter_matrix(df, dimensions=var_names, **kwargs)
    fig.update_traces(diagonal_visible=False)
    fig.update_layout(DEFAULT_LAYOUT)
    fig.update_layout({f"xaxis{n+1 if n>0 else ''}": DEFAULT_AXIS for n in range(100)})
    fig.update_layout({f"yaxis{n+1 if n>0 else ''}": DEFAULT_AXIS for n in range(100)})
    return fig


def gridplot(
    df: pd.DataFrame,
    var_names_haxis: list[str],
    var_names_vaxis: list[str],
    color=None,
    thinning: float = 1.0,
    **kwargs,
) -> go.Figure:
    """Grid plot with variables from a dataframe.

    This is similar to `pairsplot` but allows to specify different variables for the
    horizontal and vertical axis. Note that this implementation is more memory demending.

    Args:
        df (pandas.Dataframe): dataframe.
        var_names_haxis (list of str): names of variables to be ploted on the horizontal axis.
        var_names_vaxis (list of str): names of variables to be ploted on the vertical axis.
        color (str): name of the column to be used for coloring the points.
        thinning (float): fraction of the data to be used for plotting.
        **kwargs: additional arguments passed to `plotly.express.scatter`.

    Returns:
        plotly figure
    """
    if thinning < 1.0:
        df = df.sample(frac=thinning)

    fig = make_subplots(
        rows=len(var_names_vaxis),
        cols=len(var_names_haxis),
        shared_xaxes=True,
        shared_yaxes=True,
        column_titles=var_names_haxis,
        row_titles=var_names_vaxis,
    )
    if color:
        col = [DEFAULT_COLORSCHEME[i] for i in df[color].astype("category").cat.codes]
    else:
        col = DEFAULT_COLORSCHEME[0]

    rows, cols = fig._get_subplot_rows_columns()
    for i in rows:
        for j in cols:
            fig.add_trace(
                go.Scatter(
                    x=df[var_names_haxis[j - 1]],
                    y=df[var_names_vaxis[i - 1]],
                    mode="markers",
                    marker={"color": col, "opacity": 0.3},
                    showlegend=False,
                    **kwargs,
                ),
                i,
                j,
            )

    fig.update_layout(DEFAULT_LAYOUT)
    fig.update_xaxes(DEFAULT_AXIS)
    fig.update_yaxes(DEFAULT_AXIS)
    fig.update_layout(height=200 * len(rows) + 60, width=200 * len(cols) + 60)
    return fig
