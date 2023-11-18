import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

colorscheme = px.colors.qualitative.Plotly


def pairsplot(df: pd.DataFrame, var_names: list[str], **kwargs) -> go.Figure:
    """Grid plot with variables from a dataframe.

    Args:
        df (panda.Dataframe): dataframe.
        var_names (list of str): names of variables to be ploted on the horizontal axis.
        **kwargs: additional arguments passed to `plotly.express.scatter_matrix`.


    Returns:
        plotly figure
    """
    fig = px.scatter_matrix(df, dimensions=var_names, **kwargs)
    fig.update_traces(diagonal_visible=False)
    return fig


def gridplot(
    df: pd.DataFrame, var_names_haxis: list[str], var_names_vaxis: list[str], **kwargs
) -> go.Figure:
    """Grid plot with variables from a dataframe.

    This is similar to `pairsplot` but allows to specify different variables for the
    horizontal and vertical axis.

    Args:
        df (pandas.Dataframe): dataframe.
        var_names_haxis (list of str): names of variables to be ploted on the horizontal axis.
        var_names_vaxis (list of str): names of variables to be ploted on the vertical axis.
        **kwargs: additional arguments passed to `plotly.express.scatter`.

    Returns:
        plotly figure
    """
    fig = make_subplots(
        rows=len(var_names_vaxis),
        cols=len(var_names_haxis),
        shared_xaxes=True,
        shared_yaxes=True,
        column_titles=var_names_haxis,
        row_titles=var_names_vaxis,
    )

    rows, cols = fig._get_subplot_rows_columns()
    for i in rows:
        for j in cols:
            fig.add_trace(
                go.Scatter(
                    x=df[var_names_haxis[j - 1]],
                    y=df[var_names_vaxis[i - 1]],
                    mode="markers",
                    marker={"color": colorscheme[0], "opacity": 0.3},
                    showlegend=False,
                    **kwargs
                ),
                i,
                j,
            )

    fig.update_layout({"plot_bgcolor": "rgba(10, 10, 10, .1)", "paper_bgcolor": "rgba(0, 0, 0, 0)"})
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(showgrid=False, zeroline=False)
    fig.update_layout(height=300 * len(rows), width=300 * len(cols))
    return fig
