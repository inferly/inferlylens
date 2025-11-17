import gpflow
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import tensorflow as tf
from plotly.subplots import make_subplots

from ..data import Dataset
from .gpmodel import GPmodel

color = px.colors.qualitative.Plotly


def hex_to_rgb(hex_color: str, opacity: float = 1.0) -> str:
    """Convert a hex color to rgb."""
    hex_color = hex_color.lstrip("#")
    rgb = tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, {opacity})"


def plot_slices(
    model: GPmodel,
    input_names: list[str],
    output_names: list[str],
    reference_point: tf.Tensor,
    xlim: tf.Tensor,
):
    """Plot slices of the posterior distribution at a reference point.

    Args:
        model (GPmodel): A GPmodel object.
        input_names (list of strings): the name of the variables associated with each input dimension.
        output_names (list of strings): the name of the variables associated with each output dimension.
        reference_point (tf.Tensor): A reference point in the input space with shape [1, dim]. All slices will
           cut through this point.
        xlim (tf.Tensor): A tensor of shape [2, dim] with the lower and upper limits of the x-axis.
    """
    N = 100
    x = np.linspace(0, 1, N)
    fig = make_subplots(
        rows=len(output_names),
        cols=len(input_names),
        shared_xaxes=True,
        shared_yaxes=True,
        column_titles=input_names,
        row_titles=output_names,
    )
    rows, cols = fig._get_subplot_rows_columns()
    for i in rows:
        for j in cols:
            x = np.zeros((N, 1)) + reference_point
            xsweep = np.linspace(xlim[0, j - 1], xlim[1, j - 1], N)
            x[:, j - 1] = xsweep
            mean, var = (_.numpy()[:, i - 1] for _ in model.predict_y(x))
            up95 = mean + 2 * np.sqrt(var)
            low95 = mean - 2 * np.sqrt(var)
            fig.add_trace(go.Scatter(x=xsweep, y=mean, line_color=color[0], showlegend=False), i, j)
            fig.add_trace(
                go.Scatter(
                    x=xsweep,
                    y=up95,
                    line_color=color[0],
                    showlegend=False,
                ),
                i,
                j,
            )
            fig.add_trace(
                go.Scatter(
                    x=xsweep,
                    y=low95,
                    fill="tonexty",
                    line_color=color[0],
                    showlegend=False,
                ),
                i,
                j,
            )
    fig.update_layout(height=300 * len(rows), width=300 * len(cols))
    return fig


def plot_lengthscales(kernel: gpflow.kernels.Kernel, inputs: list[str], **kwargs):
    """Spider chart to inspect the lengthscales.

    Args:
        kernel (gpflow kernel): ARD kernel from a GPflow model.
        inputs (list of strings): the name of the variables associated with each input dimension.
        **kwargs: keyword arguments passed to `plotly.express.line_polar`.
    """
    assert hasattr(kernel, "lengthscales"), "The kernel must have an attribute `lengthscales`."
    df = pd.DataFrame({"inputs": inputs, "lengthscales": kernel.lengthscales.numpy().squeeze()})

    return px.line_polar(df, r="lengthscales", theta="inputs", line_close=True, **kwargs)


def plot_actual_vs_predicted(model: GPmodel, data: Dataset) -> go.Figure:
    """Plot actual vs predicted values.

    Args:
        model (GPmodel): A GPmodel object.
        data (inferlycore Dataset): Dataset to compare model train/predictions.
    """
    mean, _ = model.predict_y(data.df[data.input_names].values)
    lower, upper = model.predict_quantiles(data.df[data.input_names].values, np.array([0.025, 0.975]))
    lower, mean, upper = lower.numpy(), mean.numpy(), upper.numpy()
    fig = make_subplots(rows=1, cols=len(data.output_names), subplot_titles=data.output_names)
    for i, output_name in enumerate(data.output_names):
        fig.add_trace(
            go.Scatter(
                x=data.df[output_name],
                y=mean[:, i],
                mode="markers",
                error_y=dict(
                    type="data",
                    symmetric=False,
                    array=upper[:, i] - mean[:, i],
                    arrayminus=mean[:, i] - lower[:, i],
                    color=hex_to_rgb(color[0], opacity=0.1),
                ),
                marker_color=hex_to_rgb(color[0]),
                name=output_name,
            ),
            1,
            i + 1,
        )
        lim = [
            np.minimum(data.df[output_name].min(), mean[:, i].min()),
            np.maximum(data.df[output_name].max(), mean[:, i].max()),
        ]
        fig.add_trace(
            go.Scatter(
                x=lim,
                y=lim,
                mode="lines",
                opacity=0.5,
                line=dict(color="black", dash="dash"),
                showlegend=False,
            ),
            1,
            i + 1,
        )

    fig.update_layout(
        {
            "title": "Actual vs Predicted",
            "height": 400,
            "width": 260 * len(data.output_names),
            "showlegend": False,
        }
    )

    return fig
