import plotly

DEFAULT_LAYOUT = {
    "plot_bgcolor": "rgba(10, 10, 10, 0.1)",
    "paper_bgcolor": "rgba(0, 0, 0, 0)",
    "font": {"color": "rgba(0.5, 0.5, 0.5, 1.0)", "size": 16},
    "margin": {"l": 50, "r": 50, "t": 50, "b": 50},
    "hovermode": "closest",
}

DEFAULT_AXIS = {
    "showgrid": True,
    "zeroline": False,
    "showline": True,
    "mirror": "ticks",
    "linecolor": "rgba(0.5, 0.5, 0.5, 0.5)",
    "gridcolor": "rgba(0.5, 0.5, 0.5, 0.2)",
}

DEFAULT_COLORSCHEME = plotly.colors.qualitative.Plotly
