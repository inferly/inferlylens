import plotly.graph_objects as go
import plotly.io as pio

DEFAULT_AXIS = {
    "showgrid": True,
    "zeroline": False,
    "showline": False,
    "mirror": "ticks",
    "linecolor": "rgba(0.5, 0.5, 0.5, 0.2)",
    "gridcolor": "rgba(0.5, 0.5, 0.5, 0.1)",
    "automargin": True,
}

template = go.layout.Template()

template.layout.plot_bgcolor = "rgba(0.5, 0.5, 0.5, 0.1)"
template.layout.paper_bgcolor = "rgba(0, 0, 0, 0)"
template.layout.font = {"color": "rgba(0.5, 0.5, 0.5, 1.0)", "size": 16}

template.layout.polar.bgcolor = "rgba(0.5, 0.5, 0.5, 0.1)"
template.layout.polar.radialaxis.linecolor = "rgba(0.5, 0.5, 0.5, 0.2)"
template.layout.polar.radialaxis.gridcolor = "rgba(0.5, 0.5, 0.5, 0.1)"

template.layout.polar.angularaxis.linecolor = "rgba(0.5, 0.5, 0.5, 0.2)"
template.layout.polar.angularaxis.gridcolor = "rgba(0.5, 0.5, 0.5, 0.1)"

template.layout.xaxis = DEFAULT_AXIS
template.layout.yaxis = DEFAULT_AXIS

pio.templates["inferly"] = template

pio.templates.default = "plotly+inferly"
