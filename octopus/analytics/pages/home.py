"""Home."""

import dash
import dash_mantine_components as dmc
from dash import html

dash.register_page(
    __name__,
    "/",
    title="Octopus",
    description="Octopus main page",
)


layout = html.Div([dmc.Container(size="lg", mt=30, children=["Octopus"])])
