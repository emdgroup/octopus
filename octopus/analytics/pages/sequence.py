"""Page config."""

import dash
import dash_mantine_components as dmc
from dash import html

dash.register_page(
    __name__,
    "/sequence",
    title="Configurations",
    description="",
)


layout = html.Div(
    [
        dmc.Container(
            size="lg",
            mt=30,
            children=["Nothing to see here"],
        )
    ]
)
