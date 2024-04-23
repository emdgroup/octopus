"""EDA target page."""

import dash
import dash_mantine_components as dmc
import plotly.express as px
from dash import Input, Output, callback, dcc, html

from octopus.dashboard.lib import utils
from octopus.dashboard.lib.api import sqlite
from octopus.dashboard.lib.constants import PAGE_TITLE_PREFIX

df_data = sqlite.query("SELECT * FROM data")
target_col = sqlite.query(
    """
    SELECT Column
    FROM column_description
    WHERE Type = 'Target'
    """
)["Column"].values[0]


dash.register_page(
    __name__,
    "/target",
    title=PAGE_TITLE_PREFIX + "Target",
    description="Information about the target column.",
)

layout = html.Div(
    [
        dmc.Container(
            size="lg",
            mt=50,
            children=dmc.Title("Target"),
        ),
        dmc.Container(
            size="lg",
            mt=50,
            children=[
                utils.create_title("Histogramm", comp_id="eda_target_histo"),
                dmc.NumberInput(
                    id="number_input_eda_nbins_target",
                    label="Number of bins",
                    value=20,
                    min=0,
                    step=5,
                    w=250,
                ),
                dcc.Graph(
                    id="graph_eda_target_histo",
                ),
            ],
        ),
    ]
)


@callback(
    Output("graph_eda_target_histo", "figure"),
    Input("number_input_eda_nbins_target", "value"),
    Input("theme-store", "data"),
)
def update_feature_histogram(nbins, theme):
    """Select feature for histogram."""
    return px.histogram(
        df_data, x=target_col, nbins=nbins, template=utils.get_template(theme)
    )
