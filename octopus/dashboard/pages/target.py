"""EDA target page."""

import dash
import dash_mantine_components as dmc
import plotly.express as px
from dash import Input, Output, State, callback, dcc, html

from octopus.dashboard.library import utils
from octopus.dashboard.library.api.sqlite import SqliteAPI
from octopus.dashboard.library.constants import PAGE_TITLE_PREFIX

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
                dmc.Title("Histogramm"),
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
    State("store_db_filename", "data"),
)
def update_feature_histogram(nbins, theme, db_filename):
    """Select feature for histogram."""
    df_data = SqliteAPI(db_filename).query("SELECT * FROM dataset")
    target = utils.get_col_from_type(db_filename, "Target")
    return px.histogram(
        df_data, x=target, nbins=nbins, template=utils.get_template(theme)
    )
