"""EDA sample page."""

import dash
import dash_mantine_components as dmc
import plotly.express as px
from dash import Input, Output, State, callback, dcc, html

from octopus.dashboard.library import utils
from octopus.dashboard.library.api.sqlite import SqliteAPI
from octopus.dashboard.library.constants import PAGE_TITLE_PREFIX

dash.register_page(
    __name__,
    "/sample",
    title=PAGE_TITLE_PREFIX + "Sample Information",
    description="Basics Information about the sample.",
)

layout = html.Div(
    [
        dmc.Container(
            size="lg",
            mt=50,
            children=[
                dmc.Title("Sample ID"),
            ],
        ),
        dmc.Container(
            size="lg",
            mt=50,
            children=[
                dcc.Graph(
                    id="graph_eda_sample",
                ),
            ],
        ),
    ]
)


@callback(
    Output("graph_eda_sample", "figure"),
    Input("theme-store", "data"),
    State("store_db_filename", "data"),
)
def update_feature_histogram(theme, db_filename):
    """Select feature for histogram."""
    sample_id = (
        SqliteAPI(db_filename)
        .query(
            """
        SELECT Column
        FROM dataset_info
        WHERE Type = 'Sample_ID'
        """
        )["Column"]
        .values[0]
    )
    df_data = SqliteAPI(db_filename).query("SELECT * FROM dataset")
    fig = px.bar(
        x=df_data[sample_id].value_counts().index.astype(str).tolist(),
        y=df_data[sample_id].value_counts().values.tolist(),
        labels={"x": sample_id, "y": "count"},
        template=utils.get_template(theme),
    )
    fig.update_layout(xaxis={"showticklabels": False})

    return fig
