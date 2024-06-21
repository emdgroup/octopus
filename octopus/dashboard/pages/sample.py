"""EDA sample page."""

import dash
import dash_mantine_components as dmc
import plotly.express as px
from dash import Input, Output, callback, dcc, html

from octopus.dashboard.library import utils
from octopus.dashboard.library.api.sqlite import SqliteAPI
from octopus.dashboard.library.constants import PAGE_TITLE_PREFIX

dash.register_page(
    __name__,
    "/sample",
    title=PAGE_TITLE_PREFIX + "Sample Information",
    description="Basics Information about the sample.",
)

sqlite = SqliteAPI()
df_data = sqlite.query("SELECT * FROM dataset")

sample_id = sqlite.query(
    """
    SELECT Column
    FROM dataset_info
    WHERE Type = 'Sample_ID'
    """
)["Column"].values[0]


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
)
def update_feature_histogram(theme):
    """Select feature for histogram."""
    fig = px.bar(
        x=df_data[sample_id].value_counts().index.astype(str).tolist(),
        y=df_data[sample_id].value_counts().values.tolist(),
        labels={"x": sample_id, "y": "count"},
        template=utils.get_template(theme),
    )
    fig.update_layout(xaxis={"showticklabels": False})

    return fig
