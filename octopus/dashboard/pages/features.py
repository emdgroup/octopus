"""Results features page."""

import dash
import dash_mantine_components as dmc
import plotly.express as px
from dash import Input, Output, callback, dcc, html

from octopus.dashboard.library import utils
from octopus.dashboard.library.api.sqlite import SqliteAPI
from octopus.dashboard.library.constants import PAGE_TITLE_PREFIX

sqlite = SqliteAPI()
df_data = sqlite.query("SELECT * FROM dataset")
target = utils.get_col_from_type("Target")
features = utils.get_col_from_type("Feature")
features = [col for col in features if col in df_data.columns]

dash.register_page(
    __name__,
    "/features",
    title=PAGE_TITLE_PREFIX + "Features",
    description="Information about the features.",
)

layout = html.Div(
    [
        dmc.Container(size="lg", mt=50, children=dmc.Title("Features")),
        dmc.Container(
            size="lg",
            mt=50,
            children=[
                dmc.Select(
                    id="select_feature",
                    label="Select Feature",
                    value=features[0],
                    data=[{"value": i, "label": i} for i in features],
                ),
            ],
        ),
        dmc.Container(
            size="lg",
            mt=50,
            children=[
                dmc.Title("Histogramm"),
                dmc.NumberInput(
                    id="number_input_eda_nbins_feature",
                    label="Number of bins",
                    value=20,
                    min=0,
                    step=5,
                    w=250,
                ),
                dmc.Grid(
                    [
                        dmc.GridCol(dcc.Graph(id="graph_feature_hist"), span=9),
                        dmc.GridCol(
                            id="col_feature_description",
                            span=3,
                        ),
                    ]
                ),
            ],
        ),
        dmc.Container(
            size="lg",
            mt=50,
            children=[
                dmc.Title("Box plot"),
                dcc.Graph(id="graph_box_feature"),
            ],
        ),
        dmc.Container(
            size="lg",
            mt=50,
            children=[
                dmc.Title("Correlaction matrix"),
                dcc.Graph(
                    id="graph_correlation",
                ),
            ],
        ),
    ]
)


@callback(
    Output("graph_feature_hist", "figure"),
    Output("graph_box_feature", "figure"),
    Output("col_feature_description", "children"),
    Output("graph_correlation", "figure"),
    Input("select_feature", "value"),
    Input("number_input_eda_nbins_feature", "value"),
    Input("theme-store", "data"),
)
def update_feature_histogram(feature, nbins, theme):
    """Select feature for histogram."""
    df_feature_description = df_data[feature].describe().reset_index()
    return (
        px.histogram(
            df_data, x=feature, nbins=nbins, template=utils.get_template(theme)
        ),
        px.box(df_data, x=target, y=feature, template=utils.get_template(theme)),
        utils.table_without_header(df_feature_description),
        px.imshow(
            df_data[features].corr(),
            text_auto=True,
            aspect="auto",
            color_continuous_scale="Viridis",
            template=utils.get_template(theme),
        ),
    )
