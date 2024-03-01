"""Home."""

import dash
import dash_mantine_components as dmc
from dash import Input, Output, callback, dcc, html

from octopus.analytics.library import sqlite, utils

dash.register_page(
    __name__,
    "/configs",
    title="Configurations",
    description="Configurations",
)


layout = html.Div(
    [
        dmc.Container(
            size="lg",
            mt=30,
            children=[
                dmc.Grid(
                    [
                        dmc.Col(dmc.Title("Study configuration"), span="content"),
                        dmc.Col(
                            dcc.Clipboard(
                                id="clipboard_study_config",
                            ),
                            span="auto",
                        ),
                    ],
                    pb=20,
                ),
                utils.table_without_header(sqlite.query("SELECT * FROM config_study")),
                dmc.Title("Manager configuration", pb=20, pt=40),
                utils.table_without_header(
                    sqlite.query("SELECT * FROM config_manager")
                ),
                dmc.Title("Sequence configuration", pb=20, pt=40),
                dmc.AccordionMultiple(id="accordion_sequence_config"),
            ],
        )
    ]
)


@callback(
    Output("clipboard_study_config", "content"),
    Input("clipboard_study_config", "n_clicks"),
)
def custom_copy(_):
    """Copy config study.

    There must be a better way to do it.
    """
    dict_study_config = (
        sqlite.query("SELECT * FROM config_study").set_index("index").to_dict()["0"]
    )

    my_dict_cleaned = {}
    for key, value in dict_study_config.items():
        if isinstance(value, str):
            if value.isnumeric():
                my_dict_cleaned[key] = int(value)
            elif value.replace(".", "", 1).isdigit():
                my_dict_cleaned[key] = float(value)
            elif value.startswith("[") and value.endswith("]"):
                my_dict_cleaned[key] = eval(value)
            else:
                my_dict_cleaned[key] = value
        else:
            my_dict_cleaned[key] = value

    return str(my_dict_cleaned)


@callback(
    Output("accordion_sequence_config", "children"),
    Input("url", "pathname"),
)
def create_accordion_items(_):
    """Create accordion items."""
    accordion_items = []
    for value, df_ in sqlite.query("SELECT * FROM config_sequence").groupby(
        "sequence_id"
    ):
        accordion_items.append(
            dmc.AccordionItem(
                [
                    dmc.AccordionControl(f"Sequence_{value}"),
                    dmc.AccordionPanel(utils.table_without_header(df_[["index", "0"]])),
                ],
                value=f"Sequence {value}",
            )
        )

    return accordion_items
