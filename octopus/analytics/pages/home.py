"""Home."""

import dash
import dash_mantine_components as dmc
from dash import Input, Output, callback, dcc, html

from octopus.analytics.lib import sqlite, utils

dash.register_page(
    __name__,
    "/",
    title="Octopus",
    description="Octopus main page",
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
                dmc.Table(
                    utils.create_table_without_header(
                        sqlite.query("SELECT * FROM config_study")
                    ),
                ),
                dmc.Title("Manager configuration", pb=20, pt=40),
                dmc.Table(
                    utils.create_table_without_header(
                        sqlite.query("SELECT * FROM config_manager")
                    )
                ),
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
