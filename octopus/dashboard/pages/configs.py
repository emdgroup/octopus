"""Home."""

import dash
import dash_mantine_components as dmc
from dash import MATCH, Input, Output, State, callback, dcc, html

from octopus.dashboard.lib import utils
from octopus.dashboard.lib.api import sqlite
from octopus.dashboard.lib.constants import PAGE_TITLE_PREFIX

dash.register_page(
    __name__,
    "/configs",
    title=PAGE_TITLE_PREFIX + "Configurations",
    description="Configurations",
)


layout = html.Div(
    [
        dmc.Container(size="lg", mt=50, children=dmc.Title("Configuration")),
        dmc.Container(
            size="lg",
            mt=50,
            children=[
                dmc.Group(
                    [
                        utils.create_title("Study", comp_id="results_config_study"),
                        dcc.Clipboard(
                            id="clipboard_study_config",
                        ),
                    ]
                ),
                html.Div(id="div_results_table_study"),
            ],
        ),
        dmc.Container(
            size="lg",
            mt=50,
            children=[
                dmc.Group(
                    [
                        utils.create_title("Manager", comp_id="results_config_manager"),
                        dcc.Clipboard(
                            id="clipboard_study_manager",
                        ),
                    ]
                ),
                html.Div(id="div_results_table_manager"),
            ],
        ),
        dmc.Container(
            size="lg",
            mt=50,
            children=[
                utils.create_title("Sequence", comp_id="results_config_sequence"),
                html.Div(
                    id="div_config_sequence",
                ),
            ],
        ),
    ]
)


@callback(
    Output("div_results_table_study", "children"),
    Output("div_results_table_manager", "children"),
    Input("url", "pathname"),
)
def create_tables(_):
    """Copy config study."""
    return (
        utils.table_without_header(sqlite.query("SELECT * FROM config_study")),
        utils.table_without_header(sqlite.query("SELECT * FROM config_manager")),
    )


@callback(
    Output("clipboard_study_config", "content"),
    Input("clipboard_study_config", "n_clicks"),
    prevent_initial_call=True,
)
def copy_study_to_clipboard(_):
    """Copy config study."""
    return utils.create_config_output(sqlite.query("SELECT * FROM config_study"))


@callback(
    Output("clipboard_study_manager", "content"),
    Input("clipboard_study_manager", "n_clicks"),
    prevent_initial_call=True,
)
def copy_manager_to_clipboard(_):
    """Copy config study."""
    return utils.create_config_output(sqlite.query("SELECT * FROM config_manager"))


@callback(
    Output({"type": "clipboard_sequence", "index": MATCH}, "content"),
    Input({"type": "clipboard_sequence", "index": MATCH}, "n_clicks"),
    State({"type": "clipboard_sequence", "index": MATCH}, "id"),
    prevent_initial_call=True,
)
def copy_sequence_to_clipboard(_, selected_id):
    """Copy config study."""
    index = selected_id["index"]
    return utils.create_config_output(
        sqlite.query(f"SELECT * FROM config_sequence WHERE sequence_id={index}")
    )


@callback(
    Output("div_config_sequence", "children"),
    Input("url", "pathname"),
)
def create_accordion_items(_):
    """Create accordion items."""
    children = []
    for value, df_ in sqlite.query("SELECT * FROM config_sequence").groupby(
        "sequence_id"
    ):
        children.append(
            dmc.Group(
                [
                    dmc.Text(f"Sequence_{value}"),
                    dcc.Clipboard(
                        id={"type": "clipboard_sequence", "index": value},
                    ),
                    utils.table_without_header(df_[["index", "0"]]),
                    dmc.Space(h=30),
                ]
            )
        )

    return children
