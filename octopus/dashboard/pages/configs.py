"""Home."""

import dash
import dash_mantine_components as dmc
from dash import MATCH, Input, Output, State, callback, dcc, html

from octopus.dashboard.library import utils
from octopus.dashboard.library.api.sqlite import SqliteAPI
from octopus.dashboard.library.constants import PAGE_TITLE_PREFIX

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
                        dmc.Title("Study"),
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
                        dmc.Title("Manager"),
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
                dmc.Title("Sequence"),
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
    State("store_db_filename", "data"),
)
def create_tables(_, db_filename):
    """Copy config study."""
    return (
        utils.table_without_header(
            SqliteAPI(db_filename).query("SELECT Parameter, Value FROM config_study")
        ),
        utils.table_without_header(
            SqliteAPI(db_filename).query("SELECT Parameter, Value FROM config_manager")
        ),
    )


@callback(
    Output("clipboard_study_config", "content"),
    Input("clipboard_study_config", "n_clicks"),
    State("store_db_filename", "data"),
    prevent_initial_call=True,
)
def copy_study_to_clipboard(_, db_filename):
    """Copy config study."""
    return utils.create_config_output(
        SqliteAPI(db_filename).query("SELECT Parameter, Value FROM config_study")
    )


@callback(
    Output("clipboard_study_manager", "content"),
    Input("clipboard_study_manager", "n_clicks"),
    State("store_db_filename", "data"),
    prevent_initial_call=True,
)
def copy_manager_to_clipboard(_, db_filename):
    """Copy config study."""
    return utils.create_config_output(
        SqliteAPI(db_filename).query("SELECT Parameter, Value FROM config_manager")
    )


@callback(
    Output({"type": "clipboard_sequence", "index": MATCH}, "content"),
    Input({"type": "clipboard_sequence", "index": MATCH}, "n_clicks"),
    State({"type": "clipboard_sequence", "index": MATCH}, "id"),
    State("store_db_filename", "data"),
    prevent_initial_call=True,
)
def copy_sequence_to_clipboard(_, selected_id, db_filename):
    """Copy config study."""
    index = selected_id["index"]
    return utils.create_config_output(
        SqliteAPI(db_filename).query(
            f"SELECT Parameter, Value FROM config_sequence WHERE sequence_id={index}"
        )
    )


@callback(
    Output("div_config_sequence", "children"),
    Input("url", "pathname"),
    State("store_db_filename", "data"),
)
def create_accordion_items(_, db_filename):
    """Create accordion items."""
    children = []
    for value, df_ in (
        SqliteAPI(db_filename)
        .query("SELECT Parameter, Value, sequence_id FROM config_sequence")
        .groupby("sequence_id")
    ):
        children.append(
            dmc.Group(
                [
                    dmc.Text(f"Sequence_{value}"),
                    dcc.Clipboard(
                        id={"type": "clipboard_sequence", "index": value},
                    ),
                    utils.table_without_header(df_[["Parameter", "Value"]]),
                    dmc.Space(h=30),
                ]
            )
        )

    return children
