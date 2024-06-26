"""Results configuration page."""

import dash
import dash_ag_grid as dag
import dash_mantine_components as dmc
from dash import Input, Output, State, callback, clientside_callback, html

from octopus.dashboard.library import utils
from octopus.dashboard.library.api.sqlite import SqliteAPI
from octopus.dashboard.library.constants import PAGE_TITLE_PREFIX

dash.register_page(
    __name__,
    "/basics",
    title=PAGE_TITLE_PREFIX + "Basic Information",
    description="Basics Information about the data.",
)


layout = html.Div(
    [
        dmc.Container(
            size="lg",
            mt=50,
            id="container_df_description",
            children=[],
        ),
        dmc.Container(
            size="lg",
            mt=50,
            children=[
                dmc.Title("Dataframe"),
                dag.AgGrid(
                    id="table_eda_data",
                    rowData=[],
                    defaultColDef={"filter": True},
                    columnSize="autoSize",
                    dashGridOptions={
                        "skipHeaderOnAutoSize": True,
                    },
                    className="ag-theme-alpine-dark",
                ),
            ],
        ),
    ]
)


@callback(
    Output("container_df_description", "children"),
    Output("table_eda_data", "rowData"),
    Output("table_eda_data", "columnDefs"),
    Input("url", "pathname"),
    State("store_db_filename", "data"),
)
def add_data(_, db_filename):
    """Add initial data."""
    df_description = SqliteAPI(db_filename).query(
        "SELECT Description, Value FROM description"
    )
    df_data = SqliteAPI(db_filename).query("SELECT * FROM dataset")
    return (
        [
            dmc.Title("Dataframe description"),
            utils.table_without_header(df_description),
        ],
        df_data.to_dict("records"),
        [{"field": col} for col in df_data.columns],
    )


clientside_callback(
    """
    function(data) {
        // Return the class name based on the colorScheme value
        if (data === "light") {
            return 'ag-theme-alpine';
        } else {
            // Handle other colorScheme values
            return 'ag-theme-alpine-dark';
        }
    }
    """,
    Output("table_eda_data", "className"),
    Input("theme-store", "data"),
)
