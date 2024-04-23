"""Results configuration page."""

import dash
import dash_ag_grid as dag
import dash_mantine_components as dmc
from dash import Input, Output, clientside_callback, html

from octopus.dashboard.lib import utils
from octopus.dashboard.lib.api import sqlite
from octopus.dashboard.lib.constants import PAGE_TITLE_PREFIX
from octopus.dashboard.lib.directives.toc import TOC

dash.register_page(
    __name__,
    "/basics",
    title=PAGE_TITLE_PREFIX + "Basic Information",
    description="Basics Information about the data.",
)

df_description = sqlite.query("SELECT Description, Value FROM description")
df_data = sqlite.query("SELECT * FROM data")


layout = html.Div(
    [
        dmc.Container(
            size="lg",
            mt=50,
            children=[
                utils.create_title(
                    "Dataframe description",
                    comp_id="eda_dataframe_description",
                ),
            ],
        ),
        dmc.Container(
            size="lg",
            mt=50,
            children=[
                utils.table_without_header(df_description),
            ],
        ),
        dmc.Container(
            size="lg",
            mt=50,
            children=[
                utils.create_title(
                    "Dataframe",
                    comp_id="eda_dataframe",
                ),
            ],
        ),
        dmc.Container(
            size="lg",
            mt=50,
            children=[
                dag.AgGrid(
                    id="table_eda_data",
                    rowData=df_data.to_dict("records"),
                    columnDefs=[{"field": col} for col in df_data.columns],
                    defaultColDef={"filter": True},
                    columnSize="autoSize",
                    dashGridOptions={
                        "skipHeaderOnAutoSize": True,
                    },
                    className="ag-theme-alpine-dark",
                ),
            ],
        ),
        TOC.render(
            None,
            None,
            "Table of Contents",
            None,
            **{
                "table_of_contents": [
                    (3, "Description", "eda_dataframe_description"),
                    (3, "Dataframe", "eda_dataframe"),
                ]
            },
        ),
    ]
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
