"""Dashboard appshell."""

from typing import List

import dash_mantine_components as dmc
from dash import Input, Output, State, clientside_callback, dcc, page_container

from octopus.dashboard.components.header import create_header
from octopus.dashboard.components.navbar import create_navbar, create_navbar_drawer
from octopus.dashboard.library.constants import COLORS, PRIMARY_COLOR


def create_appshell(
    data: List, show_results: bool, db_filename: str
) -> dmc.MantineProvider:
    """Create appshell."""
    return dmc.MantineProvider(
        id="m2d-mantine-provider",
        forceColorScheme="dark",
        theme={
            "colors": COLORS,
            "primaryColor": PRIMARY_COLOR,
            "fontFamily": "'Inter', sans-serif",
            "components": {
                "Button": {"defaultProps": {"fw": 400}},
                "Alert": {"styles": {"title": {"fontWeight": 500}}},
                "AvatarGroup": {"styles": {"truncated": {"fontWeight": 500}}},
                "Badge": {"styles": {"root": {"fontWeight": 500}}},
                "Progress": {"styles": {"label": {"fontWeight": 500}}},
                "RingProgress": {"styles": {"label": {"fontWeight": 500}}},
                "CodeHighlightTabs": {"styles": {"file": {"padding": 12}}},
                "Table": {
                    "defaultProps": {
                        "highlightOnHover": True,
                        "withTableBorder": True,
                        "verticalSpacing": "sm",
                        "horizontalSpacing": "md",
                    }
                },
            },
        },
        children=[
            dcc.Store(id="theme-store", storage_type="local", data="light"),
            dcc.Store(id="store_db_filename", data=db_filename),
            dcc.Store(id="store_show_results", data=show_results),
            dcc.Location(id="url", refresh="callback-nav"),
            dmc.NotificationProvider(),
            dmc.AppShell(
                [
                    create_header(),
                    create_navbar(data, show_results),
                    create_navbar_drawer(data, show_results),
                    dmc.AppShellMain(children=page_container),
                ],
                header={"height": 70},
                padding="xl",
                zIndex=1400,
                navbar={
                    "width": 300,
                    "breakpoint": "lg",
                    "collapsed": {"mobile": True},
                },
            ),
        ],
    )


clientside_callback(
    """
    function(data) {
        return data
    }
    """,
    Output("m2d-mantine-provider", "forceColorScheme"),
    Input("theme-store", "data"),
)

clientside_callback(
    """
    function(n_clicks, data) {
        return data === "dark" ? "light" : "dark";
    }
    """,
    Output("theme-store", "data"),
    Input("color-scheme-toggle", "n_clicks"),
    State("theme-store", "data"),
    prevent_initial_call=True,
)
