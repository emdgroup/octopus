"""Dashboard header."""

import dash_mantine_components as dmc
from dash import Input, Output, clientside_callback
from dash_iconify import DashIconify


def create_header():
    """Create header."""
    return dmc.AppShellHeader(
        px=25,
        children=[
            dmc.Stack(
                justify="center",
                h=70,
                children=dmc.Grid(
                    justify="space-between",
                    children=[
                        dmc.GridCol(
                            dmc.Group(
                                [
                                    dmc.Anchor(
                                        "Octopus", size="xl", href="/", underline=False
                                    ),
                                ]
                            ),
                            span="content",
                        ),
                        dmc.GridCol(
                            span="auto",
                            children=dmc.Group(
                                justify="flex-end",
                                h=31,
                                gap="xl",
                                children=[
                                    dmc.ActionIcon(
                                        [
                                            DashIconify(
                                                icon="radix-icons:sun",
                                                width=25,
                                                id="light-theme-icon",
                                            ),
                                            DashIconify(
                                                icon="radix-icons:moon",
                                                width=25,
                                                id="dark-theme-icon",
                                            ),
                                        ],
                                        variant="transparent",
                                        color="yellow",
                                        id="color-scheme-toggle",
                                        size="lg",
                                    ),
                                    dmc.ActionIcon(
                                        DashIconify(
                                            icon="radix-icons:hamburger-menu",
                                            width=25,
                                        ),
                                        id="drawer-hamburger-button",
                                        variant="transparent",
                                        size="lg",
                                        hiddenFrom="sm",
                                    ),
                                ],
                            ),
                        ),
                    ],
                ),
            )
        ],
    )


clientside_callback(
    """
    function(value) {
        if (value) {
            return value
        }
    }
    """,
    Output("url", "href"),
    Input("select-component", "value"),
)

clientside_callback(
    """function(n_clicks) { return true }""",
    Output("components-navbar-drawer", "opened"),
    Input("drawer-hamburger-button", "n_clicks"),
    prevent_initial_call=True,
)
