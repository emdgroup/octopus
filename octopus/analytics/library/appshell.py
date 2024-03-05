"""App shell."""

import dash_mantine_components as dmc
from dash import Input, Output, State, clientside_callback, dcc, html, page_container

from octopus.analytics.library import sqlite

# from dash_iconify import DashIconify


def create_main_nav_link(icon, label, href):
    """Create main nav link."""
    return dmc.Anchor(
        dmc.Group(
            [
                # DashIconify(
                #     icon=icon, width=23, color=dmc.theme.DEFAULT_COLORS["orange"][8]
                # ),
                dmc.Text(label, size="sm"),
            ]
        ),
        href=href,
        variant="text",
        mb=5,
    )


def create_home_link(label):
    """Create home link."""
    return dmc.Anchor(
        label,
        size="xl",
        href="/",
        underline=False,
        color=dmc.theme.DEFAULT_COLORS["orange"][8],
    )


def create_side_navbar():
    """Create side navbar."""
    return dmc.Navbar(
        fixed=True,
        id="components-navbar",
        position={"top": 70},
        width={"base": 190},
        children=[
            dmc.Stack(
                spacing="lg",
                mt=20,
                pl=10,
                children=[
                    create_main_nav_link(
                        icon="material-symbols:readiness-score",
                        label="Summary",
                        href="/",
                    ),
                    create_main_nav_link(
                        icon="material-symbols:readiness-score",
                        label="Scores",
                        href="/scores",
                    ),
                    create_main_nav_link(
                        icon="material-symbols:experiment",
                        label="Individual Splits",
                        href="/individual-splits",
                    ),
                    create_main_nav_link(
                        icon="material-symbols:format-color-fill",
                        label="Sequence",
                        href="/sequence",
                    ),
                    create_main_nav_link(
                        icon="material-symbols:format-color-fill",
                        label="Optuna",
                        href="/optuna",
                    ),
                    create_main_nav_link(
                        icon="material-symbols:format-color-fill",
                        label="Configs",
                        href="/configs",
                    ),
                ],
            )
        ],
    )


def create_header():
    """Create header."""
    return dmc.Header(
        height=70,
        fixed=True,
        px=25,
        children=[
            dmc.Stack(
                justify="center",
                style={"height": 70},
                children=dmc.Grid(
                    children=[
                        dmc.Col(
                            [
                                dmc.MediaQuery(
                                    create_home_link("Octopus Analytics"),
                                    smallerThan="lg",
                                    styles={"display": "none"},
                                ),
                            ],
                            span="content",
                            pt=12,
                        ),
                        dmc.Col(
                            [
                                dmc.MediaQuery(
                                    create_home_link(
                                        sqlite.query("SELECT * FROM config_study").iloc[
                                            0, 1
                                        ],
                                    ),
                                    smallerThan="lg",
                                    styles={"display": "none"},
                                ),
                            ],
                            span="content",
                            pt=12,
                        ),
                        dmc.Col(
                            span="auto",
                            children=dmc.Group(
                                position="right",
                                spacing="xl",
                                children=[
                                    dmc.ActionIcon(
                                        # DashIconify(
                                        #     icon="radix-icons:blending-mode", width=22
                                        # ),
                                        variant="outline",
                                        radius=30,
                                        size=36,
                                        color="yellow",
                                        id="color-scheme-toggle",
                                    ),
                                ],
                            ),
                        ),
                    ],
                ),
            )
        ],
    )


def create_table_of_contents(toc_items):
    """Create table of contents."""
    children = []
    for url, name, _ in toc_items:
        children.append(
            html.A(
                dmc.Text(name, color="dimmed", size="sm", variant="text"),
                style={"textTransform": "capitalize", "textDecoration": "none"},
                href=url,
            )
        )

    heading = dmc.Text("Table of Contents", mb=10, weight=500) if children else None

    ad = html.Div(
        **{
            "data-ea-publisher": "dash-mantine-componentscom",
            "data-ea-manual": True,
            "data-ea-type": "text",
        },
        className="flat",
        style={"marginBottom": 25, "marginLeft": -15},
    )

    toc = dmc.Stack([ad, heading, *children], spacing=4, px=25, mt=20)

    return dmc.Aside(
        position={"top": 70, "right": 0},
        fixed=True,
        id="toc-navbar",
        width={"base": 300},
        zIndex=10,
        children=toc,
        withBorder=False,
    )


def create_appshell():
    """Create app shell."""
    return dmc.MantineProvider(
        dmc.MantineProvider(
            theme={
                # "fontFamily": "'Inter', sans-serif",
                "primaryColor": "orange",
                "components": {
                    "Button": {"styles": {"root": {"fontWeight": 400}}},
                    "Alert": {"styles": {"title": {"fontWeight": 500}}},
                    "AvatarGroup": {"styles": {"truncated": {"fontWeight": 500}}},
                },
            },
            inherit=True,
            children=[
                dcc.Store(
                    id="theme-store",
                    storage_type="local",
                ),
                dcc.Location(id="url", refresh="callback-nav"),
                dmc.NotificationsProvider(
                    [
                        create_header(),
                        create_side_navbar(),
                        html.Div(
                            dmc.Container(size="lg", pt=90, children=page_container),
                            id="wrapper",
                        ),
                    ]
                ),
            ],
        ),
        theme={"colorScheme": "dark"},
        id="mantine-docs-theme-provider",
        withGlobalStyles=True,
        withNormalizeCSS=True,
    )


clientside_callback(
    """ function(data) { return data; } """,
    Output("mantine-docs-theme-provider", "theme"),
    Input("theme-store", "data"),
)

clientside_callback(
    """function(n_clicks, data) {
        if (data) {
            if (n_clicks) {
                const scheme = data["colorScheme"] == "dark" ? "light" : "dark"
                return { colorScheme: scheme }
            }
            return dash_clientside.no_update
        } else {
            return { colorScheme: "light" }
        }
    }""",
    Output("theme-store", "data"),
    Input("color-scheme-toggle", "n_clicks"),
    State("theme-store", "data"),
)
