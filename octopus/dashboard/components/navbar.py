"""Dashboard navbar."""

import dash_mantine_components as dmc
from dash_iconify import DashIconify


def create_main_link(icon, label, href):
    """Create main link."""
    return dmc.Anchor(
        dmc.Group(
            [
                DashIconify(
                    icon=icon, width=23, color=dmc.DEFAULT_THEME["colors"]["indigo"][5]
                ),
                dmc.Text(label, size="sm"),
            ]
        ),
        href=href,
        variant="text",
        mb=5,
        underline=False,
    )


def create_content(data, show_results):
    """Create content."""
    links_eda = [
        dmc.NavLink(
            label=entry["name"], href=entry["path"], styles={"root": {"height": 32}}
        )
        for entry in data
        if entry["path"] in ["/target", "/features", "/basics", "/sample"]
    ]
    links_eda.insert(
        0,
        dmc.Divider(
            label=[
                dmc.Text("EDA", ml=5, size="sm"),
            ],
            labelPosition="left",
            mt=40,
            mb=10,
        ),
    )

    links_results = [
        dmc.NavLink(
            label=entry["name"], href=entry["path"], styles={"root": {"height": 32}}
        )
        for entry in data
        if entry["path"]
        in ["/optuna", "/individual-splits", "/configs", "/scores", "/summary"]
    ]
    links_results.insert(
        0,
        dmc.Divider(
            label=[
                dmc.Text("Results", ml=5, size="sm"),
            ],
            labelPosition="left",
            mt=40,
            mb=10,
        ),
    )

    children = links_eda
    if show_results:
        children.extend(links_results)

    return dmc.ScrollArea(
        offsetScrollbars=True,
        type="scroll",
        style={"height": "100%"},
        children=dmc.Stack(gap=0, children=children, px=25),
    )


def create_navbar(data, show_results):
    """Create navbar."""
    return dmc.AppShellNavbar(children=create_content(data, show_results))


def create_navbar_drawer(data, show_results):
    """Create navbar drawer."""
    return dmc.Drawer(
        id="components-navbar-drawer",
        overlayProps={"opacity": 0.55, "blur": 3},
        zIndex=1500,
        pt=20,
        size="100%",
        children=create_content(data, show_results),
    )
