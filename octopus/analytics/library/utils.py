"""Utils."""

import dash_mantine_components as dmc
from dash import html

from octopus.analytics.library import sqlite


def table_without_header(df):
    """Create table without header."""
    rows = [html.Tr([html.Td(cell) for cell in row]) for row in df.values]
    table = [html.Tbody(rows)]
    return dmc.Table(table)


def get_target_metric():
    """Get target metric."""
    return sqlite.query(
        "SELECT * FROM config_study WHERE \"index\" = 'target_metric'"
    ).at[0, "0"]
