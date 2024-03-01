"""Utils."""

import dash_mantine_components as dmc
from dash import html


def table_without_header(df):
    """Create table without header."""
    rows = [html.Tr([html.Td(cell) for cell in row]) for row in df.values]
    table = [html.Tbody(rows)]
    return dmc.Table(table)
