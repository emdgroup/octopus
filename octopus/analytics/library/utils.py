"""Utils."""

from dash import html


def create_table_without_header(df):
    """Create table without header."""
    rows = [html.Tr([html.Td(cell) for cell in row]) for row in df.values]
    table = [html.Tbody(rows)]
    return table
