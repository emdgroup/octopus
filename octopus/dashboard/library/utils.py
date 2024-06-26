"""Dashboard utils."""

import json
from ast import literal_eval
from pathlib import Path
from typing import List

import dash_mantine_components as dmc
import pandas as pd
import plotly.graph_objects as go
from dash import html
from dash_iconify import DashIconify

from octopus.dashboard.library.api.sqlite import SqliteAPI

with open(
    Path("octopus")
    .joinpath("dashboard")
    .joinpath("assets")
    .joinpath("plolty_dark_theme.json"),
    "r",
    encoding="utf-8",
) as file:
    plotly_dark = json.load(file)

with open(
    Path("octopus")
    .joinpath("dashboard")
    .joinpath("assets")
    .joinpath("plotly_light_theme.json"),
    "r",
    encoding="utf-8",
) as file:
    plotly_light = json.load(file)


def create_heading(text: str) -> dmc.Text:
    """Create headings."""
    return dmc.Text(text, ta="center", mt=10, mb=20, mx=0)


def create_tile(icon: str, heading: str, description: str, href: str) -> html.A:
    """Create tile."""
    return html.A(
        dmc.Card(
            radius="md",
            p="xl",
            withBorder=True,
            m=5,
            children=[
                DashIconify(
                    icon=icon,
                    height=20,
                    color="#2DBECD",
                ),
                dmc.Text(heading, size="lg", mt="md"),
                dmc.Text(description, size="sm", c="dimmed", mt="sm"),
            ],
        ),
        href=href,
        style={"textDecoration": "none"},
    )


def get_template(theme: str) -> dict:
    """Get template for plotly graph."""
    if theme == "light":
        return {"layout": go.Layout(**plotly_light)}
    else:
        return {"layout": go.Layout(**plotly_dark)}


def table_without_header(df: pd.DataFrame) -> dmc.Table:
    """Create table without header."""
    rows = [dmc.TableTr([dmc.TableTd(cell) for cell in row]) for row in df.values]
    body = dmc.TableTbody(rows)
    return dmc.Table(
        body,
        verticalSpacing="xs",
        striped=True,
        withTableBorder=False,
        withColumnBorders=True,
        layout="fixed",
    )


def get_target_metric(db_filename: str) -> str:
    """Get target metric."""
    return SqliteAPI(db_filename).query(
        """
            SELECT Value
            FROM config_study
            WHERE Parameter="target_metric"
            """
    )["Value"][0]


def create_config_output(data: pd.DataFrame) -> str:
    """Create config output as str."""
    dict_data = data.set_index("Parameter").to_dict()["Value"]
    my_dict_cleaned = {}
    for key, value in dict_data.items():
        if isinstance(value, str):
            if value.isnumeric():
                my_dict_cleaned[key] = int(value)
            elif value.replace(".", "", 1).isdigit():
                my_dict_cleaned[key] = float(value)
            elif value.startswith("[") and value.endswith("]"):
                my_dict_cleaned[key] = literal_eval(value)
            else:
                my_dict_cleaned[key] = value
        else:
            my_dict_cleaned[key] = value

    return str(my_dict_cleaned)


def get_plot_color(name: str) -> str:
    """Get plot color."""
    colors = {
        "train": "#2DBECD",
        "dev": "#EB3C96",
        "test": "#FFC832",
        "average": "#FFC832",
        "ensemble": "#FFC832",
    }

    return colors[name]


def get_col_from_type(db_filename: str, type: str) -> List | str:
    """Get column name for specific column type.

    Value are Feature, Target, Row_ID or Datasplit.
    """
    columns = (
        SqliteAPI(db_filename)
        .query(
            f"""
        SELECT Column
        FROM dataset_info
        WHERE Type = "{type}"
    """
        )["Column"]
        .values.tolist()
    )

    if len(columns) == 1:
        return columns[0]
    return columns
