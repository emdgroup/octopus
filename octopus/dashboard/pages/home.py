"""Home page."""

import dash
import dash_mantine_components as dmc
from dash import Input, Output, callback, html

from octopus.dashboard.library import utils
from octopus.dashboard.library.constants import PAGE_TITLE_PREFIX

dash.register_page(
    __name__,
    "/",
    title=PAGE_TITLE_PREFIX + "Home",
    description="Starting page for Octopus.",
)


layout = html.Div(
    [
        dmc.Container(
            size="lg",
            mt=30,
            children=[dmc.Title("Exploratory Data Analysis")],
        ),
        dmc.Container(
            size="lg",
            px=0,
            py=0,
            my=40,
            children=[
                dmc.SimpleGrid(
                    mt=80,
                    cols={"xs": 1, "sm": 2, "xl": 3},
                    children=[
                        utils.create_tile(
                            icon="akar-icons:calendar",
                            heading="Basic Information",
                            description="Have a look into your dataframe.",
                            href="basics",
                        ),
                        utils.create_tile(
                            icon="akar-icons:calendar",
                            heading="Target",
                            description="Get some insights into your target values.",
                            href="target",
                        ),
                        utils.create_tile(
                            icon="akar-icons:calendar",
                            heading="Features",
                            description="Analyse your features.",
                            href="features",
                        ),
                        utils.create_tile(
                            icon="akar-icons:calendar",
                            heading="Sample",
                            description="Ivestigat your sample.",
                            href="sample",
                        ),
                    ],
                )
            ],
        ),
        dmc.Space(h=20),
        html.Div(id="home_results"),
    ]
)


@callback(Output("home_results", "children"), Input("store_show_results", "data"))
def show_result_page(show_results):
    """Show result links."""
    if show_results is False:
        return []
    return [
        dmc.Container(
            size="lg",
            mt=30,
            children=[
                dmc.Title("Results"),
            ],
        ),
        dmc.Container(
            size="lg",
            px=0,
            py=0,
            my=40,
            children=[
                dmc.SimpleGrid(
                    mt=80,
                    cols={"xs": 1, "sm": 2, "xl": 3},
                    children=[
                        utils.create_tile(
                            icon="akar-icons:calendar",
                            heading="Summary",
                            description="Have a look at the overall performance.",
                            href="summary",
                        ),
                        utils.create_tile(
                            icon="akar-icons:calendar",
                            heading="Scores",
                            description="Scores",
                            href="scores",
                        ),
                        utils.create_tile(
                            icon="akar-icons:calendar",
                            heading="Optuna",
                            description="See the optuna insights.",
                            href="optuna",
                        ),
                        utils.create_tile(
                            icon="akar-icons:calendar",
                            heading="Individual splits",
                            description="Have a look at every individual data split.",
                            href="individual-splits",
                        ),
                        utils.create_tile(
                            icon="akar-icons:calendar",
                            heading="Configuration",
                            description="Selected configurations for your study.",
                            href="configs",
                        ),
                    ],
                )
            ],
        ),
    ]
