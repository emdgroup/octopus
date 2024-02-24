"""Octopus Analitics."""

import pickle

import dash_ag_grid as dag
import dash_mantine_components as dmc
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from attrs import define, field
from dash import Dash, Input, Output, State, callback, dcc

from octopus.modules import utils

# component = dmc.MantineProvider(withGlobalStyles=True, theme={"colorScheme": "dark"})

pio.templates["Octopus"] = pio.templates["plotly_dark"]
pio.templates.default = "Octopus"


@define
class OctoAnalitics:
    """Analitics."""

    study_path = field()
    """Path of Study."""

    experiments = field(default=[])
    dataset = field(default=pd.DataFrame())
    scores = field(default=pd.DataFrame())
    "List of path for each experiment file."

    def __attrs_post_init__(self):
        def _get_experiments(self):
            """Get experiments data."""
            for file in list(self.study_path.glob("**/exp*.pkl")):
                with open(file, "rb") as f:
                    exp = pickle.load(f)
                self.experiments.append(exp)

        def _get_dataset(self):
            """Get dataset."""
            self.dataset = pd.concat(
                [self.experiments[0].data_traindev, self.experiments[0].data_test]
            )

        def _get_scores(self):
            """Calculate scores from predictions."""
            dict_scores = []
            for exp in self.experiments:
                for idx, split in enumerate(exp.predictions):
                    for dataset in exp.predictions[split]:
                        if split != "test":
                            for mectric in ["MAE", "MSE", "R2"]:
                                dict_temp = {
                                    "experiment_id": exp.experiment_id,
                                    "sequence_id": exp.sequence_item_id,
                                    "split": idx,
                                    "testset": dataset,
                                    "metric": mectric,
                                    "value": utils.get_score(
                                        mectric,
                                        exp.predictions[split][dataset]["target"],
                                        exp.predictions[split][dataset]["prediction"],
                                    ),
                                }
                                dict_scores.append(dict_temp)
            self.scores = (
                pd.DataFrame(dict_scores)
                .sort_values(by=["experiment_id", "sequence_id", "split"])
                .reset_index(drop=True)
            )

        _get_experiments(self)
        _get_dataset(self)
        _get_scores(self)

    def run_analytics(self):
        """Run app."""
        app = Dash(__name__)
        print(dmc.theme.DEFAULT_COLORS["blue"][6])
        experiment_ids = sorted([exp.experiment_id for exp in self.experiments])

        def splits():
            return dmc.Paper(
                [
                    dmc.Title("Individual Splits"),
                    dmc.Grid(
                        [
                            dmc.Select(
                                label="Experiment",
                                id="select_exp",
                                value="0",
                                data=[
                                    {"value": str(i), "label": str(i)}
                                    for i in experiment_ids
                                ],
                            ),
                            dmc.Select(
                                label="Sequence",
                                id="select_sequence",
                                value="0",
                            ),
                            dmc.Select(
                                label="Split",
                                id="select_split",
                            ),
                        ],
                        style={"margin-bottom": "20px"},
                    ),
                    dmc.Tabs(
                        [
                            dmc.TabsList(
                                [
                                    dmc.Tab(
                                        "Ground Truth",
                                        value="ground_truth",
                                    ),
                                    dmc.Tab(
                                        "Feature Importances",
                                        value="feature_importances",
                                    ),
                                ]
                            ),
                            dmc.TabsPanel(
                                [
                                    dcc.Graph(id="graph_ground_truth"),
                                    dmc.Text(
                                        """Use plotly selection tool
                                          to select datapoints."""
                                    ),
                                    dag.AgGrid(
                                        id="aggrid_ground_truth",
                                        # columnDefs=
                                        defaultColDef={
                                            "resizable": True,
                                            "autoHeaderHeight": True,
                                            "wrapHeaderText": True,
                                            "suppressMovable": True,
                                        },
                                        columnSize="autoSize",
                                        className="ag-theme-alpine",
                                    ),
                                ],
                                value="ground_truth",
                            ),
                            dmc.TabsPanel(
                                [
                                    "Feature Importances",
                                    dcc.Graph(id="graph_feature_importances"),
                                ],
                                value="feature_importances",
                            ),
                        ],
                        value="ground_truth",
                    ),
                ]
            )

        def scores():
            return dmc.Paper(
                [
                    dmc.Title("Scores"),
                    dmc.Grid(
                        [
                            dmc.Col(
                                [
                                    dmc.Text("Metric"),
                                    dmc.SegmentedControl(
                                        id="segment_scores_metric",
                                        value="MAE",
                                        orientation="vertical",
                                        data=[
                                            {"value": i, "label": i}
                                            for i in self.scores["metric"].unique()
                                        ],
                                        mt=10,
                                    ),
                                    dmc.Space(h="lg"),
                                    dmc.Text("Aggregation"),
                                    dmc.SegmentedControl(
                                        id="segment_scores_aggregation",
                                        value="Average",
                                        orientation="vertical",
                                        data=[
                                            {"value": "All", "label": "All"},
                                            {"value": "Average", "label": "Average"},
                                        ],
                                        mt=10,
                                    ),
                                ],
                                span=2,
                            ),
                            dmc.Col(
                                dcc.Graph(id="graph_scores"),
                                span=10,
                            ),
                        ]
                    ),
                ]
            )

        def sequence():
            return dmc.Paper(
                [
                    dmc.Title("Sequence"),
                    dmc.AccordionMultiple(id="accordion_sequence"),
                ]
            )

        # Define the app layout
        app.layout = dmc.MantineProvider(
            theme={"colorScheme": "dark"},
            inherit=True,
            withGlobalStyles=True,
            withNormalizeCSS=True,
            children=[
                dmc.Paper(
                    [
                        # dmc.Header(
                        #     height=50,
                        #     fixed=True,
                        #     children=[
                        #         dmc.Text(self.experiments[0].config["study_name"])
                        #     ],
                        # ),
                        dmc.Container(
                            [
                                sequence(),
                                splits(),
                                scores(),
                            ],
                        ),
                    ],
                    p="md",
                )
            ],
        )

        @callback(Output("select_sequence", "data"), Input("select_exp", "value"))
        def get_sequence_ids(experiment_id):
            """Get sequence ids for selected experiment."""
            sequence_item_ids = [
                exp.sequence_item_id
                for exp in self.experiments
                if exp.experiment_id == int(experiment_id)
            ]
            return [{"value": str(i), "label": str(i)} for i in sequence_item_ids]

        @callback(
            Output("select_split", "value"),
            Output("select_split", "data"),
            Input("select_exp", "value"),
            Input("select_sequence", "value"),
        )
        def get_split_ids(
            experiment_id,
            sequence_id,
        ):
            """Get splits ids for selected experiment."""
            split_ids = [
                key
                for exp in self.experiments
                if exp.experiment_id == int(experiment_id)
                and exp.sequence_item_id == int(sequence_id)
                for key in exp.predictions.keys()
            ]
            return (
                split_ids[0],
                [{"value": str(i), "label": str(i)} for i in split_ids],
            )

        @callback(
            Output("graph_ground_truth", "figure"),
            Output("graph_feature_importances", "figure"),
            State("select_exp", "value"),
            State("select_sequence", "value"),
            Input("select_split", "value"),
        )
        def plot_ground_truth(experiment_id, sequence_id, split_id):
            """Get splits ids for selected experiment."""
            predictions = next(
                (
                    exp.predictions[split_id]
                    for exp in self.experiments
                    if exp.experiment_id == int(experiment_id)
                    and exp.sequence_item_id == int(sequence_id)
                ),
                None,
            )

            feature_importances = next(
                (
                    exp.feature_importances[split_id]
                    for exp in self.experiments
                    if exp.experiment_id == int(experiment_id)
                    and exp.sequence_item_id == int(sequence_id)
                ),
                None,
            )
            fig_1 = go.Figure()
            for i in ["train", "dev", "test"]:
                fig_1.add_trace(
                    go.Scatter(
                        x=predictions[i]["target"],
                        y=predictions[i]["prediction"],
                        mode="markers",
                        name=i,
                        text=predictions[i]["index"],
                    )
                )

            fig_1.add_shape(
                type="line",
                line=dict(dash="dash"),
                x0=min(predictions[i]["target"]),
                y0=min(predictions[i]["target"]),
                x1=max(predictions[i]["target"]),
                y1=max(predictions[i]["target"]),
            )

            fig_1.update_layout(
                xaxis_title="Ground truth",
                yaxis_title="Prediction",
            )

            fig_2 = go.Figure()
            for name, df_ in feature_importances.items():
                fig_2.add_trace(
                    go.Bar(name=name, x=df_["feature"], y=df_["importance"])
                )
            return fig_1, fig_2

        @callback(
            Output("aggrid_ground_truth", "rowData"),
            Output("aggrid_ground_truth", "columnDefs"),
            Input("graph_ground_truth", "selectedData"),
        )
        def show_selected_datapoint(selected_data):
            """Get splits ids for selected experiment."""
            columns = [{"field": i} for i in self.dataset.columns]

            if selected_data is None:
                data = pd.DataFrame()
            else:
                selected_points = [entry["text"] for entry in selected_data["points"]]
                data = self.dataset[self.dataset["index"].isin(selected_points)]

            return data.to_dict("records"), columns

        @callback(
            Output("graph_scores", "figure"),
            Input("select_exp", "value"),
            Input("segment_scores_aggregation", "value"),
            Input("segment_scores_metric", "value"),
        )
        def plot_scores(_, aggregation, metric):
            """Get splits ids for selected experiment."""
            fig = go.Figure()
            if aggregation == "All":
                for i in ["train", "dev", "test"]:
                    fig.add_trace(
                        go.Scatter(
                            x=self.scores.index.astype(str),
                            y=self.scores.query(
                                f'testset == "{i}" and metric == "{metric}"'
                            )["value"],
                            mode="markers+lines",
                            name=i,
                        )
                    )

            elif aggregation == "Average":
                for i in ["train", "dev", "test"]:
                    df_temp = self.scores.query(
                        f'testset == "{i}" and metric == "{metric}"'
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=self.scores.experiment_id.astype(str).unique(),
                            y=df_temp.groupby("experiment_id")["value"].mean(),
                            mode="markers+lines",
                            name=i,
                        )
                    )

            fig.update_layout(
                title="Plot Title",
                xaxis_title="Number",
                yaxis_title=metric,
            )
            return fig

        # Run the Dash app
        app.run_server(debug=True)
