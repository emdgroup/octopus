"""Octo Experiment module."""

import gzip
import pickle
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from attrs import define, field, validators

from octopus.modules.utils import rdc_correlation_matrix


@define
class OctoExperiment:
    """Experiment."""

    id: str = field(validator=[validators.instance_of(str)])
    experiment_id: int = field(validator=[validators.instance_of(int)])
    sequence_item_id: int = field(validator=[validators.instance_of(int)])
    path_sequence_item: Path = field(validator=[validators.instance_of(Path)])
    config: dict = field(validator=[validators.instance_of(dict)])
    datasplit_column: str = field(validator=[validators.instance_of(str)])
    row_column: str = field(validator=[validators.instance_of(str)])
    feature_columns: list = field(validator=[validators.instance_of(list)])
    stratification_column: str = field(validator=[validators.instance_of(str)])
    target_assignments: dict = field(validator=[validators.instance_of(dict)])
    data_traindev: pd.DataFrame = field(
        validator=[validators.instance_of(pd.DataFrame)]
    )
    data_test: pd.DataFrame = field(validator=[validators.instance_of(pd.DataFrame)])

    ml_module: str = field(
        init=False, default="", validator=[validators.instance_of(str)]
    )
    # number of cpus available for each experiment
    num_assigned_cpus: int = field(
        init=False, default=0, validator=[validators.instance_of(int)]
    )

    ml_config: dict = field(
        init=False, default=dict(), validator=[validators.instance_of(dict)]
    )
    # experiment outputs, initialized in post_init
    selected_features: list = field(
        init=False, validator=[validators.instance_of(list)]
    )
    results: dict = field(init=False, validator=[validators.instance_of(dict)])
    models: dict = field(init=False, validator=[validators.instance_of(dict)])
    feature_groups: dict = field(init=False, validator=[validators.instance_of(dict)])

    feature_importances: dict = field(
        init=False, validator=[validators.instance_of(dict)]
    )

    prior_feature_importances: dict = field(
        init=False, validator=[validators.instance_of(dict)]
    )

    scores: dict = field(init=False, validator=[validators.instance_of(dict)])

    predictions: dict = field(
        init=False,
        validator=[validators.instance_of(dict)],
    )

    @property
    def path_study(self) -> Path:
        """Path study."""
        return Path(self.config["output_path"], self.config["study_name"])

    @property
    def ml_type(self) -> str:
        """Shortcut to ml_type."""
        return self.config["ml_type"]

    def __attrs_post_init__(self):
        # initialization here due to "Python immutable default"
        self.selected_features = list()
        self.feature_importances = dict()
        self.prior_feature_importances = dict()
        self.scores = dict()
        self.predictions = dict()
        self.models = dict()
        self.results = dict()
        self._calculate_feature_groups()

    def _calculate_feature_groups(self) -> None:
        """Calculate Feature Groups."""
        auto_group_threshold = 0.6
        print("Calculating feature groups.")
        pos_corr_matrix = np.abs(
            rdc_correlation_matrix(self.data_traindev[self.feature_columns])
        )

        g = nx.Graph()

        for i in range(len(self.feature_columns)):
            for j in range(i + 1, len(self.feature_columns)):
                if pos_corr_matrix[i, j] > auto_group_threshold:
                    g.add_edge(i, j)

        subgraphs = [g.subgraph(c) for c in nx.connected_components(g)]

        groups = []
        for sg in subgraphs:
            groups.append([self.feature_columns[node] for node in sg.nodes()])

        auto_groups = [sorted(g) for g in groups]

        groups_dict = dict()
        for i, group in enumerate(auto_groups):
            groups_dict[f"group{i}"] = group

        self.feature_groups = groups_dict

    def to_pickle(self, file_path: str) -> None:
        """Save object to a compressed pickle file.

        Args:
            file_path: The name of the file to save the pickle data to.
        """
        with gzip.GzipFile(file_path, "wb") as file:
            pickle.dump(self, file)

    @classmethod
    def from_pickle(cls, file_path: str) -> "OctoExperiment":
        """Load object to a compressed pickle file.

        Args:
            file_path: The path to the file to load the pickle data from.

        Returns:
            OctoExperiment: The loaded instance of OctoExperiment.
        """
        with gzip.GzipFile(file_path, "rb") as file:
            return pickle.load(file)
