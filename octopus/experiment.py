"""Octo Experiment module."""

import gzip
import pickle
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import scipy.stats
from attrs import Factory, define, field, validators

from octopus.config.core import OctoConfig


@define
class OctoExperiment:
    """Experiment."""

    id: str = field(validator=[validators.instance_of(str)])
    experiment_id: int = field(validator=[validators.instance_of(int)])
    sequence_item_id: int = field(validator=[validators.instance_of(int)])
    path_sequence_item: Path = field(validator=[validators.instance_of(Path)])
    configs: OctoConfig = field(validator=[validators.instance_of(OctoConfig)])
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

    ml_config: dict = field(init=False, default=None)

    selected_features: list = field(
        default=Factory(list), validator=[validators.instance_of(list)]
    )

    feature_groups: dict = field(
        default=Factory(dict), validator=[validators.instance_of(dict)]
    )

    prior_feature_importances: dict = field(
        default=Factory(dict), validator=[validators.instance_of(dict)]
    )

    results: dict = field(
        default=Factory(dict), validator=[validators.instance_of(dict)]
    )

    @property
    def path_study(self) -> Path:
        """Path study."""
        return Path(self.configs.study.path, self.configs.study.name)

    @property
    def ml_type(self) -> str:
        """Shortcut to ml_type."""
        return self.configs.study.ml_type

    def __attrs_post_init__(self):
        # self.feature_groups = dict()
        self.calculate_feature_groups()

    def calculate_feature_groups(self) -> None:
        """Calculate Feature Groups."""
        # looking for groups arising from different thresholds
        auto_group_thresholds = [0.7, 0.8, 0.9]
        auto_groups = list()
        print("Calculating feature groups.")
        # correlation matrix
        # (A) spearmamr correlation matrix
        pos_corr_matrix, _ = scipy.stats.spearmanr(
            np.nan_to_num(self.data_traindev[self.feature_columns].values)
        )
        pos_corr_matrix = np.abs(pos_corr_matrix)

        # (B) RDC correlation matrix
        # pos_corr_matrix = np.abs(
        #    rdc_correlation_matrix(self.data_traindev[self.feature_columns])
        # )
        # get groups depending on threshold
        for threshold in auto_group_thresholds:
            g = nx.Graph()
            for i in range(len(self.feature_columns)):
                for j in range(i + 1, len(self.feature_columns)):
                    if pos_corr_matrix[i, j] > threshold:
                        g.add_edge(i, j)
            # Get connected components and sort them to ensure determinism
            subgraphs = [
                g.subgraph(c).copy()
                for c in sorted(
                    nx.connected_components(g), key=lambda x: (len(x), sorted(x))
                )
            ]
            # Create groups of feature columns
            groups = []
            for sg in subgraphs:
                groups.append(
                    [self.feature_columns[node] for node in sorted(sg.nodes())]
                )
            auto_groups.extend([sorted(g) for g in groups])

        # find unique groups
        auto_groups_unique = [list(t) for t in set(map(tuple, auto_groups))]
        # create groups dicts
        groups_dict = dict()
        for i, group in enumerate(auto_groups_unique):
            groups_dict[f"group{i}"] = group

        print("Feature Groups:", groups_dict)
        self.feature_groups = groups_dict

    def extract_fi_from_results(self):
        """Extract features importances from results."""
        feature_importances = dict()
        for key, moduleresult in self.results.items():
            feature_importances[key] = moduleresult.feature_importances
        return feature_importances

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
