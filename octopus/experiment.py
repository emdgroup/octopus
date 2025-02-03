"""Octo Experiment module."""

import gzip
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional

import networkx as nx
import numpy as np
import pandas as pd
import scipy.stats
from attrs import Factory, define, field, validators

from octopus.config.core import OctoConfig
from octopus.logger import configure_logging

configure_logging()


@define
class OctoExperiment:
    """Experiment."""

    id: str = field(validator=[validators.instance_of(str)])
    """ID"""

    experiment_id: int = field(validator=[validators.instance_of(int)])
    """Identifier for the experiment."""

    sequence_item_id: int = field(validator=[validators.instance_of(int)])
    """Identifier for the sequence item."""

    input_item_id: int = field(validator=[validators.instance_of(int)])
    """Identifier for the input sequence item."""

    path_sequence_item: Path = field(validator=[validators.instance_of(Path)])
    """File system path to the sequence item."""

    configs: OctoConfig = field(validator=[validators.instance_of(OctoConfig)])
    """Configuration settings for the experiment."""

    datasplit_column: str = field(validator=[validators.instance_of(str)])
    """Column name used for data splitting."""

    row_column: str = field(validator=[validators.instance_of(str)])
    """Column name used as row identifier."""

    feature_columns: List = field(validator=[validators.instance_of(list)])
    """List of column names used as features in the experiment."""

    target_assignments: Dict = field(validator=[validators.instance_of(dict)])
    """Mapping of target variables to their assignments."""

    data_traindev: pd.DataFrame = field(
        validator=[validators.instance_of(pd.DataFrame)]
    )
    """DataFrame containing training and development data."""

    data_test: pd.DataFrame = field(validator=[validators.instance_of(pd.DataFrame)])
    """DataFrame containing test data."""

    stratification_column: Optional[str] = field(
        default=Factory(lambda: None),
        validator=validators.optional(validators.instance_of(str)),
    )
    "Column name used for stratification, if applicable."

    ml_module: str = field(
        init=False, default="", validator=[validators.instance_of(str)]
    )
    """Name of the machine learning module used."""

    # number of cpus available for each experiment
    num_assigned_cpus: int = field(
        init=False, default=0, validator=[validators.instance_of(int)]
    )
    """Number of CPUs assigned to the experiment."""

    ml_config: Dict = field(init=False, default=None)
    """Configuration settings for the machine learning module."""

    selected_features: List = field(
        default=Factory(list), validator=[validators.instance_of(list)]
    )
    """List of features selected for the experiment."""

    feature_groups: Dict = field(
        default=Factory(dict), validator=[validators.instance_of(dict)]
    )
    """Groupings of features based on correlation analysis."""

    prior_feature_importances: Dict = field(
        default=Factory(dict), validator=[validators.instance_of(dict)]
    )
    """Prior knowledge of feature importances, if available."""

    results: Dict = field(
        default=Factory(dict), validator=[validators.instance_of(dict)]
    )
    """Results of the experiment, keyed by result type."""

    @property
    def path_study(self) -> Path:
        """Get study path."""
        return Path(self.configs.study.path, self.configs.study.name)

    @property
    def ml_type(self) -> str:
        """Get ml_type from config."""
        return self.configs.study.ml_type

    def __attrs_post_init__(self):
        self.feature_groups = self.calculate_feature_groups(self.feature_columns)

    def calculate_feature_groups(self, feature_columns) -> dict:
        """Calculate feature groups based on correlation thresholds."""
        if len(feature_columns) <= 2:
            logging.warning(
                "Not enough features to calculate correlations for feature groups."
            )
            return {}
        logging.info("Calculating feature groups.")
        auto_group_thresholds = [0.7, 0.8, 0.9]
        auto_groups = list()

        pos_corr_matrix, _ = scipy.stats.spearmanr(
            np.nan_to_num(self.data_traindev[feature_columns].values)
        )
        pos_corr_matrix = np.abs(pos_corr_matrix)

        for threshold in auto_group_thresholds:
            g = nx.Graph()
            for i in range(len(feature_columns)):
                for j in range(i + 1, len(feature_columns)):
                    if pos_corr_matrix[i, j] > threshold:
                        g.add_edge(i, j)

            subgraphs = [
                g.subgraph(c).copy()
                for c in sorted(
                    nx.connected_components(g), key=lambda x: (len(x), sorted(x))
                )
            ]
            auto_groups.extend(
                [
                    [feature_columns[node] for node in sorted(sg.nodes())]
                    for sg in subgraphs
                ]
            )

        auto_groups_unique = [list(t) for t in set(map(tuple, auto_groups))]
        return {f"group{i}": group for i, group in enumerate(auto_groups_unique)}

    def to_pickle(self, file_path: str) -> None:
        """Save object to a compressed pickle file."""
        with gzip.GzipFile(file_path, "wb") as file:
            pickle.dump(self, file)

    @classmethod
    def from_pickle(cls, file_path: str) -> "OctoExperiment":
        """Load object from a compressed pickle file."""
        with gzip.GzipFile(file_path, "rb") as file:
            return pickle.load(file)
