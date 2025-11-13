"""Octo Experiment module."""

import gzip
import logging
import pickle
from pathlib import Path
from typing import TYPE_CHECKING

import networkx as nx
import numpy as np
import pandas as pd
import scipy.stats
from attrs import Factory, define, field, validators

from octopus.config.base_sequence_item import BaseSequenceItem
from octopus.config.core import OctoConfig

if TYPE_CHECKING:
    from octopus.results import ModuleResults


@define
class OctoExperiment[ConfigType: BaseSequenceItem]:
    """Represents an Octopus experiment for ML pipeline execution.

    An OctoExperiment exists in two distinct states representing different stages of the
    ML workflow. The lifecycle begins with base experiments created during cross-validation
    data splitting. These base experiments serve as templates containing only the train/test
    data splits. When the pipeline executes, the manager deep copies base experiments and
    transforms them into sequence experiments by attaching ML module configurations (e.g.,
    feature selection, model training). This two-stage design separates data preparation
    from pipeline execution, allowing the same data splits to be reused across different
    pipeline configurations.
    """

    id: str = field(validator=[validators.instance_of(str)])
    """ID"""

    experiment_id: int = field(validator=[validators.instance_of(int), validators.ge(0)])
    """Identifier for the experiment."""

    sequence_id: int | None = field(
        validator=validators.optional(validators.and_(validators.instance_of(int), validators.ge(0)))
    )
    """Identifier for the sequence item."""

    input_sequence_id: int | None = field(
        validator=validators.optional(validators.and_(validators.instance_of(int), validators.ge(-1)))
    )
    """Identifier for the input sequence item."""

    _sequence_item_path: Path | None = field(validator=validators.optional(validators.instance_of(Path)))
    """Internal path storage. Use sequence_item_path property to access safely."""

    configs: OctoConfig = field(validator=[validators.instance_of(OctoConfig)])
    """Configuration settings for the experiment."""

    datasplit_column: str = field(validator=[validators.instance_of(str)])
    """Column name used for data splitting."""

    row_column: str = field(validator=[validators.instance_of(str)])
    """Column name used as row identifier."""

    feature_columns: list[str] = field(validator=[validators.instance_of(list)])
    """List of column names used as features in the experiment."""

    target_assignments: dict[str, str] = field(validator=[validators.instance_of(dict)])
    """Mapping of target variables to their assignments."""

    data_traindev: pd.DataFrame = field(validator=[validators.instance_of(pd.DataFrame)])
    """DataFrame containing training and development data."""

    data_test: pd.DataFrame = field(validator=[validators.instance_of(pd.DataFrame)])
    """DataFrame containing test data."""

    stratification_column: str | None = field(default=None, validator=validators.optional(validators.instance_of(str)))
    """Column name used for stratification, if applicable."""

    ml_module: str = field(init=False, default="", validator=[validators.instance_of(str)])
    """Name of the machine learning module used."""

    num_assigned_cpus: int = field(init=False, default=0, validator=[validators.instance_of(int)])
    """Number of CPUs assigned to the experiment."""

    ml_config: ConfigType = field(init=False, default=None)
    """Configuration settings for the module used by the sequence item."""

    selected_features: list = field(default=Factory(list), validator=[validators.instance_of(list)])
    """List of features selected for the experiment."""

    feature_groups: dict = field(default=Factory(dict), validator=[validators.instance_of(dict)])
    """Groupings of features based on correlation analysis."""

    results: dict[str, "ModuleResults"] = field(default=Factory(dict), validator=[validators.instance_of(dict)])
    """Results of the experiment, keyed by result type."""

    prior_results: dict[str, "ModuleResults"] = field(default=Factory(dict), validator=[validators.instance_of(dict)])
    """Results of the experiment used as input, keyed by result type."""

    @property
    def path_study(self) -> Path:
        """Get study path."""
        return Path(self.configs.study.path, self.configs.study.name)

    @property
    def is_base_experiment(self) -> bool:
        """Check if this is a base experiment (no sequence_id)."""
        return self.sequence_id is None

    @property
    def is_sequence_experiment(self) -> bool:
        """Check if this is a sequence experiment (has sequence_id)."""
        return self.sequence_id is not None

    @property
    def sequence_item_path(self) -> Path:
        """Get the sequence item path.

        Use this in modules that require a fully initialized experiment
        (not a base experiment).

        Returns:
            Path: The sequence item path

        Raises:
            ValueError: If this is a base experiment.
            RuntimeError: If validation failed and _sequence_item_path is None for a sequence experiment.
        """
        if self.is_base_experiment:
            raise ValueError(
                "Cannot access sequence_item_path on a base experiment. "
                "This operation requires a sequence experiment with sequence_item_path set."
            )
        if self._sequence_item_path is None:
            raise RuntimeError(
                "Validation failed: sequence experiment has no sequence_item_path set. "
                f"This should not happen (sequence_id={self.sequence_id})"
            )
        return self._sequence_item_path

    @property
    def ml_type(self) -> str:
        """Get ml_type from config."""
        return self.configs.study.ml_type

    def __attrs_post_init__(self):
        self._validate_experiment_state()
        self.feature_groups = self.calculate_feature_groups(self.feature_columns)

    def _validate_experiment_state(self) -> None:
        """Validate consistency between base and sequence experiment fields.

        Ensures that sequence-related fields (sequence_id, input_sequence_id,
        _sequence_item_path) are consistent with the experiment type (base vs sequence).

        Raises:
            ValueError: If fields are inconsistent with the experiment type.
        """
        if self.sequence_id is None:
            if self._sequence_item_path is not None:
                raise ValueError(
                    "Base experiments (sequence_id=None) cannot have _sequence_item_path set. "
                    f"Got _sequence_item_path={self._sequence_item_path}"
                )
            if self.input_sequence_id is not None:
                raise ValueError(
                    "Base experiments (sequence_id=None) cannot have input_sequence_id set. "
                    f"Got input_sequence_id={self.input_sequence_id}"
                )
        else:
            if self._sequence_item_path is None:
                raise ValueError(
                    f"Sequence experiments (sequence_id={self.sequence_id}) must have _sequence_item_path set"
                )
            if self.input_sequence_id is None:
                raise ValueError(
                    f"Sequence experiments (sequence_id={self.sequence_id}) must have input_sequence_id set"
                )

    def calculate_feature_groups(self, feature_columns: list[str]) -> dict[str, list[str]]:
        """Calculate feature groups based on correlation thresholds."""
        if len(feature_columns) <= 2:
            logging.warning("Not enough features to calculate correlations for feature groups.")
            return {}
        logging.info("Calculating feature groups.")

        auto_group_thresholds = [0.7, 0.8, 0.9]
        auto_groups = []

        pos_corr_matrix, _ = scipy.stats.spearmanr(np.nan_to_num(self.data_traindev[feature_columns].values))
        pos_corr_matrix = np.abs(pos_corr_matrix)

        # get groups depending on threshold
        for threshold in auto_group_thresholds:
            g: nx.Graph = nx.Graph()
            for i in range(len(feature_columns)):
                for j in range(i + 1, len(feature_columns)):
                    if pos_corr_matrix[i, j] > threshold:
                        g.add_edge(i, j)

            # Get connected components and sort them to ensure determinism
            subgraphs = [
                g.subgraph(c).copy() for c in sorted(nx.connected_components(g), key=lambda x: (len(x), sorted(x)))
            ]
            # Create groups of feature columns
            groups = []
            for sg in subgraphs:
                groups.append([feature_columns[node] for node in sorted(sg.nodes())])
            auto_groups.extend([sorted(g) for g in groups])

        # find unique groups
        auto_groups_unique = [list(t) for t in set(map(tuple, auto_groups))]

        return {f"group{i}": group for i, group in enumerate(auto_groups_unique)}

    def to_pickle(self, file_path: str | Path):
        """Save object to a compressed pickle file."""
        with gzip.GzipFile(file_path, "wb") as file:
            pickle.dump(self, file)

    @classmethod
    def from_pickle(cls, file_path: str | Path) -> "OctoExperiment":
        """Load object from a compressed pickle file."""
        with gzip.GzipFile(file_path, "rb") as file:
            data = pickle.load(file)

        if not isinstance(data, cls):
            raise TypeError(f"Loaded object is not of type {cls.__name__}")

        return data
