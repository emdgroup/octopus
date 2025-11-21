"""Octo Study."""

import json
from pathlib import Path

import pandas as pd
from attrs import Factory, asdict, define, field, fields, has, validators

from octopus.experiment import OctoExperiment
from octopus.logger import get_logger
from octopus.manager.core import OctoManager
from octopus.modules import Octo
from octopus.task import Task
from octopus.utils import DataSplit

from .data_preparator import OctoDataPreparator
from .data_validator import OctoDataValidator
from .healthChecker import HealthCheckConfig, OctoDataHealthChecker
from .types import DatasplitType, ImputationMethod, MLType
from .validation import validate_metric, validate_metrics_list, validate_tasks

logger = get_logger()


@define
class OctoStudy:
    """OctoStudy."""

    ml_type: MLType = field(
        converter=lambda x: MLType(x.lower()) if isinstance(x, str) else x,
        validator=validators.instance_of(MLType),
    )
    """The type of machine learning model."""

    target_metric: str = field(validator=[validate_metric])
    """The primary metric used for model evaluation."""

    feature_columns: list[str] = field(validator=[validators.instance_of(list)])
    """List of all feature columns in the dataset."""

    target_columns: list[str] = field(validator=[validators.instance_of(list)])
    """List of target columns in the dataset. For regression and classification,
    only one target is allowed. For time-to-event, two targets need to be provided.
    """

    sample_id: str = field(validator=validators.instance_of(str))
    """Identifier for sample instances."""

    name: str = field(validator=[validators.instance_of(str)])
    """The name of the study."""

    datasplit_type: DatasplitType = field(
        default=DatasplitType.SAMPLE,
        converter=lambda x: DatasplitType(x.lower()) if isinstance(x, str) else x,
        validator=validators.instance_of(DatasplitType),
    )
    """Type of datasplit. Allowed are `sample`, `group_features` and `group_sample_and_features`."""

    row_id: str | None = field(
        default=Factory(lambda: None),
        validator=validators.optional(validators.instance_of(str)),
    )
    """Unique row identifier."""

    target_assignments: dict[str, str] = field(default=Factory(dict), validator=[validators.instance_of(dict)])
    """Mapping of target assignments."""

    stratification_column: str | None = field(
        default=Factory(lambda: None),
        validator=validators.optional(validators.instance_of(str)),
    )
    """List of columns used for stratification."""

    positive_class: int = field(default=1, validator=validators.instance_of(int))
    """The positive class label for binary classification. Defaults to 1. Not relevant for other ml_types."""

    n_folds_outer: int = field(default=5, validator=[validators.instance_of(int)])
    """The number of outer folds for cross-validation. Defaults to 5."""

    datasplit_seed_outer: int = field(default=0, validator=[validators.instance_of(int)])
    """The seed used for data splitting in outer cross-validation. Defaults to 0."""

    imputation_method: ImputationMethod = field(
        default=ImputationMethod.MEDIAN,
        converter=lambda x: ImputationMethod(x.lower()) if isinstance(x, str) else x,
        validator=validators.instance_of(ImputationMethod),
    )

    metrics: list = field(
        default=Factory(lambda self: [self.target_metric], takes_self=True),
        validator=[validators.instance_of(list), validate_metrics_list],
    )
    """A list of metrics to be calculated. Defaults to target_metric value."""

    ignore_data_health_warning: bool = field(default=Factory(lambda: False), validator=[validators.instance_of(bool)])
    """Ignore data health checks warning and run machine learning workflow."""

    outer_parallelization: bool = field(default=Factory(lambda: True), validator=[validators.instance_of(bool)])
    """Indicates whether outer parallelization is enabled. Defaults to True."""

    run_single_experiment_num: int = field(default=Factory(lambda: -1), validator=[validators.instance_of(int)])
    """Select a single experiment to execute. Defaults to -1 to run all experiments"""

    tasks: list[Task] = field(
        default=Factory(lambda: [Octo(task_id=0)]),
        validator=[validators.instance_of(list), validate_tasks],
    )
    """A list of workflow tasks that defines the processing workflow. Each item in the list is an instance of `Task`."""

    path: str = field(default="./studies/")
    """The path where study outputs are saved. Defaults to "./studies/"."""

    relevant_columns: list[str] = field(init=False)
    """Relevant columns for the dataset. Set during fit()."""

    @property
    def output_path(self) -> Path:
        """Full output path for this study (path/name)."""
        return Path(self.path) / self.name

    def _validate_data(self, data: pd.DataFrame) -> None:
        """Validate the input data."""
        validator = OctoDataValidator(
            data=data,
            feature_columns=self.feature_columns,
            target_columns=self.target_columns,
            sample_id=self.sample_id,
            row_id=self.row_id,
            stratification_column=self.stratification_column,
            target_assignments=self.target_assignments,
            ml_type=self.ml_type.value,
            positive_class=self.positive_class,
        )
        validator.validate()

    def _initialize_study_outputs(self, data: pd.DataFrame) -> None:
        """Initialize study by setting up directory and saving config and data."""
        # Create output directory
        self.output_path.mkdir(parents=True, exist_ok=True)

        # Save configuration to JSON
        def serialize_value(value):
            """Convert a value to JSON-serializable format."""
            if hasattr(value, "value"):
                return value.value
            elif isinstance(value, Path):
                return str(value)
            elif has(type(value)):
                return asdict(value, value_serializer=lambda _, __, v: serialize_value(v))
            elif isinstance(value, list):
                return [serialize_value(item) for item in value]
            elif isinstance(value, dict):
                return {k: serialize_value(v) for k, v in value.items()}
            return value

        config = {}
        for attr in fields(OctoStudy):
            if attr.name == "relevant_columns":
                continue
            value = getattr(self, attr.name)
            config[attr.name] = serialize_value(value)

        config_path = self.output_path / "config.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

        data.to_parquet(self.output_path / "data.parquet", index=False)

    def _prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare the data for training."""
        preparator = OctoDataPreparator(
            data=data,
            feature_columns=self.feature_columns,
            target_columns=self.target_columns,
            sample_id=self.sample_id,
            row_id=self.row_id,
            target_assignments=self.target_assignments,
        )
        prepared_data, feature_columns, row_id, target_assignments = preparator.prepare()

        object.__setattr__(self, "feature_columns", feature_columns)
        object.__setattr__(self, "row_id", row_id)
        object.__setattr__(self, "target_assignments", target_assignments)

        return prepared_data

    def _compute_relevant_columns(self, data: pd.DataFrame) -> None:
        """Compute and store relevant columns for the dataset."""
        relevant_columns = list(set(self.feature_columns + self.target_columns))
        if self.sample_id:
            relevant_columns.append(self.sample_id)
        if self.row_id:
            relevant_columns.append(self.row_id)
        if self.stratification_column:
            relevant_columns.append(self.stratification_column)
        if "group_features" in data.columns:
            relevant_columns.append("group_features")
        if "group_sample_and_features" in data.columns:
            relevant_columns.append("group_sample_and_features")
        object.__setattr__(self, "relevant_columns", list(set(relevant_columns)))

    def _create_datasplits(self, data: pd.DataFrame) -> dict:
        """Create datasplits for outer cross-validation."""
        data_clean = data[self.relevant_columns]

        if self.datasplit_type.value == "sample":
            datasplit_col = self.sample_id
        else:
            datasplit_col = self.datasplit_type.value

        datasplits: dict = DataSplit(
            dataset=data_clean,
            datasplit_col=datasplit_col,
            seeds=[self.datasplit_seed_outer],
            num_folds=self.n_folds_outer,
            stratification_col=self.stratification_column,
        ).get_datasplits()

        return datasplits

    def _create_experiments(self, datasplits: dict) -> list[OctoExperiment]:
        """Create experiments from datasplits."""
        experiments = []

        # Get datasplit column based on datasplit_type
        if self.datasplit_type.value == "sample":
            datasplit_col = self.sample_id
        else:
            datasplit_col = self.datasplit_type.value

        for key, value in datasplits.items():
            experiment_path = self.output_path / f"experiment{key}"
            experiment_path.mkdir(parents=True, exist_ok=True)

            experiment: OctoExperiment = OctoExperiment(
                id=str(key),
                experiment_id=int(key),
                task_id=None,  # indicating base experiment
                depends_on_task=None,  # indicating base experiment
                task_path=None,  # indicating base experiment
                study_path=self.path,
                study_name=self.name,
                ml_type=self.ml_type.value,
                target_metric=self.target_metric,
                positive_class=self.positive_class,
                metrics=self.metrics,
                imputation_method=self.imputation_method.value,
                datasplit_column=datasplit_col,
                row_column=self.row_id,  # type: ignore[arg-type]  # row_id is always set after _prepare_data
                feature_columns=self.feature_columns,
                target_assignments=self.target_assignments,
                data_traindev=value["train"],
                data_test=value["test"],
            )
            experiments.append(experiment)

        return experiments

    def _run_health_check(self, data: pd.DataFrame, config: HealthCheckConfig | None) -> None:
        """Run data health check, save results, and check for issues."""
        checker = OctoDataHealthChecker(
            data=data,
            feature_columns=self.feature_columns,
            target_columns=self.target_columns,
            row_id=self.row_id,
            sample_id=self.sample_id,
            stratification_column=self.stratification_column,
            config=config or HealthCheckConfig(),
        )
        report = checker.generate_report()
        report_path = self.output_path / "health_check_report.csv"
        report.to_csv(report_path, index=False)

        if report.empty:
            return

        has_critical = False
        has_warning = False

        if "severity" in report.columns:
            has_critical = (report["severity"] == "critical").any()
            has_warning = (report["severity"] == "warning").any()

        if has_critical:
            raise ValueError(f"Critical data issues detected. Please check: {report_path}")

        if has_warning and not self.ignore_data_health_warning:
            raise ValueError(
                f"Data issues detected. Please check: {report_path}\n"
                f"To proceed despite warnings, set `ignore_data_health_warning=True`."
            )

    def fit(
        self,
        data: pd.DataFrame,
        health_check_config: HealthCheckConfig | None = None,
    ) -> None:
        """Fit study to data.

        Args:
            data: DataFrame containing the dataset.
            health_check_config: Optional configuration for health check thresholds.
        """
        # TODO: relevant columns can be updated during preparation, check if it makes sense to validate after preparation
        self._validate_data(data)
        prepared_data = self._prepare_data(data)
        self._compute_relevant_columns(prepared_data)
        self._initialize_study_outputs(prepared_data)
        self._run_health_check(prepared_data, health_check_config)

        datasplits = self._create_datasplits(prepared_data)
        experiments = self._create_experiments(datasplits)
        manager = OctoManager(
            base_experiments=experiments,
            tasks=self.tasks,
            n_folds_outer=self.n_folds_outer,
            outer_parallelization=self.outer_parallelization,
            run_single_experiment_num=self.run_single_experiment_num,
        )
        manager.run_outer_experiments()
