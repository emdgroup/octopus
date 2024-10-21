"""Data health check example."""

from decimal import Decimal

import numpy as np
import pandas as pd
from attrs import define, field
from sklearn.datasets import make_classification

from octopus import OctoData, OctoML
from octopus.config import ConfigManager, ConfigSequence, ConfigStudy
from octopus.modules import Octo


@define
class DataFrameGenerator:
    """A class to generate an example DataFrame."""

    n_samples: int = 1000
    n_features: int = 20
    n_informative: int = 10
    n_redundant: int = 10
    n_classes: int = 3
    random_state: int = None

    df: pd.DataFrame = field(init=False)

    def __attrs_post_init__(self):
        self._generate_data()

    def _generate_data(self):
        """Generate the classification dataset and initialize the DataFrame."""
        X, y = make_classification(
            n_samples=self.n_samples,
            n_features=self.n_features,
            n_informative=self.n_informative,
            n_redundant=self.n_redundant,
            n_classes=self.n_classes,
            n_clusters_per_class=2,
            class_sep=5.0,
            random_state=self.random_state,
        )

        # Create DataFrame from features
        feature_names = [f"feature_{i+1}" for i in range(self.n_features)]
        self.df = pd.DataFrame(X, columns=feature_names)

        # Add the target column
        self.df["target"] = y

    def add_nan_to_features(self, min_frac=0.02, max_frac=0.8):
        """Add a random proportion of NaNs to the first half of the feature columns."""
        half_features = self.df.columns[: self.n_features // 2]
        rng = np.random.default_rng(self.random_state)
        num_rows = len(self.df)

        for feature in half_features:
            # Determine the number of NaNs to introduce based on the random fraction
            nan_fraction = rng.uniform(min_frac, max_frac)
            num_nan = int(nan_fraction * num_rows)

            # Select random indices to set as NaN
            nan_indices = rng.choice(self.df.index, size=num_nan, replace=False)
            self.df.loc[nan_indices, feature] = np.nan

    def add_nan_to_target(self, num_nan=10):
        """Add NaN values to the target column."""
        rng = np.random.default_rng(self.random_state)
        nan_indices = rng.choice(self.df.index, size=num_nan, replace=False)
        self.df.loc[nan_indices, "target"] = np.nan

    def add_id_column(
        self,
        column_name="id",
        prefix="ID_",
        unique=True,
        duplicate_factor=2,
        include_nans=False,
        nan_ratio=0.1,
    ):
        """Add an ID column with unique or non-unique identifiers."""
        if prefix is None:
            # Use integers for IDs
            ids = np.arange(len(self.df), dtype="uint" if unique else "int")
            if not unique:
                ids = np.repeat(ids, duplicate_factor)[: len(self.df)]
        elif unique:
            # Create unique IDs with prefix
            ids = [prefix + str(i) for i in self.df.index]
        else:
            # Create non-unique IDs with prefix
            ids = [prefix + str(i) for i in range(len(self.df) // duplicate_factor)]
            non_unique_ids = ids * duplicate_factor
            ids = non_unique_ids[: len(self.df)]

        if include_nans:
            # Determine number of NaNs to include
            num_nans = int(len(self.df) * nan_ratio)
            nan_indices = np.random.choice(len(self.df), num_nans, replace=False)
            ids = np.array(ids, dtype=object)  # Convert to a mutable array
            ids[nan_indices] = np.nan

        self.df[column_name] = ids

    def add_constant_column(self, column_name="one", value=1):
        """Add a constant column to the DataFrame."""
        self.df[column_name] = value

    def add_decimal_columns(self, column_names=["decimal_1", "decimal_2"], precision=8):
        """Add columns with Decimal data type."""
        rng = np.random.default_rng(self.random_state)
        for col_name in column_names:
            random_numbers = rng.random(size=len(self.df))
            formatted_numbers = [
                Decimal(f"{num:.{precision}f}") for num in random_numbers
            ]
            self.df[col_name] = formatted_numbers

    def add_inf_columns(self, column_names=["inf_col"], num_inf=10):
        """Add columns with infinite values."""
        rng = np.random.default_rng(self.random_state)
        for col_name in column_names:
            # Initialize the column with random float values
            self.df[col_name] = rng.standard_normal(size=len(self.df))
            # Introduce inf values
            inf_indices = rng.choice(self.df.index, size=num_inf, replace=False)
            self.df.loc[inf_indices, col_name] = np.inf

    def get_dataframe(self):
        """Return the generated DataFrame.

        Returns:
        - pd.DataFrame: The generated DataFrame.
        """
        return self.df.copy()


# Example usage
generator = DataFrameGenerator(random_state=42)
generator.add_nan_to_features()
generator.add_nan_to_target(num_nan=10)
generator.add_id_column(unique=False, include_nans=True)
generator.add_id_column(
    column_name="sample_id", prefix="Sample", unique=True, include_nans=True
)
generator.add_id_column(
    column_name="stratification",
    prefix="Strat_",
    unique=True,
    include_nans=True,
)
generator.add_constant_column()
generator.add_decimal_columns()
generator.add_inf_columns()

df = generator.get_dataframe()

octo_data = OctoData(
    data=df,
    target_columns=["target"],
    feature_columns=df.columns.drop("target")
    .drop("id")
    .drop("sample_id")
    .drop("stratification")
    .tolist(),
    row_id="id",
    sample_id="sample_id",
    datasplit_type="group_sample_and_features",
    stratification_column=["stratification"],
)

config_study = ConfigStudy(
    name="basic_classification",
    ml_type="classification",
    target_metric="AUCROC",
    silently_overwrite_study=True,
)

config_manager = ConfigManager(outer_parallelization=True)

config_sequence = ConfigSequence(
    sequence_items=[
        Octo(description="step_1_octo", models=["RandomForestClassifier"], n_trials=3)
    ]
)


### Execute the Machine Learning Workflow

# We add the data and the configurations defined earlier
# and run the machine learning workflow.

octo_ml = OctoML(
    octo_data,
    config_study=config_study,
    config_manager=config_manager,
    config_sequence=config_sequence,
)
# octo_ml.create_outer_experiments()
# octo_ml.run_outer_experiments()

# print("Workflow completed")
