"""Octo Training."""

import copy
import math

import numpy as np
import pandas as pd
import scipy
import shap
from attrs import Factory, define, field, validators
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from octopus.logger import LogGroup, get_logger
from octopus.models.inventory import ModelInventory
from octopus.modules.utils import get_performance_score

## TOBEDONE pipeline
# - establish pre-processing pipeline
# - get model info and connect with pipeline
# - enhance categorical processing: ohe + targetenconding?
# - modify data class to provide info on categorical (nominal, ordinal) columns
# - test script testing all models
#   + numeric, categoricals, different categorical dtypes
#   + end-to-end testing including permutation FI and shap
# - remove imputation code
# - keep mice imputation?
# - how to provide categorical info to catboost and other models?


logger = get_logger()

scorer_string_inventory = {
    "AUCROC": "roc_auc",
    "ACC": "accuracy",
    "ACCBAL": "balanced_accuracy",
    "LOGLOSS": "neg_log_loss",
    "MAE": "neg_mean_absolute_error",
    "MSE": "neg_mean_squared_error",
    "R2": "r2",
}


@define
class Training:
    """Model Training Class."""

    training_id: str = field(validator=[validators.instance_of(str)])
    """Training id."""

    ml_type: str = field(validator=[validators.instance_of(str)])
    """ML-type."""

    target_assignments: dict = field(validator=[validators.instance_of(dict)])
    """Target assignments."""

    feature_columns: list = field(validator=[validators.instance_of(list)])
    """Feature columns."""

    row_column: str = field(validator=[validators.instance_of(str)])
    """Row column name."""

    data_train: pd.DataFrame = field(validator=[validators.instance_of(pd.DataFrame)])
    "Data train."

    data_dev: pd.DataFrame = field(validator=[validators.instance_of(pd.DataFrame)])
    "Data dev."

    data_test: pd.DataFrame = field(validator=[validators.instance_of(pd.DataFrame)])
    """Data test."""

    target_metric: str = field(validator=[validators.instance_of(str)])
    """Target metric."""

    max_features: int = field(validator=[validators.instance_of(int)])
    """Maximum number of features."""

    feature_groups: dict = field(validator=[validators.instance_of(dict)])
    """Feature Groups."""

    config_training: dict = field(validator=[validators.instance_of(dict)])
    """Training configuration."""

    training_weight: int = field(default=1, validator=[validators.instance_of(int)])
    """Training weight for ensembling"""

    model = field(default=None)
    """Model."""

    predictions: dict = field(default=Factory(dict), validator=[validators.instance_of(dict)])
    """Model predictions."""

    feature_importances: dict = field(default=Factory(dict), validator=[validators.instance_of(dict)])
    """Feature importances."""

    features_used: list = field(default=Factory(list), validator=[validators.instance_of(list)])
    """Features used."""

    outlier_samples: list = field(default=Factory(list), validator=[validators.instance_of(list)])
    """Outlie samples identified."""

    preprocessing_pipeline = field(init=False)
    """Preprocessing pipeline for data scaling, imputation, and categorical encoding."""

    x_train_processed = field(default=None, init=False)
    """Training data after pre-processing (outlier, impuation, scaling)."""

    @property
    def outl_reduction(self) -> int:
        """Parameter outlier reduction method."""
        return self.config_training["outl_reduction"]

    @property
    def ml_model_type(self) -> str:
        """ML model type."""
        return self.config_training["ml_model_type"]

    @property
    def ml_model_params(self) -> dict:
        """ML model parameters."""
        return self.config_training["ml_model_params"]

    @property
    def x_train(self):
        """x_train."""
        return self.data_train[self.feature_columns]

    @property
    def x_dev_processed(self):
        """x_dev_processed."""
        processed_data = self.preprocessing_pipeline.transform(self.data_dev[self.feature_columns])
        # Convert back to DataFrame to preserve column names
        if hasattr(processed_data, "shape") and len(processed_data.shape) == 2:
            return pd.DataFrame(processed_data, columns=self.feature_columns, index=self.data_dev.index)
        return processed_data

    @property
    def x_test_processed(self):
        """x_test_processed."""
        processed_data = self.preprocessing_pipeline.transform(self.data_test[self.feature_columns])
        # Convert back to DataFrame to preserve column names
        if hasattr(processed_data, "shape") and len(processed_data.shape) == 2:
            return pd.DataFrame(processed_data, columns=self.feature_columns, index=self.data_test.index)
        return processed_data

    @property
    def y_train(self):
        """y_train."""
        if self.ml_type == "timetoevent":
            duration = self.data_train[self.target_assignments["duration"]]
            event = self.data_train[self.target_assignments["event"]]
            return np.array(
                list(zip(event, duration, strict=False)),
                dtype={"names": ("c1", "c2"), "formats": ("bool", "f8")},
            )
        else:
            return self.data_train[self.target_assignments.values()]

    @property
    def y_dev(self):
        """y_dev."""
        if self.ml_type == "timetoevent":
            duration = self.data_dev[self.target_assignments["duration"]]
            event = self.data_dev[self.target_assignments["event"]]
            return np.array(
                list(zip(event, duration, strict=False)),
                dtype={"names": ("c1", "c2"), "formats": ("bool", "f8")},
            )
        else:
            return self.data_dev[self.target_assignments.values()]

    @property
    def y_test(self):
        """y_dev."""
        if self.ml_type == "timetoevent":
            duration = self.data_test[self.target_assignments["duration"]]
            event = self.data_test[self.target_assignments["event"]]
            return np.array(
                list(zip(event, duration, strict=False)),
                dtype={"names": ("c1", "c2"), "formats": ("bool", "f8")},
            )
        else:
            return self.data_test[self.target_assignments.values()]

    def __attrs_post_init__(self):
        # Set up preprocessing pipeline
        self._setup_preprocessing_pipeline()

    def _setup_preprocessing_pipeline(self):
        """Set up the preprocessing pipeline with imputation and scaling only.

        Simplified pipeline that only handles:
        - Imputation: Fill missing values (median for numerical, most_frequent for categorical)
        - Scaling: StandardScaler for all columns (replaces existing columns)

        Note: Categorical encoding (one-hot, ordinal) is handled elsewhere in the pipeline.
        """
        # Get sample data to determine column types
        sample_data = self.data_train[self.feature_columns]

        # Identify numerical and categorical columns
        numerical_columns = []
        categorical_columns = []

        for col in self.feature_columns:
            if sample_data[col].dtype in ["object", "category", "bool"]:
                categorical_columns.append(col)
            else:
                numerical_columns.append(col)

        # Create transformers for different column types
        transformers = []

        # Numerical columns: imputation (median) + scaling (StandardScaler)
        # This is the correct order: impute missing values first, then scale
        if numerical_columns:
            numerical_transformer = Pipeline(
                steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
            )
            transformers.append(("num", numerical_transformer, numerical_columns))

        # Categorical columns: imputation only (most_frequent)
        # No encoding is performed - categorical values are preserved as-is
        if categorical_columns:
            categorical_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="most_frequent"))])
            transformers.append(("cat", categorical_transformer, categorical_columns))

        # Create the column transformer
        if transformers:
            self.preprocessing_pipeline = ColumnTransformer(
                transformers=transformers,
                remainder="passthrough",  # Keep any remaining columns as-is
                verbose_feature_names_out=False,  # Keep original feature names where possible
            )
        else:
            # If no transformers needed, create a simple pipeline with just imputation and scaling
            self.preprocessing_pipeline = Pipeline(
                steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
            )

    # Training class functionality:
    # (1) outlier removal
    # (2) preprocessing pipeline
    # (3) model training
    # (4) model predictions
    # (5) calculate feature importance, on request

    def fit(self):
        """Preprocess and fit model."""
        # use copy of all train variables, as they may be change due to outlier detec.
        data_train = self.data_train.copy()
        x_train = self.x_train.copy()
        y_train = self.y_train.copy()

        # (1) outlier removal in x_train
        if self.outl_reduction > 0:
            # IsolationForest for outlier detection
            clf = IsolationForest(
                contamination=self.outl_reduction / len(x_train),
                random_state=42,
                n_jobs=1,
            )
            clf.fit(x_train)

            # Get the outlier prediction labels
            # (-1:outliers, 1:inliers)
            outlier_pred = clf.predict(x_train)
            # sometimes there seems to be a mismatch in the number of outliers
            # assert self.outl_reduction == np.sum(outlier_pred == -1)
            # print("Number of outliers specified:", self.outl_reduction)
            # print("Number of outliers found:", np.sum(outlier_pred == -1))

            # identify outlier samples
            self.outlier_samples = data_train[outlier_pred == -1][self.row_column].tolist()
            # print("Outlier samples:", self.outlier_samples)

            # Remove outliers from data_train, x_train, y_train
            data_train = data_train[outlier_pred == 1].copy()
            x_train = x_train[outlier_pred == 1].copy()
            y_train = y_train[outlier_pred == 1].copy()

        # (2) Imputation and scaling (after outlier removal)
        processed_data = self.preprocessing_pipeline.fit_transform(x_train)
        # Convert back to DataFrame to preserve column names
        if hasattr(processed_data, "shape") and len(processed_data.shape) == 2:
            self.x_train_processed = pd.DataFrame(processed_data, columns=self.feature_columns, index=x_train.index)
        else:
            self.x_train_processed = processed_data

        # (3) Model training
        self.model = ModelInventory().get_model_instance(self.ml_model_type, self.ml_model_params)

        if len(self.target_assignments) == 1:
            # standard sklearn single target models
            self.model.fit(
                self.x_train_processed,
                y_train.squeeze(axis=1),
            )
        else:
            # multi target models, incl. time2event
            self.model.fit(self.x_train_processed, y_train)

        # (4) Model prediction
        self.predictions["train"] = pd.DataFrame()
        self.predictions["train"][self.row_column] = data_train[self.row_column]
        self.predictions["train"]["prediction"] = self.model.predict(self.x_train_processed)

        self.predictions["dev"] = pd.DataFrame()
        self.predictions["dev"][self.row_column] = self.data_dev[self.row_column]
        self.predictions["dev"]["prediction"] = self.model.predict(self.x_dev_processed)

        self.predictions["test"] = pd.DataFrame()
        self.predictions["test"][self.row_column] = self.data_test[self.row_column]
        self.predictions["test"]["prediction"] = self.model.predict(self.x_test_processed)

        # special treatment of targets due to sklearn
        if len(self.target_assignments) == 1:
            target_col = list(self.target_assignments.values())[0]
            self.predictions["train"][target_col] = y_train.squeeze(axis=1)
            self.predictions["dev"][target_col] = self.y_dev.squeeze(axis=1)
            self.predictions["test"][target_col] = self.y_test.squeeze(axis=1)
        else:
            for target_col in self.target_assignments.values():
                self.predictions["train"][target_col] = data_train[target_col]
                self.predictions["dev"][target_col] = self.data_dev[target_col]
                self.predictions["test"][target_col] = self.data_test[target_col]

        # add additional predictions for classifications
        if self.ml_type == "classification":
            columns = [int(x) for x in self.model.classes_]  # column names --> int
            self.predictions["train"][columns] = self.model.predict_proba(self.x_train_processed)
            self.predictions["dev"][columns] = self.model.predict_proba(self.x_dev_processed)
            self.predictions["test"][columns] = self.model.predict_proba(self.x_test_processed)

        # add additional predictions for time to event predictions
        if self.ml_type == "timetoevent":
            pass

        # calculate used features, but only if required for optuna max_features>0
        # (to save time, shap or permutation importances may take a lot of time)
        if self.max_features > 0:
            self.features_used = self._calculate_features_used()
        else:
            self.features_used = []

        return self

    def _calculate_features_used(self):
        """Calculate used features, method based on model type."""
        feature_method = ModelInventory().get_feature_method(self.ml_model_type)

        if feature_method == "internal":
            self.calculate_fi_internal()
            fi_df = self.feature_importances["internal"]
        elif feature_method == "shap":
            self.calculate_fi_featuresused_shap(partition="dev")
            fi_df = self.feature_importances["shap_dev"]
        elif feature_method == "permutation":
            self.calculate_fi_permutation(partition="dev", n_repeats=2)  # only 2 repeats!
            fi_df = self.feature_importances["permutation_dev"]
        elif feature_method == "constant":
            self.calculate_fi_constant()
            fi_df = self.feature_importances["constant"]
        else:
            raise ValueError("feature method provided in model config not supported")

        return fi_df[fi_df["importance"] != 0]["feature"].tolist()

    def calculate_fi_constant(self):
        """Provide flat feature importance table."""
        fi_df = pd.DataFrame()
        fi_df["feature"] = self.feature_columns
        fi_df["importance"] = 1
        self.feature_importances["constant"] = fi_df

    def calculate_fi_internal(self):
        """Sklearn-provided internal feature importance (based on train dataset)."""
        # Handle unsupported "timetoevent" case as in your original code
        if getattr(self, "ml_type", None) == "timetoevent":
            fi_df = pd.DataFrame(columns=["feature", "importance"])
            logger.warning("Internal features importances not available for timetoevent.")
            self.feature_importances["internal"] = fi_df
            return

        # 1) Tree-based models exposing feature_importances_
        if hasattr(self.model, "feature_importances_"):
            fi = np.asarray(self.model.feature_importances_)
            fi_df = pd.DataFrame({"feature": self.feature_columns, "importance": fi})
            self.feature_importances["internal"] = fi_df
            return

        # 2) Linear models exposing coef_: Ridge, LinearSVC/LinearSVR, SVC/SVR with kernel='linear'
        if hasattr(self.model, "coef_"):
            coef = np.asarray(self.model.coef_)
            # coef_ can be:
            # - shape (n_features,) for single-target regression (e.g., Ridge)
            # - shape (n_targets, n_features) for multi-target regression
            # - shape (n_classes, n_features) for LinearSVC/LinearSVR (OvR)
            # - shape (n_class_pairs, n_features) for SVC with kernel='linear' (OvO)
            if coef.ndim == 1:
                importance = np.abs(coef)
            else:
                # Aggregate across classes/targets/pairs
                importance = np.mean(np.abs(coef), axis=0)

            if len(importance) != len(self.feature_columns):
                # Defensive check in case columns mismatch model coefficients
                logger.warning(
                    "Length mismatch between coefficients (%d) and feature columns (%d). "
                    "Skipping internal importances.",
                    len(importance),
                    len(self.feature_columns),
                )
                fi_df = pd.DataFrame(columns=["feature", "importance"])
            else:
                fi_df = pd.DataFrame({"feature": self.feature_columns, "importance": importance})

            self.feature_importances["internal"] = fi_df
            return

        # Fallback
        fi_df = pd.DataFrame(columns=["feature", "importance"])
        logger.warning("Internal features importances not available for this estimator.")
        self.feature_importances["internal"] = fi_df

    def calculate_fi_group_permutation(self, partition="dev", n_repeats=10):
        """Permutation feature importance, group version."""
        logger.set_log_group(LogGroup.TRAINING, f"{self.training_id}")

        logger.info(f"Calculating permutation feature importances ({partition}). This may take a while...")
        np.random.seed(42)  # reproducibility
        # fixed confidence level
        confidence_level = 0.95
        feature_columns = self.feature_columns
        target_assignments = self.target_assignments
        target_metric = self.target_metric
        model = self.model
        feature_groups = self.feature_groups

        if partition == "dev":
            # concat processed input + target columns
            target_col = list(target_assignments.values())[0]
            data = pd.concat([self.x_dev_processed, self.data_dev[[target_col]]], axis=1)
        elif partition == "test":
            target_col = list(target_assignments.values())[0]
            data = pd.concat([self.x_test_processed, self.data_test[[target_col]]], axis=1)

        if not set(feature_columns).issubset(data.columns):
            raise ValueError("Features missing in provided dataset.")

        # check that targets are in dataset
        # MISSING

        # keep all features and add group features
        # create features dict
        feature_columns_dict = {x: [x] for x in feature_columns}
        features_dict = {**feature_columns_dict, **feature_groups}

        # calculate baseline score
        baseline_score = get_performance_score(model, data, feature_columns, target_metric, target_assignments)

        results_df = pd.DataFrame(
            columns=[
                "feature",
                "importance",
                "stddev",
                "p-value",
                "n",
                "ci_low_95",
                "ci_high_95",
            ]
        )
        # calculate pfi
        for name, feature in features_dict.items():
            data_pfi = data.copy()
            fi_lst = list()

            for _ in range(n_repeats):
                # replace column with random selection from that column of data_all
                # we use data_all as the validation dataset may be small
                for feat in feature:
                    data_pfi[feat] = np.random.choice(data[feat], len(data_pfi), replace=False)
                pfi_score = get_performance_score(model, data_pfi, feature_columns, target_metric, target_assignments)
                fi_lst.append(baseline_score - pfi_score)

            # calculate statistics
            pfi_mean = np.mean(fi_lst)
            n = len(fi_lst)
            p_value = np.nan
            stddev = np.std(fi_lst, ddof=1) if n > 1 else np.nan
            if stddev not in (np.nan, 0):
                t_stat = pfi_mean / (stddev / math.sqrt(n))
                p_value = scipy.stats.t.sf(t_stat, n - 1)
            elif stddev == 0:
                p_value = 0.5

            # calculate confidence intervals
            if any(np.isnan(val) for val in [stddev, n, pfi_mean]) or n == 1:
                ci_high = np.nan
                ci_low = np.nan
            else:
                t_val = scipy.stats.t.ppf(1 - (1 - confidence_level) / 2, n - 1)
                ci_high = pfi_mean + t_val * stddev / math.sqrt(n)
                ci_low = pfi_mean - t_val * stddev / math.sqrt(n)

            # save results
            results_df.loc[len(results_df)] = [
                name,
                pfi_mean,
                stddev,
                p_value,
                n,
                ci_low,
                ci_high,
            ]

        results_df = results_df.sort_values(by="importance", ascending=False)
        self.feature_importances["permutation" + "_" + partition] = results_df

    def calculate_fi_permutation(self, partition="dev", n_repeats=10):
        """Permutation feature importance."""
        logger.info(f"Calculating permutation feature importances ({partition}). This may take a while...")
        np.random.seed(42)  # reproducibility
        if self.ml_type == "timetoevent":
            # sksurv models only provide inbuilt scorer (CI)
            # more work needed to support other metrics
            scoring_type = None
        else:
            scoring_type = scorer_string_inventory[self.target_metric]

        if partition == "dev":
            x = self.x_dev_processed
            y = self.y_dev
        elif partition == "test":
            x = self.x_test_processed
            y = self.y_test

        perm_importance = permutation_importance(
            self.model,
            X=x,
            y=y,
            n_repeats=n_repeats,
            random_state=0,
            scoring=scoring_type,
        )

        fi_df = pd.DataFrame()
        fi_df["feature"] = self.feature_columns
        fi_df["importance"] = perm_importance.importances_mean
        fi_df["importance_std"] = perm_importance.importances_std
        self.feature_importances["permutation" + "_" + partition] = fi_df

    def calculate_fi_lofo(self):
        """LOFO feature importance."""
        np.random.seed(42)  # reproducibility
        logger.info("Calculating LOFO feature importance. This may take a while...")
        # first, dev only
        feature_columns = self.feature_columns
        target_assignments = self.target_assignments
        # calculate dev+test baseline scores
        target_col = list(target_assignments.values())[0]
        data_dev = pd.concat([self.x_dev_processed, self.data_dev[[target_col]]], axis=1)
        data_test = pd.concat([self.x_test_processed, self.data_test[[target_col]]], axis=1)

        baseline_dev = get_performance_score(
            self.model,
            data_dev,
            feature_columns,
            self.target_metric,
            self.target_assignments,
        )
        baseline_test = get_performance_score(
            self.model,
            data_test,
            feature_columns,
            self.target_metric,
            self.target_assignments,
        )

        # create features dict
        feature_columns_dict = {x: [x] for x in feature_columns}
        lofo_features = {**feature_columns_dict, **self.feature_groups}

        # lofo
        fi_dev_df = pd.DataFrame(columns=["feature", "importance"])
        fi_test_df = pd.DataFrame(columns=["feature", "importance"])
        for name, lofo_feature in lofo_features.items():
            selected_features = copy.deepcopy(feature_columns)
            model = copy.deepcopy(self.model)
            selected_features = [x for x in selected_features if x not in lofo_feature]
            # retrain model
            if len(self.target_assignments) == 1:
                # standard sklearn single target models
                model.fit(
                    self.x_train_processed[selected_features],
                    self.y_train.squeeze(axis=1),
                )
            else:
                # multi target models, incl. time2event
                model.fit(self.x_train_processed[selected_features], self.y_train)

            # get lofo dev + test scores
            score_dev = get_performance_score(
                model,
                data_dev,
                selected_features,
                self.target_metric,
                self.target_assignments,
            )
            score_test = get_performance_score(
                model,
                data_test,
                selected_features,
                self.target_metric,
                self.target_assignments,
            )

            fi_dev_df.loc[len(fi_dev_df)] = [name, baseline_dev - score_dev]
            fi_test_df.loc[len(fi_test_df)] = [name, baseline_test - score_test]

        self.feature_importances["lofo" + "_dev"] = fi_dev_df
        self.feature_importances["lofo" + "_test"] = fi_test_df

    def calculate_fi_featuresused_shap(self, partition="dev", bg_max=200):
        """Shap feature importance, specifically for calc_features_used."""
        if getattr(self, "ml_type", None) == "timetoevent":
            raise ValueError("SHAP feature importance not supported for timetoevent")

        X_eval = {"dev": self.x_dev_processed, "test": self.x_test_processed}.get(partition)
        if X_eval is None:
            raise ValueError("dataset type not supported")

        # small background for speed
        X_bg = self.x_train_processed
        if hasattr(X_bg, "sample") and X_bg.shape[0] > bg_max:
            X_bg = X_bg.sample(n=bg_max, replace=False, random_state=0)

        # Try auto; if it fails (e.g., SVR/GPR), pass a callable + masker
        try:
            explainer = shap.Explainer(self.model, X_bg, feature_names=list(self.x_train_processed.columns))
            explanation = explainer(X_eval)
        except Exception:
            if hasattr(self.model, "predict_proba"):
                explainer = shap.Explainer(self.model.predict_proba, X_bg, link="logit")
                explanation = explainer(X_eval, link="logit")
            else:
                explainer = shap.Explainer(self.model.predict, X_bg)
                explanation = explainer(X_eval)

        vals = np.asarray(explanation.values)
        n_features = len(self.feature_columns)

        # Aggregate absolute SHAP values to per-feature importance
        if vals.ndim == 2 and vals.shape[1] == n_features:
            importance = np.abs(vals).mean(axis=0)
        else:
            # find the feature axis and average over the rest
            feat_axes = [i for i, d in enumerate(vals.shape) if d == n_features]
            if len(feat_axes) != 1:
                raise ValueError(f"Unexpected SHAP values shape {vals.shape}")
            feat_axis = feat_axes[0]
            importance = np.mean(np.abs(vals), axis=tuple(i for i in range(vals.ndim) if i != feat_axis))

        if len(importance) != n_features:
            raise ValueError("Feature count mismatch between SHAP values and feature_columns.")

        fi_df = pd.DataFrame({"feature": self.feature_columns, "importance": importance})
        if not fi_df["importance"].empty:
            fi_df = fi_df[fi_df["importance"] > fi_df["importance"].max() / 1000.0]

        self.feature_importances[f"shap_{partition}"] = fi_df

    def calculate_fi_shap(self, partition="dev", shap_type="kernel"):
        """Compute SHAP feature importance with a model-agnostic explainer."""
        logger.info(f"Calculating SHAP feature importances ({partition})...")

        # Prediction function for SHAP
        predict_fn = (
            self.model.predict_proba if getattr(self, "ml_type", None) == "classification" else self.model.predict
        )

        # Select data
        if partition == "dev":
            data = self.x_dev_processed
        elif partition == "test":
            data = self.x_test_processed
        else:
            raise ValueError("dataset type not supported")

        # Select explainer
        explainers = {
            "exact": shap.explainers.Exact,
            "permutation": shap.explainers.Permutation,
            "kernel": shap.explainers.Kernel,
        }
        explainer_cls = explainers.get(shap_type)
        if explainer_cls is None:
            raise ValueError(f"SHAP type {shap_type} not supported.")
        explainer = explainer_cls(predict_fn, data)

        # Compute SHAP values (Explanation API in shap>=0.48)
        vals = np.asarray(explainer(data).values)  # shape varies by model/output
        n_features = data.shape[1]

        # Aggregate absolute SHAP to per-feature importances
        if vals.ndim == 2 and vals.shape[1] == n_features:
            importance = np.abs(vals).mean(axis=0)
        else:
            # Find the feature axis and average over others
            feat_axes = [i for i, d in enumerate(vals.shape) if d == n_features]
            if len(feat_axes) != 1:
                raise ValueError(f"Unexpected SHAP values shape {vals.shape}")
            feat_axis = feat_axes[0]
            importance = np.mean(np.abs(vals), axis=tuple(i for i in range(vals.ndim) if i != feat_axis))

        # Build importance DataFrame
        fi_df = pd.DataFrame({"feature": data.columns.tolist(), "importance": importance})
        if not fi_df["importance"].empty:
            fi_df = fi_df[fi_df["importance"] > fi_df["importance"].max() / 1000.0]

        self.feature_importances[f"shap_{partition}"] = fi_df

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        """Predict.

        Args:
            x: Input data to make predictions on. Should have the same structure as training data.

        Returns:
            Predictions from the model.
        """
        # Apply the same preprocessing pipeline used during training
        x_processed = self.preprocessing_pipeline.transform(x)
        return self.model.predict(x_processed)

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        """Predict_proba.

        Args:
            x: Input data to make probability predictions on. Should have the same structure as training data.

        Returns:
            Probability predictions from the model.
        """
        # Apply the same preprocessing pipeline used during training
        x_processed = self.preprocessing_pipeline.transform(x)
        return self.model.predict_proba(x_processed)

    def to_pickle(self, path):
        """Save training."""

    @classmethod
    def from_pickle(cls, path):
        """Load training."""
