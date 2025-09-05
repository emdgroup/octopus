"""AutoGluon module."""

from typing import ClassVar, Literal

from attrs import define, field, validators

from octopus.config.base_sequence_item import BaseSequenceItem


@define
class AutoGluon(BaseSequenceItem):
    """AG Config."""

    module: ClassVar[str] = "autogluon"
    """Module name."""

    description: str | None = field(default="", validator=validators.instance_of(str))
    """Description."""

    verbosity: int = field(default=2, validator=validators.instance_of(int))
    """Verbosity levels control how much information is printed."""
    # 0: Only log exceptions
    # 1: Only log warnings + exceptions
    # 2: Standard logging
    # 3: Verbose logging (ex: log validation score every 50 iterations)
    # 4: Maximally verbose logging (ex: log validation score every iteration)

    time_limit: int | None = field(default=None, validator=validators.optional(validators.instance_of(int)))
    """Approximately, how long a fit should run, in seconds. Default: No limit."""

    infer_limit: int | None = field(default=None, validator=validators.optional(validators.instance_of(int)))
    """ Inference time limit in seconds per row to adhere to during fit."""

    memory_limit: float | Literal["auto"] = field(
        default="auto",
        validator=validators.optional(validators.or_(validators.instance_of(float), validators.in_(["auto"]))),
    )
    """Amount of memory in GB you want AutoGluon predictor to use."""

    fit_strategy: Literal["sequential"] = field(
        default="sequential", validator=validators.in_(["sequential", "parallel"])
    )
    """The strategy used to fit models."""

    presets: list[str] = field(
        default=["medium_quality"],
        validator=validators.deep_iterable(
            member_validator=validators.and_(
                validators.instance_of(str),
                validators.in_(
                    [
                        "best_quality",
                        "high_quality",
                        "good_quality",
                        "medium_quality",
                        "experimental_quality",
                        "optimize_for_deployment",
                        "interpretable",
                        "ignore_text",
                    ]
                ),
            ),
            iterable_validator=validators.instance_of(list),
        ),
    )
    """Autogluon presets."""
    # best_quality: Best predictive accuracy, high inference time and disk usage.
    # high_quality: High predictive accuracy, fast inference. ~8x faster than best.
    # good_quality: Good predictive accuracy, fast inference. ~4x faster than high.
    # medium_quality: Medium accuracy, fast inference and training time. ~20x faster.
    # experimental_quality: testing ground, later be added to the best_quality preset.
    # optimize_for_deployment: deletes unused models and removes training artifacts.
    # interpretable: Trades off predictive accuracy for conciseness.
    # ignore_text: Disables automated feature generation for text features.

    num_cpus: int | Literal["auto"] = field(
        default="auto",
        validator=validators.optional(validators.or_(validators.instance_of(int), validators.in_(["auto"]))),
    )
    """Number of CPUs used by Autogluon instance. Can be an integer or "auto"."""

    num_bag_folds: int = field(default=5, validator=[validators.instance_of(int), validators.gt(1)])
    """Number of cross validation folds."""

    included_model_types: list[str] | None = field(
        default=None,
        validator=validators.optional(
            validators.deep_iterable(
                member_validator=validators.and_(
                    validators.instance_of(str),
                    validators.in_(
                        [
                            "GBM",  # LightGBM
                            "CAT",  # CatBoost
                            "XGB",  # XGBoost
                            "RF",  # Random Forest
                            "XT",  # Extremely Randomized Trees
                            "KNN",  # K-Nearest Neighbors
                            "LR",  # Linear Regression
                            "NN_TORCH",  # Neural Network implemented in Pytorch
                            "FASTAI",  # Neural Network with FastAI backend
                        ]
                    ),
                ),
                iterable_validator=validators.optional(validators.instance_of(list)),
            )
        ),
    )
    """Includes only listed model types for training during fit."""
