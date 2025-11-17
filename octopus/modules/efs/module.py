"""Efs module."""

from typing import ClassVar

from attrs import define, field, validators

from octopus.config.base_workflow_task import BaseWorkflowTask


@define
class Efs(BaseWorkflowTask):
    """EFS Config."""

    module: ClassVar[str] = "efs"
    """Module name."""

    description: str = field(validator=[validators.instance_of(str)], default="")
    """Description."""

    model: str = field(validator=[validators.instance_of(str)], default="")
    """Model used by EFS."""

    subset_size: int = field(validator=[validators.instance_of(int)], default=30)
    """Number of features in the subset."""

    n_subsets: int = field(validator=[validators.instance_of(int)], default=100)
    """Number of subsets."""

    cv: int = field(validator=[validators.instance_of(int)], default=5)
    """Number of CV folds for EFS."""

    max_n_iterations: int = field(validator=[validators.instance_of(int)], default=50)
    """Number of iterations for ensemble optimization."""

    max_n_models: int = field(validator=[validators.instance_of(int)], default=30)
    """Maximum number of models used in optimization, pruning."""
