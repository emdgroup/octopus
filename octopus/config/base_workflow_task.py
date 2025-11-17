"""Config for Base Workflow Task item."""

from attrs import define, field, validators


@define
class BaseWorkflowTask:
    """Base class for all workflow tasks.

    Contains all common parameters for all workflow tasks.
    """

    task_id: int = field(validator=[validators.instance_of(int), validators.ge(0)])
    """Task ID, greater or equal than 0."""

    depends_on_task: int = field(default=-1, validator=[validators.instance_of(int), validators.ge(-1)])
    """Specify ID of input task. Input ID of start task: -1."""

    load_task: bool = field(default=False, validator=[validators.instance_of(bool)])
    """Whether to load the task item."""

    description: str = field(default="", validator=[validators.instance_of(str)])
    """Description for the workflow task."""

    categorical_encoding: bool = field(default=False, validator=[validators.instance_of(bool)])
    """Enforce categorical encoding on module level (and not model) to stay compatible with feature importances"""
