"""Model hyperparameter class."""

from typing import Any, List, Union

from attrs import define, field


@define
class Hyperparameter:
    """Class to create hyperparameter space."""

    type: str
    name: str
    low: Union[int, float, None] = None
    high: Union[int, float, None] = None
    step: Union[int, float, None] = None
    choices: List[Any] = field(factory=list)
    log: bool = False
    value: Any = None  # For fixed values
