"""Config manager."""

from attrs import Factory, define, field, validators


@define
class ConfigManager:
    """Configuration for manager parameters.

    Will later be used to connect to HPC.
    """

    outer_parallelization: bool = field(default=Factory(lambda: True), validator=[validators.instance_of(bool)])
    """Indicates whether outer parallelization is enabled. Defaults to True."""

    run_single_experiment_num: int = field(default=Factory(lambda: -1), validator=[validators.instance_of(int)])
    """Select a single experiment to execute. Defaults to -1 to run all experiments"""
