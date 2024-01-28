"""Opto Config module."""
import json
import pickle

from attrs import asdict, define, field, validators


@define
class OctoConfig:
    """Octo Config."""

    # --- mandatory attributes ---
    cfg_manager: dict = field(validator=[validators.instance_of(dict)])
    cfg_sequence: list = field(validator=[validators.instance_of(list)])
    study_name: str = field(validator=[validators.instance_of(str)])
    output_path: str = field(validator=[validators.instance_of(str)])
    ml_type: str = field(
        validator=[
            validators.in_(["classification", "regression", "timetoevent"]),
        ],
    )

    # --- Outer loop config ---
    production_mode: bool = field(
        default=True, validator=[validators.instance_of(bool)]
    )

    k_outer: int = field(default=5, validator=[validators.instance_of(int)])
    target_metric: str = field(
        default="AUCROC",
        validator=[
            validators.in_(["AUCROC", "ACCBAL", "ACC", "LOGLOSS", "MAE", "MSE", "R2"]),
        ],
    )

    metrics: list = field(
        default=["AUCROC", "ACCBAL", "ACC", "LOGLOSS", "MAE", "MSE", "R2"],
        validator=[validators.instance_of(list)],
    )

    datasplit_seed_outer: int = field(
        default=1234, validator=[validators.instance_of(int)]
    )

    def to_json(self, filename):
        """Save config to json file."""
        with open(filename, "w", encoding="utf-8") as file:
            json.dump(asdict(self), file)

    def to_pickle(self, filename):
        """Save object to dill file."""
        with open(filename, "wb") as file:
            pickle.dump(self, file)

    @classmethod
    def from_pickle(cls, filename):
        """Load object from dill file."""
        with open(filename, "rb") as file:
            return pickle.load(file)
