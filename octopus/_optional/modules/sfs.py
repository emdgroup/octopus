"""Optional boruta imports."""

from octopus.exceptions import OptionalImportError

try:
    from mlxtend.feature_selection import SequentialFeatureSelector as SFS


except ModuleNotFoundError as ex:
    raise OptionalImportError(
        "SFS is unavailable because the necessary optional "
        "dependencies are not installed. "
        "Consider installing Octopus with 'sfs' dependency, "
        "e.g. via `pip install octopus[sfs]`."
    ) from ex

__all__ = ["SFS"]
