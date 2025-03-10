"""Optional tabpfn imports."""

from octopus.exceptions import OptionalImportError

try:
    from tabpfn import TabPFNClassifier, TabPFNRegressor


except ModuleNotFoundError as ex:
    raise OptionalImportError(
        "TabPFN is unavailable because the necessary optional "
        "dependencies are not installed. "
        "Consider installing Octopus with 'tabpfn' dependency, "
        "e.g. via `pip install octopus[tabpfn]`."
    ) from ex

__all__ = [
    "TabPFNClassifier",
    "TabPFNRegressor",
]
