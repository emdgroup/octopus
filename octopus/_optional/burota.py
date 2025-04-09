"""Optional boruta imports."""

from octopus.exceptions import OptionalImportError

try:
    from boruta import BorutaPy


except ModuleNotFoundError as ex:
    raise OptionalImportError(
        "Boruta is unavailable because the necessary optional "
        "dependencies are not installed. "
        'Consider installing Octopus with "boruta" dependency, '
        'e.g. via `pip install -e ".[boruta]"`.'
    ) from ex

__all__ = ["BorutaPy"]
