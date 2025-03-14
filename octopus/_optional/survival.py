from octopus.exceptions import OptionalImportError

try:
    from sksurv.ensemble import ExtraSurvivalTrees
    from sksurv.metrics import concordance_index_censored


except ModuleNotFoundError as ex:
    raise OptionalImportError(
        "ml_type 'survival' is unavailable because the necessary optional "
        "dependencies are not installed. "
        "Consider installing Octopus with 'survival' dependency, "
        "e.g. via `pip install octopus[survival]`."
    ) from ex

__all__ = ["ExtraSurvivalTrees", "concordance_index_censored"]
