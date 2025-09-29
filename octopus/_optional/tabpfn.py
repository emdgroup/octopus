"""Optional tabpfn imports."""

from pathlib import Path

from octopus.exceptions import OptionalImportError

try:
    from tabpfn import TabPFNClassifier, TabPFNRegressor
except ModuleNotFoundError as ex:
    raise OptionalImportError(
        "TabPFN is unavailable because the necessary optional "
        "dependencies are not installed. "
        'Consider installing Octopus with "tabpfn" dependency, '
        'e.g. via `pip install -e ".[tabpfn]"`.'
    ) from ex


# Check if required TabPFN model files exist
def _check_tabpfn_models():
    """Check if required TabPFN model files exist and raise error if not."""
    try:
        from .tabpfn_utils import get_tabpfn_model_path  # noqa: PLC0415
    except ImportError:
        # If utils are not available, skip the check
        return

    classifier_path = Path(get_tabpfn_model_path("classifier"))
    regressor_path = Path(get_tabpfn_model_path("regressor"))

    missing_models = []
    if not classifier_path.exists():
        missing_models.append(f"Classifier model: {classifier_path}")
    if not regressor_path.exists():
        missing_models.append(f"Regressor model: {regressor_path}")

    if missing_models:
        missing_list = "\n  - ".join(missing_models)
        raise OptionalImportError(
            f"TabPFN model files are missing:\n  - {missing_list}\n\n"
            "Please download the required models by running:\n"
            "  python octopus/scripts/combined_tabpfn_download.py\n\n"
            "This will download both the classifier and regressor models "
            "to the appropriate cache directory."
        )


# Perform the model check when the module is imported
_check_tabpfn_models()

__all__ = [
    "TabPFNClassifier",
    "TabPFNRegressor",
]
