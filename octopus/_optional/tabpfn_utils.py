"""Utilities for TabPFN cross-platform model path resolution."""

import platform
from pathlib import Path


def get_tabpfn_cache_dir() -> Path:
    """Get the default TabPFN cache directory for the current platform.

    Returns:
        Path: The platform-specific cache directory for TabPFN models.
    """
    system = platform.system().lower()

    if system == "windows":
        # Windows: %USERPROFILE%\.cache\tabpfn
        cache_base = Path.home() / ".cache"
    elif system == "darwin":  # macOS
        # macOS: ~/Library/Caches or ~/.cache
        cache_base = Path.home() / "Library" / "Caches"
        # Fallback to ~/.cache if Library/Caches doesn't exist
        if not cache_base.exists():
            cache_base = Path.home() / ".cache"
    else:  # Linux and other Unix-like systems
        # Linux: ~/.cache
        cache_base = Path.home() / ".cache"

    return cache_base / "tabpfn"


def get_tabpfn_model_path(model_type: str) -> Path:
    """Get the full path to a TabPFN model file.

    Args:
        model_type: Either 'classifier' or 'regressor'

    Returns:
        str: Full path to the model file

    Raises:
        ValueError: If model_type is not 'classifier' or 'regressor'
    """
    if model_type not in ["classifier", "regressor"]:
        raise ValueError("model_type must be 'classifier' or 'regressor'")

    cache_dir = get_tabpfn_cache_dir()
    model_filename = f"tabpfn-v2-{model_type}.ckpt"

    return cache_dir / model_filename
