"""Init modules."""

# Import the classes from their respective modules
from .mrmr.module import Mrmr
from .octo.module import Octo
from .rfe.module import Rfe
from .roc.module import Roc
from .sfs.module import Sfs

# Define the __all__ variable to specify what is available to import from this package
__all__ = ["Mrmr", "Octo", "Rfe", "Roc", "Sfs"]
