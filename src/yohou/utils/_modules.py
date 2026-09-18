"""Detection of optional third-party libraries without importing them.

Lives under ``yohou.utils`` rather than beside its first caller because both
``yohou.base`` and ``yohou.model_selection`` need it, and ``model_selection``
already imports ``base`` at module level, so the reverse import would be
circular.
"""

import sys
from typing import Any


def _loaded_module(name: str) -> Any:
    """Return an already imported library module, without importing it.

    An estimator of a library can only exist once that library is imported, so
    an absent module means the estimator cannot belong to it.

    Parameters
    ----------
    name : str
        Top-level module name.

    Returns
    -------
    module or None
        The module, or None when it has not been imported.

    """
    return sys.modules.get(name)
