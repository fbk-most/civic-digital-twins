"""Civic Digital Twins Modelling Framework."""

import sys
import warnings

# Deprecation warning for Python 3.11
if sys.version_info < (3, 12):  # pragma: no cover
    warnings.warn(
        "Python 3.11 support is deprecated and will be removed in a future version. "
        "Please upgrade to Python 3.12 or later.",
        DeprecationWarning,
        stacklevel=2,
    )
