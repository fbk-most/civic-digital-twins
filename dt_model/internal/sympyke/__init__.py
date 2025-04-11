"""SymPy compatibility layer.

This package implements a tiny SymPy compatibility layer that maps
sympy-like methods and functions to frontend.graph operations.

This package is *internal*. Feel free to use it in your code but know that
there are no API stability guarantees whatsoever.
"""

from .operators import Eq
from .piecewise import Piecewise
from .symbol import Symbol

__all__ = ["Eq", "Piecewise", "Symbol"]
