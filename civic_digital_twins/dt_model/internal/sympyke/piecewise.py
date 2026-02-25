"""Backward-compatibility shim for graph.piecewise.

The implementation has moved to ``engine.frontend.graph.piecewise``.
This module re-exports it as ``Piecewise`` (uppercase) to preserve the
sympy-compatible API used by existing callers.
"""

from ...engine.frontend.graph import piecewise as Piecewise

__all__ = ["Piecewise"]
