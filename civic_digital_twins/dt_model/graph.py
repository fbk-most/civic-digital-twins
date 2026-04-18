"""Public graph-building helpers for dt_model users.

This module re-exports the subset of ``engine.frontend.graph`` that is
intended for use in model definitions.  Import it as::

    from civic_digital_twins.dt_model import graph

    peak = Index("peak", graph.piecewise((1.8, season == "summer"), (1.0, True)))
    smoothed = TimeseriesIndex("smoothed", graph.function_call("smooth", demand_ts))

The raw engine path (``dt_model.engine.frontend.graph``) remains available
for engine-level work (DAG construction, backends, debugging).
"""

from .engine.frontend.graph import (
    Node,
    exp,
    function_call,
    log,
    maximum,
    piecewise,
    where,
)

__all__ = [
    "Node",
    "exp",
    "function_call",
    "log",
    "maximum",
    "piecewise",
    "where",
]
