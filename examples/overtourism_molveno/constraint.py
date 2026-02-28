"""Constraint definition for the overtourism vertical.

A constraint represents the relationship between the capacity of a resource
and its usage.  Two types are modelled:

1. **Deterministic** — capacity is a graph node; sustainability is a
   hard threshold (``usage <= capacity``).
2. **Probabilistic** — capacity is a random variable; sustainability is
   expressed as the probability that capacity exceeds usage
   (``1 - CDF(usage)``).
"""

from __future__ import annotations

from dataclasses import dataclass

from civic_digital_twins.dt_model.engine.frontend import graph
from civic_digital_twins.dt_model.model.index import Index


@dataclass(eq=False)
class Constraint:
    """Named pairing of a usage formula and a capacity index.

    Identity-based hashing (``eq=False``) keeps ``Constraint`` objects usable
    as dict keys, matching the convention used by ``graph.Node`` and
    ``GenericIndex``.
    """

    name: str
    usage: graph.Node       # raw formula node
    capacity: Index         # index (for .node, .value, Distribution checks)
