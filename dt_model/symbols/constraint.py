"""
This module defines the Constraint symbol.

A constraint represents the relationship between the capacity of a given
resource and the usage of that resource. We model two types of constraints:

1. deterministic constraints where the capacity is a graph node

2. probabilistic constraints where the capacity is a random variable
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from ..engine.frontend import graph


@runtime_checkable
class CumulativeDistribution(Protocol):
    """Protocol for classes allowing to sample from a cumulative distribution."""

    def cdf(self, x: float | np.ndarray, *args, **kwds) -> float | np.ndarray: ...


class Constraint:
    """
    Constraint class.

    This class is used to define constraints for the model.
    """

    def __init__(
        self,
        usage: graph.Node,
        capacity: graph.Node | CumulativeDistribution,
        group: str | None = None,
        name: str = "",
    ) -> None:
        self.usage = usage
        self.capacity = capacity
        self.name = name

        # TODO(bassosimone): this field is only used by the view. We could consider
        # deprecating it and moving the view mapping logic inside the view itself, which
        # would work as intended as long as we have a working __hash__. By doing this,
        # we would probably reduce the churn and coupling between the computational
        # model and the related view.
        self.group = group


class ProbabilisticConstraint(Constraint):
    """
    ProbabilisticConstraint class.

    This class is used to define probabilistic constraints for the model.
    """

    def __init__(
        self,
        name: str,
        usage: graph.Node,
        capacity: CumulativeDistribution,
        group: str | None = None,
    ) -> None:
        # TODO(bassosimone): consider passing maybe directly the index here? I am a bit
        # suprised the type checker likes this code since the .value of an Index could
        # be many things and I am not sure any of them is a cumulative distribution. In
        # other words, I am missing something here but don't know what it is.
        super().__init__(usage, capacity, group, name)

    def __repr__(self) -> str:
        return f"ProbabilisticConstraint({self.name})"


class DeterministicConstraint(Constraint):
    """
    DeterministicConstraint class.

    This class is used to define deterministic constraints for the model.
    """

    def __init__(
        self,
        name: str,
        usage: graph.Node,
        capacity: graph.Node,
        group: str | None = None,
    ) -> None:
        super().__init__(usage, capacity, group, name)

    def __repr__(self) -> str:
        return f"DeterministicConstraint({self.name})"
