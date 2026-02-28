"""
Context variables for the overtourism vertical.

A context variable is a placeholder index that is not directly controlled by
the model but influences its behaviour.  In general, context variables are
sampled from a distribution, either categorical or continuous.

Context variables extend :class:`~dt_model.symbols.index.Index` so that they
participate in the standard abstract-index detection logic of
:class:`~dt_model.model.model.Model`.
"""

from __future__ import annotations

import random
from abc import abstractmethod
from typing import Any

from scipy.stats import rv_continuous

from civic_digital_twins.dt_model.model.index import Index


class ContextVariable(Index):
    """Placeholder index with a sampling method.

    Subclasses implement :meth:`sample` and :meth:`support_size` according
    to the distribution they represent.  The underlying graph node is always
    a placeholder (``value=None``).
    """

    def __init__(self, name: str) -> None:
        super().__init__(name, None)

    @abstractmethod
    def support_size(self) -> int:
        """Return the size of the support of the context variable."""
        ...

    @abstractmethod
    def sample(
        self,
        nr: int = 1,
        *,
        subset: list | None = None,
        force_sample: bool = False,
    ) -> list:
        """Return a list of ``(probability, value)`` tuples.

        If the distribution is discrete, or if a subset is provided, the
        whole support/subset may be returned when its size does not exceed
        *nr*.  Set *force_sample* to ``True`` to override this and always
        draw *nr* random samples.

        Parameters
        ----------
        nr:
            Number of values to sample.
        subset:
            Values to sample from.  The full support is used when ``None``.
        force_sample:
            Force random sampling even when the support/subset is smaller
            than *nr*.
        """
        ...


class UniformCategoricalContextVariable(ContextVariable):
    """Categorical context variable with uniform probability mass.

    All values returned by :meth:`sample` carry the same probability,
    even when the entire support is returned.
    """

    def __init__(self, name: str, values: list) -> None:
        super().__init__(name)
        self.values = values
        self.size = len(self.values)

    def support_size(self) -> int:
        """Return the size of the support."""
        return self.size

    def sample(
        self,
        nr: int = 1,
        *,
        subset: list | None = None,
        force_sample: bool = False,
    ) -> list[tuple[float, Any]]:
        """Sample values from the support."""
        (values, size) = (self.values, self.size) if subset is None else (subset, len(subset))

        if force_sample or nr < size:
            assert nr > 0
            return [(1 / nr, r) for r in random.choices(values, k=nr)]

        return [(1 / size, v) for v in values]


class CategoricalContextVariable(ContextVariable):
    """Categorical context variable with an explicit probability distribution."""

    def __init__(self, name: str, distribution: dict[Any, float]) -> None:
        super().__init__(name)
        self.distribution = distribution
        self.values = list(self.distribution.keys())
        self.size = len(self.values)

    def support_size(self) -> int:
        """Return the size of the support."""
        return self.size

    def sample(
        self,
        nr: int = 1,
        *,
        subset: list | None = None,
        force_sample: bool = False,
    ) -> list[tuple[float, Any]]:
        """Return a sample from the categorical context variable."""
        (values, size) = (self.values, self.size) if subset is None else (subset, len(subset))

        if force_sample or nr < size:
            assert nr > 0
            return [
                (1 / nr, r)
                for r in random.choices(values, k=nr, weights=[self.distribution[v] for v in values])
            ]

        if subset is None:
            return [(self.distribution[v], v) for v in values]

        subset_probability = [self.distribution[v] for v in values]
        subset_probability_sum = sum(subset_probability)
        return [(p / subset_probability_sum, v) for (p, v) in zip(subset_probability, subset)]


class ContinuousContextVariable(ContextVariable):
    """Continuous context variable backed by a scipy continuous distribution."""

    def __init__(self, name: str, rvc: rv_continuous) -> None:
        super().__init__(name)
        self.rvc = rvc

    def support_size(self) -> int:
        """Return the size of the support (``-1`` for continuous distributions)."""
        return -1

    def sample(
        self,
        nr: int = 1,
        *,
        subset: list | None = None,
        force_sample: bool = False,
    ) -> list:
        """Sample from the continuous context variable."""
        if force_sample or subset is None or nr < len(subset):
            assert nr > 0
            return [(1 / nr, r) for r in list(self.rvc.rvs(size=nr))]

        subset_probability = list(self.rvc.pdf(subset))
        subset_probability_sum = sum(subset_probability)
        return [(p / subset_probability_sum, v) for (p, v) in zip(subset_probability, subset)]
