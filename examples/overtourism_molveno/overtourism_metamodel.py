"""Overtourism metamodel — generic classes for overtourism digital twins.

This module provides the building blocks for an overtourism model:

* :class:`ContextVariable` and its subclasses — categorical or continuous
  variables that influence the model but are not directly controlled
  (weekday, season, weather, …).
* :class:`PresenceVariable` — placeholder index representing visitor presence,
  sampled from a context-dependent truncated-normal distribution.
* :class:`Constraint` — named pairing of a usage formula index and a capacity
  index.
* :class:`OvertourismModel` — :class:`~dt_model.model.model.Model` subclass
  with overtourism domain structure (CVs, PVs, usage indexes, capacity
  indexes, constraints).
* :class:`OvertourismEnsemble` — iterable that yields weighted scenarios by
  enumerating CV combinations and pre-sampling distribution-backed indexes.
"""

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import itertools
import random
from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
from scipy import stats
from scipy.stats import rv_continuous

from civic_digital_twins.dt_model.model.index import Distribution, GenericIndex, Index
from civic_digital_twins.dt_model.model.model import Model
from civic_digital_twins.dt_model.simulation.ensemble import WeightedScenario


# ---------------------------------------------------------------------------
# Context variables
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Presence variable
# ---------------------------------------------------------------------------


class PresenceVariable(Index):
    """Placeholder index with a presence-distribution sampler.

    Parameters
    ----------
    name:
        Name of the presence variable.
    cvs:
        Context variables that influence the presence distribution.
    distribution:
        Callable that accepts the CV values and returns a dict with
        ``"mean"`` and ``"std"`` keys used to parameterise a truncated
        normal distribution.
    """

    def __init__(
        self,
        name: str,
        cvs: list[ContextVariable],
        distribution: Callable | None = None,
    ) -> None:
        super().__init__(name, None)
        self.cvs = cvs
        self.distribution = distribution

    def sample(self, cvs: dict | None = None, nr: int = 1) -> np.ndarray:
        """Return values sampled from the presence distribution.

        Parameters
        ----------
        cvs:
            Mapping from context variable to its current value.
        nr:
            Number of samples to draw.

        Returns
        -------
        np.ndarray
            Array of sampled presence values.
        """
        assert nr > 0

        all_cvs = []
        if cvs is not None:
            all_cvs = [cvs[cv] for cv in self.cvs if cv in cvs.keys()]
        assert self.distribution is not None
        distr: dict = self.distribution(*all_cvs)
        return np.asarray(
            stats.truncnorm.rvs(
                -distr["mean"] / distr["std"],
                10,
                loc=distr["mean"],
                scale=distr["std"],
                size=nr,
            ),
        )


# ---------------------------------------------------------------------------
# Constraint
# ---------------------------------------------------------------------------


@dataclass(eq=False)
class Constraint:
    """Named pairing of a usage formula index and a capacity index.

    Both *usage* and *capacity* are formula-mode or distribution-backed
    :class:`~dt_model.model.index.Index` objects, so the entire constraint is
    expressed in terms of :class:`~dt_model.model.index.GenericIndex` — no
    engine-layer types appear in the public API.

    Identity-based hashing (``eq=False``) keeps ``Constraint`` objects usable
    as dict keys, matching the convention used by ``graph.Node`` and
    ``GenericIndex``.
    """

    name: str
    usage: Index      # formula-mode Index wrapping the usage expression
    capacity: Index   # constant, distribution-backed, or formula-mode Index


# ---------------------------------------------------------------------------
# OvertourismModel
# ---------------------------------------------------------------------------


class OvertourismModel(Model):
    """A :class:`~dt_model.model.model.Model` with overtourism domain structure.

    Extends the core ``Model`` by labelling subsets of indexes as context
    variables, presence variables, and capacity indexes, and by attaching
    named constraints.  All items in *cvs*, *pvs*, *indexes*, *capacities*,
    and the usage/capacity indexes of *constraints* are merged into the flat
    ``Model.indexes`` list automatically.

    Parameters
    ----------
    name:
        Human-readable name for the model.
    cvs:
        Context variables (sampled externally by the ensemble).
    pvs:
        Presence variables (grid axes in evaluation).
    indexes:
        Plain model indexes (conversion factors, usage factors, etc.).
    capacities:
        Capacity indexes for each constraint.
    constraints:
        Named usage/capacity constraint pairs.
    """

    def __init__(
        self,
        name: str,
        cvs: list[ContextVariable],
        pvs: list[PresenceVariable],
        indexes: list[GenericIndex],
        capacities: list[GenericIndex],
        constraints: list[Constraint],
    ) -> None:
        # Collect all indexes into the flat Model list; CVs and PVs are
        # abstract (placeholder) indexes and are included so that
        # abstract_indexes() finds them automatically.  Usage indexes from
        # constraints are formula-mode and are included so that
        # nodes_of_interest=None in Evaluation.evaluate covers them.
        usage_indexes: list[GenericIndex] = [c.usage for c in constraints]
        all_indexes: list[GenericIndex] = list(cvs) + list(pvs) + list(indexes) + list(capacities) + usage_indexes
        super().__init__(name, all_indexes)

        self.cvs = cvs
        self.pvs = pvs
        self.domain_indexes = indexes
        self.capacities = capacities
        self.constraints = constraints


# ---------------------------------------------------------------------------
# OvertourismEnsemble
# ---------------------------------------------------------------------------


class OvertourismEnsemble:
    """Iterable that yields weighted scenarios for an :class:`OvertourismModel`.

    Each iteration produces a :data:`~dt_model.simulation.evaluation.WeightedScenario`
    assigning a concrete value to every context variable and every
    distribution-backed (non-PV, non-CV) abstract index in the model.
    Presence variables are *not* included — they are provided as grid axes
    to :meth:`~dt_model.simulation.evaluation.Evaluation.evaluate`.

    Parameters
    ----------
    model:
        The overtourism model whose CVs are to be sampled.
    scenario:
        Optional override: maps a CV to a list of specific values to use
        instead of sampling from its full support.
    cv_ensemble_size:
        Number of samples per CV when sampling from the full support.
    """

    def __init__(
        self,
        model: OvertourismModel,
        scenario: dict[ContextVariable, list],
        cv_ensemble_size: int = 20,
    ) -> None:
        self.model = model

        # Build per-CV list of (prob, value) pairs.
        self._cv_samples: dict[ContextVariable, list] = {}
        for cv in model.cvs:
            if cv in scenario:
                subset = scenario[cv]
                if len(subset) == 1:
                    self._cv_samples[cv] = [(1.0, subset[0])]
                else:
                    self._cv_samples[cv] = cv.sample(cv_ensemble_size, subset=subset)
            else:
                self._cv_samples[cv] = cv.sample(cv_ensemble_size)

        # Total number of scenarios = product of all per-CV sample sizes.
        self.size = 1
        for samples in self._cv_samples.values():
            self.size *= len(samples)

        # Distribution-backed non-PV non-CV abstract indexes (e.g. capacities
        # and distribution-backed conversion factors).
        # NOTE: use identity comparison (id()) to exclude PVs because
        # GenericIndex.__eq__ returns a graph.Node (always truthy), making
        # the list `in` operator unreliable for these objects.
        pv_ids = {id(pv) for pv in model.pvs}
        self._dist_indexes: list[Index] = [
            idx  # type: ignore[misc]
            for idx in model.abstract_indexes()
            if not isinstance(idx, ContextVariable)
            and id(idx) not in pv_ids
            and isinstance(idx, Index)
            and isinstance(idx.value, Distribution)
        ]

        # Pre-sample S values for each distribution-backed index.
        S = max(self.size, 1)
        self._dist_samples: dict[Index, object] = {
            idx: idx.value.rvs(size=S)  # type: ignore[union-attr]
            for idx in self._dist_indexes
        }

    def __iter__(self):
        """Iterate over all CV combinations, yielding one WeightedScenario each."""
        cvs = list(self._cv_samples.keys())
        cv_sample_lists = [self._cv_samples[cv] for cv in cvs]

        for i, combo in enumerate(itertools.product(*cv_sample_lists)):
            weight = 1.0
            assignments: dict = {}

            for cv, (prob, val) in zip(cvs, combo):
                weight *= prob
                assignments[cv] = val

            # Include pre-sampled values for distribution-backed indexes.
            for idx in self._dist_indexes:
                assignments[idx] = self._dist_samples[idx][i]  # type: ignore[index]

            yield weight, assignments

    def __len__(self) -> int:
        """Return the total number of scenarios."""
        return self.size
