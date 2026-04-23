"""Overtourism metamodel — generic classes for overtourism digital twins.

This module provides the building blocks for an overtourism model:

* :class:`PresenceVariable` — placeholder index representing visitor presence,
  sampled from a context-dependent distribution (e.g. uniform or truncnorm).
* :class:`Constraint` — named pairing of a usage formula index and a capacity
  index.
* :class:`OvertourismModel` — :class:`~dt_model.model.model.Model` subclass
  with overtourism domain structure (CVs, PVs, usage indexes, capacity
  indexes, constraints).
* :class:`OvertourismEnsemble` — iterable that yields weighted scenarios by
  enumerating CV combinations and pre-sampling distribution-backed indexes.

Context variables are now plain
:class:`~civic_digital_twins.dt_model.CategoricalIndex` instances: the
dedicated ``ContextVariable`` hierarchy that previously lived here has been
removed, since ``CategoricalIndex`` already provides the same
string-keyed finite-support sampling semantics with a declarative API.
"""

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import itertools
import random
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from civic_digital_twins.dt_model import (
    ENSEMBLE,
    Axis,
    CategoricalIndex,
    Distribution,
    GenericIndex,
    Index,
    Model,
)

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
        Context variables (``CategoricalIndex`` objects) that influence the
        presence distribution.
    distribution:
        Callable that accepts the CV values and returns a frozen scipy
        distribution (e.g. ``scipy.stats.truncnorm``).
    """

    def __init__(
        self,
        name: str,
        cvs: list[CategoricalIndex],
        distribution: Callable[..., Any] | None = None,
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
        distr = self.distribution(*all_cvs)
        return np.asarray(distr.rvs(size=nr))


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
    usage: Index  # formula-mode Index wrapping the usage expression
    capacity: Index  # constant, distribution-backed, or formula-mode Index


# ---------------------------------------------------------------------------
# OvertourismModel
# ---------------------------------------------------------------------------


class OvertourismModel(Model):
    """A :class:`~dt_model.model.model.Model` with overtourism domain structure.

    Wraps the core :class:`~dt_model.model.model.Model` with named lists for
    the overtourism-specific categories (CVs, PVs, capacities, domain
    indexes, constraints).  Internally uses the dataclass-based
    ``Inputs`` / ``Outputs`` API so all abstract indexes are declared in the
    inputs contract: no :class:`~dt_model.model.model.ModelContractWarning`
    is emitted at construction time.

    Parameters
    ----------
    name:
        Human-readable name for the model.
    cvs:
        Context variables (sampled externally by the ensemble).  These are
        ``CategoricalIndex`` objects.
    pvs:
        Presence variables (grid axes in evaluation).
    indexes:
        Plain model indexes (conversion factors, usage factors, …).
    capacities:
        Capacity indexes for each constraint.
    constraints:
        Named usage/capacity constraint pairs.
    """

    @dataclass
    class Inputs:
        """Contractual inputs of :class:`OvertourismModel`.

        The field order matches the legacy
        ``cvs + pvs + indexes + capacities`` flat-list ordering so that the
        :class:`OvertourismEnsemble` RNG draw order is preserved for
        downstream reproducibility.
        """

        cvs: list[CategoricalIndex]
        pvs: list[PresenceVariable]
        domain_indexes: list[GenericIndex]
        capacities: list[GenericIndex]

    @dataclass
    class Outputs:
        """Contractual outputs of :class:`OvertourismModel`."""

        usage_indexes: list[GenericIndex]

    def __init__(
        self,
        name: str,
        *,
        cvs: list[CategoricalIndex],
        pvs: list[PresenceVariable],
        indexes: list[GenericIndex],
        capacities: list[GenericIndex],
        constraints: list[Constraint],
    ) -> None:
        Inputs = OvertourismModel.Inputs
        Outputs = OvertourismModel.Outputs

        super().__init__(
            name,
            inputs=Inputs(
                cvs=list(cvs),
                pvs=list(pvs),
                domain_indexes=list(indexes),
                capacities=list(capacities),
            ),
            outputs=Outputs(usage_indexes=[c.usage for c in constraints]),
        )

        self.cvs = cvs
        self.pvs = pvs
        self.domain_indexes = indexes
        self.capacities = capacities
        self.constraints = constraints


# ---------------------------------------------------------------------------
# OvertourismEnsemble
# ---------------------------------------------------------------------------


def _sample_categorical(
    cv: CategoricalIndex,
    nr: int,
    subset: list[str] | None = None,
) -> list[tuple[float, str]]:
    """Return a list of ``(probability, value)`` tuples for *cv*.

    If the support (or subset) size is at most *nr* the full support is
    enumerated; otherwise *nr* Monte-Carlo samples are drawn, each carrying
    uniform probability ``1/nr`` in the resulting ensemble.

    When *subset* is passed, the per-value probabilities are renormalised
    over the subset so that they sum to 1.0.
    """
    outcomes = cv.outcomes
    values = cv.support if subset is None else list(subset)
    size = len(values)

    if nr < size:
        assert nr > 0
        weights = [outcomes[v] for v in values]
        choices = random.choices(values, k=nr, weights=weights)
        return [(1.0 / nr, r) for r in choices]

    probs = [outcomes[v] for v in values]
    total = sum(probs)
    return [(p / total, v) for (p, v) in zip(probs, values)]


class OvertourismEnsemble:
    """Batched ensemble for an :class:`OvertourismModel`.

    Implements :class:`~dt_model.simulation.ensemble.AxisEnsemble`.

    Enumerates all combinations of CV values and pre-samples
    distribution-backed indexes, materialising the results into a single
    batched ENSEMBLE axis.  Presence variables are *not* included — they
    are provided as PARAMETER axes to
    :meth:`~dt_model.simulation.evaluation.Evaluation.evaluate`.

    For each :class:`~civic_digital_twins.dt_model.CategoricalIndex` CV:

    * if the scenario selects a single value, that value is used verbatim
      with probability 1.0;
    * if the scenario selects a subset of values whose size is at most
      *cv_ensemble_size*, the subset is enumerated with probabilities
      renormalised over it;
    * if the support (or subset) size exceeds *cv_ensemble_size*, the CV is
      Monte-Carlo sampled *cv_ensemble_size* times with uniform per-sample
      probability.

    Parameters
    ----------
    model:
        The overtourism model whose CVs are to be sampled.
    scenario:
        Maps a CV to a list of specific values to use instead of sampling
        from its full support.
    cv_ensemble_size:
        Number of samples per CV when sampling from the full support (or
        from a subset larger than this threshold).
    """

    def __init__(
        self,
        model: OvertourismModel,
        scenario: dict[CategoricalIndex, list[str]],
        cv_ensemble_size: int = 20,
    ) -> None:
        self.model = model

        # Per-CV list of (probability, value) pairs.
        cv_samples: dict[CategoricalIndex, list[tuple[float, str]]] = {}
        for cv in model.cvs:
            if cv in scenario:
                subset = scenario[cv]
                if len(subset) == 1:
                    cv_samples[cv] = [(1.0, subset[0])]
                else:
                    cv_samples[cv] = _sample_categorical(cv, cv_ensemble_size, subset=subset)
            else:
                cv_samples[cv] = _sample_categorical(cv, cv_ensemble_size)

        # Total number of scenarios = product of all per-CV sample sizes.
        S = 1
        for samples in cv_samples.values():
            S *= len(samples)
        self.size = S

        # Distribution-backed non-PV non-CV abstract indexes (e.g. capacities
        # and distribution-backed conversion factors).
        # NOTE: use identity comparison (id()) to exclude PVs and CVs because
        # GenericIndex.__eq__ returns a graph.Node (always truthy), making
        # the list `in` operator unreliable for these objects.
        cv_pv_ids = {id(cv) for cv in model.cvs} | {id(pv) for pv in model.pvs}
        dist_indexes: list[Index] = [
            idx  # type: ignore[misc]
            for idx in model.abstract_indexes()
            if id(idx) not in cv_pv_ids and isinstance(idx, Index) and isinstance(idx.value, Distribution)
        ]

        # Materialise all scenario combinations into batched arrays.
        cvs = list(cv_samples.keys())
        cv_sample_lists = [cv_samples[cv] for cv in cvs]

        weights = np.ones(S)
        cv_batched: dict[CategoricalIndex, list[str]] = {cv: [] for cv in cvs}
        for i, combo in enumerate(itertools.product(*cv_sample_lists)):
            for cv, (prob, val) in zip(cvs, combo):
                weights[i] *= prob
                cv_batched[cv].append(val)

        self._axis = Axis("_overtourism", ENSEMBLE)
        self._weights = weights
        self._assignments: dict[GenericIndex, np.ndarray] = {cv: np.asarray(cv_batched[cv]) for cv in cvs}
        # Pre-sample S values for each distribution-backed index.
        for idx in dist_indexes:
            self._assignments[idx] = idx.value.rvs(size=S)  # type: ignore[union-attr]

    @property
    def ensemble_axes(self) -> tuple[Axis, ...]:
        """Return the single ENSEMBLE axis for this ensemble."""
        return (self._axis,)

    @property
    def ensemble_weights(self) -> tuple[np.ndarray, ...]:
        """Return the weight array for the single ENSEMBLE axis."""
        return (self._weights,)

    def assignments(self) -> dict[GenericIndex, np.ndarray]:
        """Return batched assignments for all CVs and distribution-backed indexes."""
        return self._assignments

    def __len__(self) -> int:
        """Return the total number of scenarios."""
        return self.size
