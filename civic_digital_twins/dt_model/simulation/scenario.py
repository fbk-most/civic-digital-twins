# SPDX-License-Identifier: Apache-2.0
"""Scenario: a concrete instantiation of a Model for evaluation."""

from __future__ import annotations

import numpy as np

from ..engine.frontend import graph
from ..model.index import (
    CategoricalIndex,
    ConditionalCategoricalIndex,
    ConditionalDistributionIndex,
    ConstIndex,
    ConstTimeseriesIndex,
    Distribution,
    DistributionIndex,
    DomainValue,
    GenericIndex,
    Index,
    TimeseriesIndex,
)
from ..model.model import Model
from ..model.model_variant import ModelVariant

__all__ = ["Scenario"]


class Scenario:
    """Concrete instantiation of a :class:`~model.model.Model` for evaluation.

    A ``Scenario`` wraps a model and optionally carries a set of
    *value overrides* that shadow the model's own index values at
    evaluation time.  It is the canonical first argument to
    :class:`~simulation.evaluation.Evaluation` and all ensemble classes.

    The *canonical chain* for evaluation is::

        model → Scenario(model, overrides={…}) → Evaluation(scenario).evaluate(ensemble=ens)

    Parameters
    ----------
    model:
        The model (or variant) to evaluate.
    overrides:
        Optional mapping from index to a :data:`~model.index.DomainValue`.
        Overrides shadow the index's own ``value`` when :meth:`base_substitutions`
        is called.  See the override compatibility table for which value types are
        accepted for each index kind.

    Override compatibility
    ----------------------
    The table below shows which override types are accepted for each index
    kind and how each Scenario method handles them.

    .. code-block:: text

        Index type                   │ float  str  ndarray  Distribution  dict[str,float]
        ─────────────────────────────┼───────────────────────────────────────────────────
        Index                        │  ✓     ✗      ✗         ✗              ✗
        TimeseriesIndex              │  ✗     ✗      ✓(1-D)    ✗              ✗
        ConstIndex / ConstTimeseries │  ✗     ✗      ✗         ✗              ✗
        DistributionIndex            │  ✗     ✗      ✗         ✓              ✗
        ConditionalDistributionIndex │  ✗     ✗      ✗         ✗              ✗
        CategoricalIndex             │  ✗     ✓*     ✗         ✗              ✓**
        ConditionalCategoricalIndex  │  ✗     ✓*     ✗         ✗              ✗

        * str must be in idx.support
        ** dict keys must be a non-empty subset of idx.support, positive probs summing to 1.0

        Method behaviour for accepted overrides:

        Index type + override           │ abstract_indexes  base_subs      effective_distribution  effective_outcomes
        ────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────
        Index(scalar), no override      │ absent            own scalar     —                       —
        Index(scalar), float override   │ absent            override float —                       —
        Index(None), no override        │ present           —              —                       —
        Index(None), float override     │ absent            override float —                       —
        TimeseriesIndex(arr), no ovr    │ absent            own array      —                       —
        TimeseriesIndex(arr), ndarray   │ absent            override array —                       —
        TimeseriesIndex(None), no ovr   │ present           —              —                       —
        TimeseriesIndex(None), ndarray  │ absent            override array —                       —
        DistributionIndex, no override  │ present           —              _frozen                 —
        DistributionIndex, Distribution │ present           —              override dist           —
        CondDistribution, no override   │ present           —              —                       —
        CategoricalIndex, no override   │ present           —              —                       idx.outcomes
        CategoricalIndex, str           │ absent            asarray(str)   —                       {str: 1.0}
        CategoricalIndex, dict          │ present           —              —                       override dict
        CondCategorical, no override    │ present           —              —                       None
        CondCategorical, str            │ absent            asarray(str)   —                       {str: 1.0}

    Examples
    --------
    >>> from civic_digital_twins.dt_model import Index, Model, Scenario
    >>> cost = Index("cost", 8.0)
    >>> model = Model("parking", indexes=[cost])
    >>> # Base scenario — uses model's own value
    >>> base = Scenario(model)
    >>> # What-if scenario — parking costs €12
    >>> expensive = Scenario(model, overrides={cost: 12.0})
    """

    def __init__(
        self,
        model: Model | ModelVariant,
        overrides: dict[GenericIndex, DomainValue] | None = None,
    ) -> None:
        self._model = model
        self._overrides: dict[GenericIndex, DomainValue] = dict(overrides or {})
        for idx, val in self._overrides.items():
            # Structural constants cannot be overridden.
            if isinstance(idx, (ConstIndex, ConstTimeseriesIndex)):
                raise TypeError(
                    f"Index {idx.name!r} is a structural constant and cannot be overridden in a Scenario. "
                    f"Use Index / TimeseriesIndex for values that vary between scenarios."
                )

            # TimeseriesIndex: only 1-D ndarray override allowed.
            if isinstance(idx, TimeseriesIndex):
                if not isinstance(val, np.ndarray) or val.ndim != 1:
                    shape_info = f" with shape {val.shape}" if isinstance(val, np.ndarray) else ""
                    raise TypeError(
                        f"Override for TimeseriesIndex {idx.name!r} must be a 1-D ndarray; "
                        f"got {type(val).__name__!r}{shape_info}."
                    )
                continue

            # DistributionIndex: only Distribution override allowed.
            if isinstance(idx, DistributionIndex):
                if not isinstance(val, Distribution):
                    raise TypeError(
                        f"Index {idx.name!r} is distribution-backed; override must be a Distribution, "
                        f"not {type(val).__name__!r}."
                    )
                continue

            # ConditionalDistributionIndex: overrides not supported — the conditional
            # mapping lives in the factory and cannot be replaced by a simple value.
            if isinstance(idx, ConditionalDistributionIndex):
                raise TypeError(
                    f"ConditionalDistributionIndex {idx.name!r} does not support overrides in Scenario. "
                    f"To change the conditional mapping, supply a model variant with a different factory."
                )

            # CategoricalIndex: str (concrete pin) or dict[str, float] (new weights),
            # both must preserve the declared support.
            if isinstance(idx, CategoricalIndex):
                if isinstance(val, str):
                    if val not in idx.support:
                        raise ValueError(
                            f"Override {val!r} is not in the support of CategoricalIndex {idx.name!r}: {idx.support!r}."
                        )
                elif isinstance(val, dict):
                    keys, support = set(val.keys()), set(idx.support)
                    if not keys:
                        raise ValueError(f"Override dict for CategoricalIndex {idx.name!r} must not be empty.")
                    extra = keys - support
                    if extra:
                        raise ValueError(
                            f"Override dict for CategoricalIndex {idx.name!r} contains keys outside its support "
                            f"{sorted(support)!r}: {sorted(extra)!r}."
                        )
                    non_positive = [k for k, p in val.items() if p <= 0]
                    if non_positive:
                        raise ValueError(
                            f"Override dict for CategoricalIndex {idx.name!r}: all probabilities must be strictly "
                            f"positive; non-positive keys: {non_positive}."
                        )
                    if not np.isclose(sum(val.values()), 1.0):
                        raise ValueError(
                            f"Override dict for CategoricalIndex {idx.name!r}: probabilities must sum to 1.0; "
                            f"got {sum(val.values())}."
                        )
                else:
                    raise TypeError(
                        f"Override for CategoricalIndex {idx.name!r} must be a str (concrete outcome) or "
                        f"dict[str, float] (new probabilities); got {type(val).__name__!r}."
                    )
                continue

            # ConditionalCategoricalIndex: only str (concrete pin) is allowed.
            if isinstance(idx, ConditionalCategoricalIndex):
                if not isinstance(val, str):
                    raise TypeError(
                        f"Override for ConditionalCategoricalIndex {idx.name!r} must be a str (concrete outcome); "
                        f"got {type(val).__name__!r}."
                    )
                if val not in idx.support:
                    raise ValueError(
                        f"Override {val!r} is not in the support of "
                        f"ConditionalCategoricalIndex {idx.name!r}: {idx.support!r}."
                    )
                continue

            # Plain scalar Index: only scalar override allowed (no Distribution, no array).
            if isinstance(idx, Index):
                if isinstance(val, Distribution):
                    raise TypeError(
                        f"Scalar Index {idx.name!r} cannot be overridden with a Distribution; "
                        f"use DistributionIndex for distribution-backed indexes."
                    )
                if isinstance(val, np.ndarray) and val.ndim > 0:
                    raise TypeError(
                        f"Override for scalar Index {idx.name!r} must be a scalar; got ndarray with shape {val.shape}."
                    )

    @property
    def model(self) -> Model | ModelVariant:
        """The wrapped model."""
        return self._model

    @property
    def overrides(self) -> dict[GenericIndex, DomainValue]:
        """A shallow copy of the active override mapping.

        Returns a new dict so that callers cannot mutate the internal state.
        Each key is a :class:`~model.index.GenericIndex`; each value is the
        :data:`~model.index.DomainValue` that shadows the index's own value
        at evaluation time.

        Returns
        -------
        dict[GenericIndex, DomainValue]
            Copy of the overrides passed at construction, or an empty dict
            when no overrides were given.
        """
        return dict(self._overrides)

    def effective_distribution(self, idx: GenericIndex) -> Distribution | None:
        """Return the effective :class:`~model.index.Distribution` for *idx*.

        Checks ``overrides`` first; falls back to ``idx.value`` if that is a
        :class:`~model.index.Distribution`.  Returns ``None`` when no
        distribution is configured.

        Parameters
        ----------
        idx:
            The index to look up.

        Returns
        -------
        Distribution or None
            The effective distribution, or ``None`` if *idx* is not
            distribution-backed (in either the model or any active override).
        """
        val = self._overrides.get(idx)
        if val is not None:
            return val if isinstance(val, Distribution) else None
        if isinstance(idx, DistributionIndex):
            return idx.value
        return None

    def effective_outcomes(self, idx: CategoricalIndex | ConditionalCategoricalIndex) -> dict[str, float] | None:
        """Return the effective outcome probabilities for *idx*.

        Checks ``overrides`` first.  Return value depends on index type and override:

        * :class:`~model.index.CategoricalIndex`, no override → ``idx.outcomes``.
        * :class:`~model.index.CategoricalIndex`, ``dict`` override → the override dict.
        * Either type, ``str`` override (concrete pin) → singleton ``{val: 1.0}``.
        * :class:`~model.index.ConditionalCategoricalIndex`, no override → ``None``
          (no unconditional outcomes dict; the ensemble uses ``idx.outcomes_for(...)``).

        Parameters
        ----------
        idx:
            A :class:`~model.index.CategoricalIndex` or
            :class:`~model.index.ConditionalCategoricalIndex` to look up.

        Returns
        -------
        dict[str, float] or None
            The outcome probability mapping, or ``None`` when *idx* is a
            :class:`~model.index.ConditionalCategoricalIndex` with no concrete pin.
        """
        val = self._overrides.get(idx)
        if isinstance(val, str):
            return {val: 1.0}
        if isinstance(val, dict):
            return val
        if isinstance(idx, CategoricalIndex):
            return idx.outcomes
        return None  # ConditionalCategoricalIndex with no override: no simple outcomes dict

    def abstract_indexes(self) -> list[GenericIndex]:
        """Return indexes that are abstract from this scenario's perspective.

        An index is scenario-abstract if the ensemble must sample it:

        * A model-abstract index (``None``-valued or distribution-backed) that
          is **not** concretely overridden by this scenario.
        * A :class:`~model.index.DistributionIndex` overridden with a **different**
          :class:`~model.index.Distribution` — ``effective_distribution`` returns
          the override distribution.
        * A :class:`~model.index.CategoricalIndex` overridden with a
          ``dict[str, float]`` (new probability weights) — ``effective_outcomes``
          returns the override probabilities.

        Indexes whose effective value is a concrete scalar or string (either because
        the model defines them that way, or because the scenario overrides them) are
        excluded: they are handled by :meth:`base_substitutions`.

        Returns
        -------
        list[GenericIndex]
            Indexes that need to be sampled by an ensemble.
        """
        result: list[GenericIndex] = []

        for idx in self._model.abstract_indexes():
            override = self._overrides.get(idx)
            if override is not None and not isinstance(override, (Distribution, dict)):
                # Concretely overridden (scalar or str) — handled by base_substitutions; skip.
                continue
            result.append(idx)

        return result

    def base_substitutions(self) -> dict[graph.Node, np.ndarray]:
        """Collect concrete values for injection into the executor state.

        Iterates over all model indexes, applies overrides, and returns a
        mapping from each index's *graph node* to a ``numpy`` array ready for
        injection.  Only indexes whose effective value is a concrete
        ``float`` or ``np.ndarray`` (not ``None``, not a
        :class:`~model.index.Distribution`) are included.  Constant-node
        indexes (:class:`~model.index.ConstIndex`,
        :class:`~model.index.ConstTimeseriesIndex`) are skipped because
        their values are already baked into the graph.

        The returned dict is intended to be merged into the executor
        :class:`~engine.numpybackend.executor.State` as the *initial* state
        before running the plan, but **must not** be added to the
        ``substituted_nodes`` set — the shape-normalisation pass treats these
        injected values as "untouched" scalars/timeseries, prepending the
        correct number of leading singleton dimensions automatically.

        Returns
        -------
        dict[graph.Node, np.ndarray]
            Maps each placeholder graph node to its concrete value array.
        """
        subs: dict[graph.Node, np.ndarray] = {}
        for idx in self._model.indexes:
            if isinstance(idx, (ConstIndex, ConstTimeseriesIndex)):
                continue  # value already baked into the graph as a constant node

            # Determine the effective value: override takes precedence.
            val: DomainValue | graph.Node | None = self._overrides.get(idx)
            if val is None and isinstance(idx, (Index, TimeseriesIndex)):
                val = idx.value  # type: ignore[assignment]  # Scalar ⊄ DomainValue

            if val is None or isinstance(val, (Distribution, dict, graph.Node)):
                continue  # no concrete value or formula node; ensemble's responsibility

            subs[idx.node] = np.asarray(val)
        return subs
