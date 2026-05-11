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
        Optional mapping from index to a concrete :data:`~model.index.DomainValue`.
        Overrides shadow the index's own ``value`` when :meth:`base_substitutions`
        is called.  A ``Distribution`` override replaces the index's distribution
        for ensemble sampling purposes (see :meth:`effective_distribution`).

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

            # Distribution-backed indexes: only Distribution override allowed.
            if isinstance(idx, (DistributionIndex, ConditionalDistributionIndex)):
                if not isinstance(val, Distribution):
                    raise TypeError(
                        f"Index {idx.name!r} is distribution-backed; override must be a Distribution, "
                        f"not {type(val).__name__!r}."
                    )
                continue

            # Categorical indexes: value overrides are not supported.
            if isinstance(idx, (CategoricalIndex, ConditionalCategoricalIndex)):
                raise TypeError(
                    f"CategoricalIndex {idx.name!r} does not support value overrides in Scenario."
                )

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

    def abstract_indexes(self) -> list[GenericIndex]:
        """Return indexes that are abstract from this scenario's perspective.

        An index is scenario-abstract if the ensemble must sample it:

        * A model-abstract index (``None``-valued or distribution-backed) that
          is **not** concretely overridden by this scenario.
        * A :class:`~model.index.DistributionIndex` overridden with a **different**
          Distribution — ``effective_distribution`` returns the override distribution.

        Indexes whose effective value is a concrete scalar/array (either because the
        model defines them that way, or because the scenario overrides them with a
        concrete value) are excluded: they are handled by :meth:`base_substitutions`.

        Returns
        -------
        list[GenericIndex]
            Indexes that need to be sampled by an ensemble.
        """
        result: list[GenericIndex] = []

        for idx in self._model.abstract_indexes():
            override = self._overrides.get(idx)
            if override is not None and not isinstance(override, Distribution):
                # Concretely overridden — handled by base_substitutions; skip.
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

            if val is None or isinstance(val, (Distribution, graph.Node)):
                continue  # no concrete value or formula node; ensemble's responsibility

            subs[idx.node] = np.asarray(val)
        return subs
