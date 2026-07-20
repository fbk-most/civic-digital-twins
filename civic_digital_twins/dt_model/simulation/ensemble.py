"""Ensemble protocol and built-in ensemble implementations."""

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np

from ..axes import ENSEMBLE, Axis
from ..model.index import (
    CategoricalIndex,
    ConditionalCategoricalIndex,
    ConditionalDistributionIndex,
    DistributionIndex,
    GenericIndex,
    Index,
)
from .axis_layout import AxisLayout
from .plan import EvaluationPlan
from .scenario import Scenario


def _grown_axes(axes: tuple[Axis, ...], sizes: tuple[int, ...], axis_name: str, new_size: int) -> tuple[Axis, ...]:
    """Return *axes* with the named axis regenerated at *new_size* (position preserved).

    Thin wrapper around :meth:`~simulation.axis_layout.AxisLayout.with_grown_axis`
    so growing a :class:`FrozenEnsemble` axis goes through the same primitive
    as the merge functions in ``handle.py``, rather than hand-rolling a fresh
    ``Axis(name, ENSEMBLE)``.
    """
    return AxisLayout.build(ensemble=list(zip(axes, sizes))).with_grown_axis(axis_name, new_size).axes


WeightedScenario = tuple[float, dict[GenericIndex, Any]]
"""A weighted scenario maps each abstract index to a concrete value.

The first element is the scenario weight (probability); the second is a
mapping from each abstract index to its concrete value for this scenario.
Together a list of ``WeightedScenario`` objects defines a discrete
probability distribution over instantiations of an abstract model.
"""


@runtime_checkable
class Ensemble(Protocol):
    """Protocol for iterables that yield :data:`WeightedScenario` instances.

    Any object that implements ``__iter__`` returning an iterator over
    ``WeightedScenario`` tuples satisfies this protocol.  This is used as
    a common type for ensemble generators (e.g. domain-specific classes
    that enumerate context-variable combinations with associated weights).
    """

    def __iter__(self) -> Iterator[WeightedScenario]:
        """Yield weighted scenarios."""
        ...  # pragma: no cover


@dataclass(eq=False)
class EnsembleAxisSpec:
    """Specification for one named ENSEMBLE axis in a :class:`PartitionedEnsemble`.

    Parameters
    ----------
    name:
        Lower-case axis name; must be unique within the ensemble.
    indexes:
        Abstract indexes assigned to this axis.  Each index must appear in
        at most one :class:`EnsembleAxisSpec` within a single
        :class:`PartitionedEnsemble`.
    size:
        Number of samples along this axis.
    """

    name: str
    indexes: list[GenericIndex] = field(default_factory=list)
    size: int = 1


@runtime_checkable
class AxisEnsemble(Protocol):
    """Batched ensemble over one or more ENSEMBLE axes (no scenario enumeration).

    This protocol is the canonical ensemble input to
    :meth:`~civic_digital_twins.dt_model.simulation.evaluation.Evaluation.evaluate`.
    :class:`DistributionEnsemble` implements it natively.

    Conventions / invariants
    ------------------------
    - All axes in ``ensemble_axes`` have ``role == "ENSEMBLE"``.
    - ``ensemble_axes`` order defines the canonical ENSEMBLE dimension order.
    - ``ensemble_weights[i]`` is the factorized weight vector for
      ``ensemble_axes[i]``.
    - :meth:`assignments` returns concrete batched arrays for abstract indexes,
      without enumerating scenarios.

    Shape contract (strict)
    -----------------------
    Let ``M = len(ensemble_axes)`` and sizes ``S0..S(M-1)``.

    For each ``(idx, value)`` in :meth:`assignments`:

    .. code-block:: text

        value.shape == (d0, d1, ..., d(M-1), *domain_shape(idx))

    where for each ``j``:

    - ``dj == Sj`` if *idx* is assigned to ``ensemble_axes[j]``
    - ``dj == 1``  otherwise

    and:

    - scalar values: ``domain_shape(idx) == ()``
    - timeseries values: ``domain_shape(idx) == (T,)``  (time is last)

    The ENSEMBLE dims ``(d0..d(M-1))`` are **mandatory** and must be present
    in-order for every assigned index (size 1 where not applicable).  No axis
    may be omitted.  This is the rule that prevents ``S == T`` ambiguities.
    """

    @property
    def ensemble_axes(self) -> tuple[Axis, ...]:
        """Ordered ENSEMBLE axes for this ensemble."""
        ...  # pragma: no cover

    @property
    def ensemble_weights(self) -> tuple[np.ndarray, ...]:
        """Factorized weight vectors aligned with :attr:`ensemble_axes`.

        For each ``i``:

        - ``ensemble_weights[i].ndim == 1``
        - ``ensemble_weights[i].shape == (size_of(ensemble_axes[i]),)``
        - weights sum to 1.0 (recommended invariant)
        """
        ...  # pragma: no cover

    def assignments(self) -> Mapping[GenericIndex, np.ndarray]:
        """Return batched concrete values for abstract indexes (index-keyed)."""
        ...  # pragma: no cover


class FrozenEnsemble:
    """An :class:`AxisEnsemble` backed by pre-computed sample arrays — no RNG draws.

    A frozen ensemble holds the samples already drawn for one or more ENSEMBLE
    axes so the plan can be re-executed over the *same* scenarios (e.g. when
    extending the PARAMETER grid) without advancing any random generator.
    Produced by :meth:`DistributionEnsemble.draw_batch` and similar methods on
    other ensemble types; held by
    :class:`~simulation.handle.EvaluationHandle` as the accumulated sample set.

    Parameters
    ----------
    axes:
        Ordered ENSEMBLE axes (one per partition for multi-axis ensembles).
    weights:
        Factorized weight vectors aligned with *axes*.
    cached_assignments:
        Pre-drawn samples for every abstract index.  Shape contract follows
        :class:`AxisEnsemble`: size ``Sj`` at dimension ``j`` when the index
        is assigned to ``axes[j]``, size 1 otherwise.
    """

    def __init__(
        self,
        axes: tuple[Axis, ...],
        weights: tuple[np.ndarray, ...],
        cached_assignments: dict[GenericIndex, np.ndarray],
    ) -> None:
        self._axes = axes
        self._weights = weights
        self._cached_assignments = cached_assignments

    @property
    def ensemble_axes(self) -> tuple[Axis, ...]:
        """ENSEMBLE axes carrying the frozen samples."""
        return self._axes

    @property
    def ensemble_weights(self) -> tuple[np.ndarray, ...]:
        """Factorized weight vectors for the frozen samples."""
        return self._weights

    def assignments(self) -> Mapping[GenericIndex, np.ndarray]:
        """Return the cached batched samples for every abstract index."""
        return self._cached_assignments

    def draw_batch(self, size: int, rng: np.random.Generator, *, axis: str | None = None) -> FrozenEnsemble:
        """Not supported — :class:`FrozenEnsemble` is immutable.

        Use the live ensemble recipe (:class:`DistributionEnsemble`,
        :class:`CrossProductEnsemble`, etc.) stored in
        :attr:`~simulation.handle.EvaluationHandle._ensemble_recipe` to draw
        more samples.
        """
        raise TypeError(
            "FrozenEnsemble cannot draw new samples — it holds a fixed snapshot. "
            "Use the live ensemble recipe to draw more samples."
        )

    def concat(self, other: FrozenEnsemble) -> FrozenEnsemble:
        """Return a new single-axis frozen ensemble concatenating *other*'s samples.

        Both *self* and *other* must be single-axis.  Use :meth:`concat_along`
        for multi-axis ensembles.
        """
        assert len(self._axes) == 1 and len(other._axes) == 1, (
            "FrozenEnsemble.concat is for single-axis ensembles; use concat_along for multi-axis."
        )
        S1 = self._weights[0].size
        S2 = other._weights[0].size
        alpha = S1 / (S1 + S2)
        merged_weights = np.concatenate([self._weights[0] * alpha, other._weights[0] * (1.0 - alpha)])
        merged_assignments = {
            idx: np.concatenate([self._cached_assignments[idx], other._cached_assignments[idx]])
            for idx in self._cached_assignments
        }
        new_axes = _grown_axes(self._axes, (S1,), self._axes[0].name, S1 + S2)
        return FrozenEnsemble(
            new_axes,
            (merged_weights,),
            merged_assignments,
        )

    def concat_along(self, axis_name: str, other: FrozenEnsemble) -> FrozenEnsemble:
        """Extend the named axis by appending *other* (single-axis) into this ensemble.

        Indexes assigned to *axis_name* (shape > 1 at that dimension) are
        concatenated; all other indexes are carried forward unchanged.  Weights
        are combined with the size-proportional mixture rule.

        Parameters
        ----------
        axis_name:
            Name of the ENSEMBLE axis to grow.
        other:
            Single-axis :class:`FrozenEnsemble` produced by the recipe's
            :meth:`draw_batch`.
        """
        ax_idx = next(i for i, ax in enumerate(self._axes) if ax.name == axis_name)
        S1 = self._weights[ax_idx].size
        S2 = other._weights[0].size
        alpha = S1 / (S1 + S2)
        merged_weights = np.concatenate([self._weights[ax_idx] * alpha, other._weights[0] * (1.0 - alpha)])
        sizes = tuple(w.size for w in self._weights)
        new_axes = _grown_axes(self._axes, sizes, axis_name, S1 + S2)
        new_weights = tuple(merged_weights if i == ax_idx else w for i, w in enumerate(self._weights))
        M = len(self._axes)
        merged_assignments: dict[GenericIndex, np.ndarray] = {}
        for gidx, arr in self._cached_assignments.items():
            if arr.shape[ax_idx] > 1:
                other_arr = other._cached_assignments[gidx]  # shape (S2,)
                target_shape = [1] * M
                target_shape[ax_idx] = S2
                merged_assignments[gidx] = np.concatenate([arr, other_arr.reshape(target_shape)], axis=ax_idx)
            else:
                merged_assignments[gidx] = arr
        return FrozenEnsemble(new_axes, new_weights, merged_assignments)

    def with_replaced_axis(self, axis_name: str, other: FrozenEnsemble) -> FrozenEnsemble:
        """Return a copy with the named axis replaced by *other*'s samples.

        Used when extending one axis of a multi-axis ensemble: *other*
        (single-axis) provides fresh samples for the target axis; all other
        axes keep their existing samples.  The result is a full multi-axis
        ensemble suitable for :meth:`~simulation.evaluation.Evaluation.execute_plan`.

        Parameters
        ----------
        axis_name:
            Name of the ENSEMBLE axis to replace.
        other:
            Single-axis :class:`FrozenEnsemble` produced by the recipe's
            :meth:`draw_batch`.
        """
        ax_idx = next(i for i, ax in enumerate(self._axes) if ax.name == axis_name)
        S2 = other._weights[0].size
        sizes = tuple(w.size for w in self._weights)
        new_axes = _grown_axes(self._axes, sizes, axis_name, S2)
        new_weights = tuple(other._weights[0] if i == ax_idx else w for i, w in enumerate(self._weights))
        M = len(self._axes)
        merged_assignments: dict[GenericIndex, np.ndarray] = {}
        for gidx, arr in self._cached_assignments.items():
            if arr.shape[ax_idx] > 1:
                other_arr = other._cached_assignments[gidx]  # shape (S2,)
                target_shape = [1] * M
                target_shape[ax_idx] = S2
                merged_assignments[gidx] = other_arr.reshape(target_shape)
            else:
                merged_assignments[gidx] = arr
        return FrozenEnsemble(new_axes, new_weights, merged_assignments)


@runtime_checkable
class BatchDrawable(Protocol):
    """Protocol for ensemble types that can draw additional Monte Carlo batches.

    Implemented by :class:`DistributionEnsemble`, :class:`CrossProductEnsemble`,
    and :class:`PartitionedEnsemble`.  Types where incremental extension is
    meaningless (e.g. :class:`FrozenEnsemble`) raise :class:`TypeError`.

    The caller (e.g. :class:`~simulation.handle.EvaluationHandle`) owns the
    RNG and passes it in.  Implementations must **not** store *rng* after the
    call returns and must not accumulate internal state across calls.
    """

    def draw_batch(self, size: int, rng: np.random.Generator, *, axis: str | None = None) -> FrozenEnsemble:
        """Draw *size* new samples and return them as a :class:`FrozenEnsemble`.

        Parameters
        ----------
        size:
            New Monte Carlo budget: total samples for :class:`DistributionEnsemble`,
            additional samples per combo for :class:`CrossProductEnsemble`,
            new samples on the named axis for :class:`PartitionedEnsemble`.
        rng:
            Caller-owned :class:`numpy.random.Generator`.  Advanced in-place.
        axis:
            For :class:`PartitionedEnsemble`: name of the ENSEMBLE axis to
            extend.  Must be ``None`` for single-axis ensemble types.
        """
        ...  # pragma: no cover


class PartitionedEnsemble:
    """Ensemble that distributes abstract indexes across multiple named ENSEMBLE axes.

    Each :class:`EnsembleAxisSpec` defines one ENSEMBLE axis: its name, which
    abstract indexes belong to it, and how many independent samples to draw.
    The result tensor has shape ``(S0, S1, …, S(M-1))`` before marginalization,
    where ``Sj`` is the size of axis ``j``.  This allows orthogonal Monte Carlo
    budgets for independent uncertainty sources.

    Abstract indexes not mentioned in any ``axes`` spec must be covered by
    ``default_axis``; otherwise a :class:`ValueError` is raised at construction.

    Parameters
    ----------
    scenario:
        The scenario whose abstract indexes are sampled.
    axes:
        Ordered list of :class:`EnsembleAxisSpec` objects, each naming a subset
        of abstract indexes and a sample size.
    default_axis:
        Optional catch-all :class:`EnsembleAxisSpec` for abstract indexes not
        listed in *axes*.  Its ``indexes`` list is extended automatically.
    rng:
        Optional :class:`numpy.random.Generator` for reproducibility.

    Raises
    ------
    ValueError
        If any abstract index is not covered by any spec and no *default_axis*
        is provided, or if any spec index is not abstract in *model*.
    """

    def __init__(
        self,
        scenario: Scenario,
        axes: list[EnsembleAxisSpec],
        default_axis: EnsembleAxisSpec | None = None,
        rng: np.random.Generator | None = None,
    ) -> None:
        if not isinstance(scenario, Scenario):
            raise TypeError(f"{type(self).__name__} expects a Scenario, got {type(scenario).__name__}")
        abstract = list(scenario.abstract_indexes())
        abstract_set = set(abstract)

        # Build mapping: index → spec
        assigned: dict[GenericIndex, EnsembleAxisSpec] = {}
        for spec in axes:
            for idx in spec.indexes:
                if idx not in abstract_set:
                    raise ValueError(
                        f"Index {getattr(idx, 'name', repr(idx))!r} in EnsembleAxisSpec "
                        f"{spec.name!r} is not an abstract index of the model."
                    )
                if idx in assigned:
                    raise ValueError(
                        f"Index {getattr(idx, 'name', repr(idx))!r} appears in more than one EnsembleAxisSpec."
                    )
                assigned[idx] = spec

        non_samplable = [
            idx
            for idx in abstract
            if not (isinstance(idx, CategoricalIndex) or scenario.effective_distribution(idx) is not None)
        ]
        if non_samplable:
            names = ", ".join(getattr(idx, "name", repr(idx)) for idx in non_samplable)
            raise ValueError(
                f"{type(self).__name__} requires all abstract indexes to be Distribution-backed "
                f"or CategoricalIndex; unsupported indexes: {names}"
            )

        # Handle unassigned indexes
        unassigned = [idx for idx in abstract if idx not in assigned]
        if unassigned:
            if default_axis is None:
                names = ", ".join(getattr(idx, "name", repr(idx)) for idx in unassigned)
                raise ValueError(
                    f"Abstract indexes not covered by any EnsembleAxisSpec and no default_axis provided: {names}"
                )
            for idx in unassigned:
                default_axis.indexes.append(idx)
                assigned[idx] = default_axis

        # Final ordered spec list (include default_axis if used)
        all_specs: list[EnsembleAxisSpec] = list(axes)
        if default_axis is not None and default_axis.indexes:
            all_specs.append(default_axis)

        # Validate unique axis names across all specs (including default_axis).
        seen_names: set[str] = set()
        for spec in all_specs:
            if spec.name in seen_names:
                raise ValueError(f"Duplicate EnsembleAxisSpec name: {spec.name!r}")
            seen_names.add(spec.name)

        self._axes: tuple[Axis, ...] = tuple(Axis(spec.name, ENSEMBLE) for spec in all_specs)
        self._weights: tuple[np.ndarray, ...] = tuple(np.full(spec.size, 1.0 / spec.size) for spec in all_specs)
        self._specs = all_specs
        self._assigned = assigned
        self._rng = rng
        self._scenario = scenario

    @property
    def ensemble_axes(self) -> tuple[Axis, ...]:
        """Return the ENSEMBLE axes, one per partition."""
        return self._axes

    @property
    def ensemble_weights(self) -> tuple[np.ndarray, ...]:
        """Return the weight arrays, one per partition axis."""
        return self._weights

    def assignments(self) -> Mapping[GenericIndex, np.ndarray]:
        """Return batched samples for every abstract index.

        Each value has shape ``(1, …, Sj, …, 1)`` — size ``Sj`` only at the
        dimension corresponding to the index's own axis, 1 everywhere else.
        """
        M = len(self._specs)
        result: dict[GenericIndex, np.ndarray] = {}

        for j, spec in enumerate(self._specs):
            Sj = spec.size
            for idx in spec.indexes:
                # Sample Sj values for this index.
                if isinstance(idx, CategoricalIndex):
                    samples = idx.sample(self._rng, size=Sj)  # shape (Sj,)
                else:
                    dist = self._scenario.effective_distribution(idx)
                    if dist is None:  # pragma: no cover — guarded by __init__ validation
                        raise ValueError(
                            f"Index {getattr(idx, 'name', repr(idx))!r} is not Distribution-backed "
                            f"or CategoricalIndex in this scenario; cannot sample."
                        )
                    if self._rng is not None:
                        samples = np.asarray(dist.rvs(size=Sj, random_state=self._rng))
                    else:
                        samples = np.asarray(dist.rvs(size=Sj))

                # Reshape to (1, …, Sj, …, 1): size Sj at position j, 1 elsewhere.
                shape = [1] * M
                shape[j] = Sj
                result[idx] = samples.reshape(shape)

        return result

    def draw_batch(self, size: int, rng: np.random.Generator, *, axis: str | None = None) -> FrozenEnsemble:
        """Draw *size* new samples for the named ENSEMBLE axis.

        Returns a **single-axis** :class:`FrozenEnsemble` carrying fresh samples
        only for the indexes assigned to *axis*.  The caller
        (:class:`~simulation.handle.EvaluationHandle`) is responsible for
        combining this batch with the existing frozen state for all other axes
        via :meth:`FrozenEnsemble.with_replaced_axis` before passing it to
        :meth:`~simulation.evaluation.Evaluation.execute_plan`.

        Parameters
        ----------
        size:
            Number of new samples to draw for the target axis.
        rng:
            Caller-owned :class:`numpy.random.Generator`.  Advanced in-place.
        axis:
            Name of the ENSEMBLE axis to extend.  Required when this ensemble
            has more than one axis; inferred automatically when there is only
            one axis.

        Raises
        ------
        ValueError
            If *axis* is ``None`` and the ensemble has more than one axis, or
            if *axis* does not name a known axis.
        """
        if axis is None:
            if len(self._specs) == 1:
                axis = self._specs[0].name
            else:
                names = [s.name for s in self._specs]
                raise ValueError(
                    f"PartitionedEnsemble has {len(self._specs)} axes {names!r}; "
                    "specify axis= to indicate which axis to extend."
                )
        spec = next((s for s in self._specs if s.name == axis), None)
        if spec is None:
            raise ValueError(f"PartitionedEnsemble has no axis named {axis!r}.")
        new_assignments: dict[GenericIndex, np.ndarray] = {}
        for idx in spec.indexes:
            if isinstance(idx, CategoricalIndex):
                new_assignments[idx] = idx.sample(rng, size=size)  # shape (size,)
            else:
                dist = self._scenario.effective_distribution(idx)
                if dist is None:  # pragma: no cover — guarded by __init__ validation
                    raise ValueError(f"Index {getattr(idx, 'name', repr(idx))!r} is not samplable.")
                raw = dist.rvs(size=size, random_state=rng) if rng is not None else dist.rvs(size=size)
                new_assignments[idx] = np.asarray(raw)  # shape (size,)
        return FrozenEnsemble(
            (Axis(axis, ENSEMBLE),),
            (np.full(size, 1.0 / size),),
            new_assignments,
        )


class DistributionEnsemble:
    """Ensemble that independently samples each samplable abstract index.

    Each of the *size* scenarios draws one sample from every abstract index in
    *model* and assigns equal weight ``1 / size``.  Two kinds of abstract index
    are supported:

    * :class:`~model.index.Index` backed by a :class:`~model.index.Distribution`
      — sampled via ``Distribution.rvs``.
    * :class:`~model.index.CategoricalIndex` — sampled via
      :meth:`~model.index.CategoricalIndex.sample`.

    This is the standard ensemble for models whose only source of uncertainty
    is a set of independently distributed parameters (e.g., the Bologna
    mobility example) or runtime model variants selected via a
    :class:`~model.index.CategoricalIndex`.

    Implements the :class:`AxisEnsemble` protocol: :meth:`assignments` returns
    batched arrays of shape ``(size,)`` for each abstract index (no scenario
    enumeration).  The legacy :meth:`__iter__` interface is preserved for
    backward compatibility.

    Parameters
    ----------
    scenario:
        The scenario whose abstract indexes are sampled.  Every abstract index
        must be either :class:`~model.index.Distribution`-backed or a
        :class:`~model.index.CategoricalIndex`; a :class:`ValueError` is
        raised at construction time otherwise.
    size:
        Number of scenarios (samples).
    rng:
        Optional :class:`numpy.random.Generator` for reproducibility.  When
        ``None``, the global NumPy random state is used.
    exclude:
        Abstract indexes that will be supplied externally (e.g. via
        ``parameters=`` at :meth:`~evaluation.Evaluation.evaluate` time) and
        must not be sampled by this ensemble.  These indexes are silently
        skipped in both the constructor validation and :meth:`assignments`.
        Callers of :meth:`EvaluationHandle.evaluate` and
        :meth:`AsyncEvaluationHandle.evaluate` should not set this
        directly; it is managed automatically from the ``parameters=`` dict.
    plan:
        Optional :class:`~simulation.plan.EvaluationPlan`.  When supplied
        (and built with ``strategy="regional"``), :meth:`assignments`
        performs *per-scope sampling* (see :meth:`_scoped_assignments`):
        indexes scoped to a particular region are sampled only at the
        scenario positions where that region is active.  This aligns the
        Monte Carlo budget with branch selection probabilities.  When
        ``None`` (default), the original "all indexes, all scenarios"
        sampling is used.

    Notes
    -----
    **Known limitation — categorical sampling overhead**

    When the model has only :class:`~model.index.CategoricalIndex` abstract
    indexes, it would be possible to enumerate outcomes exactly — yielding one
    scenario per outcome key weighted by its declared probability — eliminating
    Monte Carlo noise entirely.  This optimisation is not implemented because
    it does not compose with :class:`~model.index.Distribution`-backed indexes:
    once sampling is required for any index, all indexes share the same Monte
    Carlo budget and the categorical dimension cannot be separated out.
    """

    def __init__(
        self,
        scenario: Scenario,
        size: int,
        rng: np.random.Generator | None = None,
        *,
        exclude: frozenset["GenericIndex"] | None = None,
        plan: EvaluationPlan | None = None,
    ) -> None:
        if not isinstance(scenario, Scenario):
            raise TypeError(f"{type(self).__name__} expects a Scenario, got {type(scenario).__name__}")
        self._scenario = scenario
        self._size = size
        self._rng = rng
        self._axis = Axis("_ensemble", ENSEMBLE)
        self._exclude: frozenset[GenericIndex] = exclude or frozenset()
        self._plan: EvaluationPlan | None = plan
        # Per-scope sampling cache.  Populated lazily on the first
        # call to _scoped_assignments; reused thereafter.  Both fields
        # are functions of (plan, scenario) which are fixed at
        # construction time, so the cache is valid for the ensemble's
        # lifetime (mutating the scenario after construction is not
        # supported — Scenario.overrides returns a copy, and the
        # abstract_indexes() result is computed once at scenario use).
        self._scoped: dict[tuple[Any, ...], frozenset[GenericIndex]] | None = None
        self._node_to_idx: dict[Any, GenericIndex] | None = None
        # Validate that all abstract indexes can be sampled by this ensemble.
        # Indexes in `exclude` are covered by parameters= at evaluate time and
        # are skipped here. Abstract indexes that are neither Distribution-backed
        # nor CategoricalIndex cannot be assigned here and will cause
        # PlaceholderValueNotProvided at runtime.
        abstract = scenario.abstract_indexes()
        non_samplable = [
            idx
            for idx in abstract
            if idx not in self._exclude
            and not (isinstance(idx, CategoricalIndex) or scenario.effective_distribution(idx) is not None)
        ]
        if non_samplable:
            names = ", ".join(getattr(idx, "name", repr(idx)) for idx in non_samplable)
            raise ValueError(
                f"DistributionEnsemble requires all abstract indexes to be Distribution-backed "
                f"or CategoricalIndex; unsupported indexes: {names}"
            )

    # ------------------------------------------------------------------
    # AxisEnsemble protocol
    # ------------------------------------------------------------------

    @property
    def ensemble_axes(self) -> tuple[Axis, ...]:
        """Single ENSEMBLE axis of size *size*."""
        return (self._axis,)

    @property
    def ensemble_weights(self) -> tuple[np.ndarray, ...]:
        """Uniform weight vector of shape ``(size,)``."""
        return (np.full(self._size, 1.0 / self._size),)

    def assignments(self) -> Mapping[GenericIndex, np.ndarray]:
        """Return batched samples for every abstract index.

        Each value has shape ``(size,)`` — the single ENSEMBLE axis dimension.
        Scalar-valued :class:`~model.index.Distribution`-backed indexes yield
        float arrays; :class:`~model.index.CategoricalIndex` indexes yield
        object arrays of string keys.

        When the ensemble is constructed with ``plan=``, sampling is
        *per-scope* (see :meth:`_scoped_assignments`): indexes that are
        active in only some regions are sampled only at the scenario
        positions where those regions are active; unsampled slots are
        filled with a sentinel (``np.nan`` for
        :class:`~model.index.Distribution`-backed indexes, ``None`` for
        :class:`~model.index.CategoricalIndex`).  The executor's region
        masking ensures these slots are never read for regional plans;
        downstream code that inspects assignments directly should
        filter out sentinels (``~np.isnan(arr)`` for float arrays,
        ``arr != None`` for object arrays) before statistical analysis.

        When ``plan=`` is set with a focused :class:`EvaluationPlan`
        (built via ``nodes_of_interest=[...]``), the result includes
        only the abstract indexes that are dependencies of
        ``nodes_of_interest``; other abstract indexes from the
        scenario are not sampled.
        """
        if self._plan is not None:
            return self._scoped_assignments()
        abstract = [idx for idx in self._scenario.abstract_indexes() if idx not in self._exclude]
        result: dict[GenericIndex, np.ndarray] = {}
        for idx in abstract:
            if isinstance(idx, CategoricalIndex):
                raw_keys = idx.sample(self._rng, size=self._size)  # shape (S,)
                result[idx] = raw_keys  # shape (S,)
            else:
                dist = self._scenario.effective_distribution(idx)
                assert dist is not None
                if self._rng is not None:
                    raw = dist.rvs(size=self._size, random_state=self._rng)
                else:
                    raw = dist.rvs(size=self._size)
                result[idx] = np.asarray(raw)  # shape (S,)
        return result

    def _scoped_assignments(self) -> dict[GenericIndex, np.ndarray]:
        """Per-scope sampling: allocate each abstract index to its scope bucket.

        Iterates the buckets from :meth:`EvaluationPlan.scoped_abstract_indexes`
        in first-seen order — which is outer-to-inner by construction
        (the outer shared region is the first region in the plan, so its
        ``guards=()`` bucket is inserted first, then ``(outer_branch,)``
        buckets, then ``(outer_branch, inner_branch)`` buckets, etc.).
        Each bucket is sampled at the positions where every guard in the
        chain matches the selector values already drawn in ancestor
        buckets; unsampled slots are filled with a semantically valid
        default.
        """
        plan = self._plan
        assert plan is not None
        if self._scoped is None:
            scoped = plan.scoped_abstract_indexes(self._scenario)
            node_to_idx = {idx.node: idx for idx in self._scenario.abstract_indexes()}
            self._scoped, self._node_to_idx = scoped, node_to_idx
        scoped = self._scoped
        node_to_idx = self._node_to_idx
        assert scoped is not None and node_to_idx is not None
        exclude = self._exclude
        result: dict[GenericIndex, np.ndarray] = {}
        for guards, indexes in scoped.items():
            active_indexes = [idx for idx in indexes if idx not in exclude]
            if not active_indexes:
                continue
            positions = self._active_positions(guards, node_to_idx, result)
            for idx in active_indexes:
                result[idx] = self._sample_with_default(idx, positions)
        return result

    def _active_positions(
        self,
        guards: tuple[Any, ...],
        node_to_idx: dict[Any, GenericIndex],
        result: dict[GenericIndex, np.ndarray],
    ) -> np.ndarray:
        """Return the scenario indices where every guard in *guards* matches.

        Requires that each guard's selector is in ``result`` already —
        i.e. its bucket was processed in an earlier (more outer) pass.
        """
        mask = np.ones(self._size, dtype=bool)
        for g in guards:
            idx = node_to_idx.get(g.selector_node)
            if idx is None:
                raise ValueError(
                    f"Selector for guard {g.branch_key!r} is not in "
                    f"scenario.abstract_indexes(); per-scope sampling requires every "
                    f"guard's selector to be an abstract index.  Omit `plan=` to fall "
                    f"back to the all-abstract sampler, or use a plan whose guards "
                    f"only reference abstract selectors."
                )
            if idx not in result:
                raise RuntimeError(
                    f"Selector {idx.name!r} for guard {g.branch_key!r} has not been "
                    f"sampled yet; the bucket order from "
                    f"EvaluationPlan.scoped_abstract_indexes is not outer-to-inner."
                )
            mask &= result[idx] == g.branch_key
        return np.flatnonzero(mask)

    def _sample_with_default(self, idx: GenericIndex, positions: np.ndarray) -> np.ndarray:
        """Sample ``len(positions)`` values for *idx* and fill the rest with a placeholder."""
        n_branch = int(len(positions))
        placeholder = self._placeholder_value(idx)
        is_categorical = isinstance(idx, CategoricalIndex)
        out = np.full(
            self._size,
            placeholder,
            dtype=object if is_categorical else None,
        )
        if n_branch == 0:
            return out
        if is_categorical:
            out[positions] = idx.sample(self._rng, size=n_branch)
        else:
            dist = self._scenario.effective_distribution(idx)
            assert dist is not None
            if self._rng is not None:
                raw = dist.rvs(size=n_branch, random_state=self._rng)
            else:
                raw = dist.rvs(size=n_branch)
            out[positions] = np.asarray(raw)
        return out

    def _placeholder_value(self, idx: GenericIndex) -> Any:
        """Sentinel value used to fill unsampled slots.

        * :class:`~model.index.CategoricalIndex` — ``None`` (object-dtype
          sentinel).  Filter real samples with ``arr != None``.
        * :class:`~model.index.Distribution`-backed indexes — ``np.nan``
          (float-dtype sentinel).  Filter real samples with
          ``~np.isnan(arr)``.

        Sentinels are deliberately *not* type-safe defaults like
        ``dist.mean()`` or ``argmax(outcomes)``: a downstream user
        naively computing ``arr.mean()`` would get a silently biased
        estimate with the latter, but a loud ``nan`` with the former
        that forces them to filter.  The executor's region masking
        ensures unsampled slots are never read for regional plans, so
        the sentinels are "don't care" for execution — they exist
        purely to inform downstream consumers.
        """
        if isinstance(idx, CategoricalIndex):
            return None
        return float("nan")

    def draw_batch(self, size: int, rng: np.random.Generator, *, axis: str | None = None) -> FrozenEnsemble:
        """Draw *size* fresh samples and return them as a :class:`FrozenEnsemble`.

        The caller (e.g. :class:`~simulation.handle.EvaluationHandle`) owns *rng*
        and passes it in so the ensemble remains stateless — it never stores *rng*
        and calling this method multiple times with the same *rng* will advance it
        reproducibly.

        Parameters
        ----------
        size:
            Number of new Monte Carlo samples to draw.
        rng:
            Caller-owned :class:`numpy.random.Generator`.  Advanced in-place.
        axis:
            Must be ``None``; :class:`DistributionEnsemble` has a single ENSEMBLE
            axis and does not support named-axis extension.

        Returns
        -------
        FrozenEnsemble
            Frozen batch whose samples are identical to what
            ``DistributionEnsemble(self._scenario, size, rng=rng,
            exclude=self._exclude)`` would produce.

        Raises
        ------
        ValueError
            If *axis* is not ``None``.
        """
        if axis is not None:
            raise ValueError(f"DistributionEnsemble has a single ENSEMBLE axis; axis= must be None, got {axis!r}.")
        return self._sample_batch(size, rng)

    def _sample_batch(self, size: int, rng: np.random.Generator | None) -> FrozenEnsemble:
        """Sample a frozen batch of the given size without re-running ``__init__`` validation.

        The constructor has already validated ``scenario``, ``exclude``,
        and ``plan``; only ``size`` and ``rng`` change between batches.
        Temporarily overrides ``self._size`` / ``self._rng`` and restores
        them in a ``finally`` block so the original ensemble state is
        preserved even if sampling raises.
        """
        original_size, original_rng = self._size, self._rng
        self._size, self._rng = size, rng
        try:
            return FrozenEnsemble(
                self.ensemble_axes,
                self.ensemble_weights,
                dict(self.assignments()),
            )
        finally:
            self._size, self._rng = original_size, original_rng

    # ------------------------------------------------------------------
    # Legacy iterable interface (backward compatible)
    # ------------------------------------------------------------------

    def __iter__(self) -> Iterator[WeightedScenario]:
        """Yield *size* equally-weighted scenarios, one sample per index per scenario."""
        abstract = self._scenario.abstract_indexes()
        weight = 1.0 / self._size

        # Pre-sample each index: shape (size, 1) so that stacking produces
        # (size, 1) substitution arrays, which broadcast correctly against
        # timeseries of shape (T,) via numpy's (size, 1) × (T,) → (size, T).
        samples: dict[GenericIndex, np.ndarray] = {}
        for idx in abstract:
            if isinstance(idx, CategoricalIndex):
                raw_keys = idx.sample(self._rng, size=self._size)
                samples[idx] = raw_keys.reshape(self._size, 1)
            else:
                dist = self._scenario.effective_distribution(idx)
                assert dist is not None
                if self._rng is not None:
                    raw = dist.rvs(size=self._size, random_state=self._rng)
                else:
                    raw = dist.rvs(size=self._size)
                # Wrap each sample as a 1-element array so stacking gives (S, 1).
                samples[idx] = np.asarray(raw).reshape(self._size, 1)

        for i in range(self._size):
            assignments: dict[GenericIndex, Any] = {idx: samples[idx][i] for idx in abstract}
            yield weight, assignments


# ---------------------------------------------------------------------------
# CrossProductEnsemble — helpers
# ---------------------------------------------------------------------------


def _topo_sort_categoricals(
    cats: list[CategoricalIndex | ConditionalCategoricalIndex],
) -> list[CategoricalIndex | ConditionalCategoricalIndex]:
    """Return *cats* in topological order (parents before children)."""
    cat_ids = {id(c) for c in cats}
    visited: set[int] = set()
    order: list[CategoricalIndex | ConditionalCategoricalIndex] = []

    def visit(c: CategoricalIndex | ConditionalCategoricalIndex) -> None:
        if id(c) in visited:
            return
        visited.add(id(c))
        if isinstance(c, ConditionalCategoricalIndex):
            for p in c.parents:
                if id(p) in cat_ids:
                    visit(p)
        order.append(c)

    for c in cats:
        visit(c)
    return order


def _topo_sort_dists(
    dists: list[Index],
) -> list[Index]:
    """Return *dists* in topological order (parents before children).

    Only distribution-to-distribution edges are followed; categorical parents
    are already resolved before this sort runs.
    """
    dist_ids = {id(d) for d in dists}
    visited: set[int] = set()
    order: list[Index] = []

    def visit(d: Index) -> None:
        if id(d) in visited:
            return
        visited.add(id(d))
        if isinstance(d, ConditionalDistributionIndex):
            for p in d.parents:
                if id(p) in dist_ids:
                    visit(p)  # type: ignore[arg-type]
        order.append(d)

    for d in dists:
        visit(d)
    return order


def _cat_samples(
    values: list[str],
    probs: list[float],
    max_categorical_size: int,
    rng: np.random.Generator | None,
) -> list[tuple[float, str]]:
    """Return ``(weight, value)`` pairs for one categorical iteration.

    Enumerates when ``max_categorical_size >= len(values)``; Monte-Carlo samples
    otherwise.  Probabilities are renormalised over *values* (handles subsets).
    """
    total = sum(probs)
    norm_probs = [p / total for p in probs]
    if max_categorical_size < len(values):
        arr = np.array(values, dtype=object)
        if rng is not None:
            choices = rng.choice(arr, size=max_categorical_size, p=norm_probs)
        else:
            choices = np.random.choice(arr, size=max_categorical_size, p=norm_probs)
        return [(1.0 / max_categorical_size, str(c)) for c in choices]
    return [(p / total, v) for p, v in zip(probs, values)]


# ---------------------------------------------------------------------------
# CrossProductEnsemble
# ---------------------------------------------------------------------------


class CrossProductEnsemble:
    """Ensemble that enumerates categorical combinations and samples distribution-backed indexes.

    Handles any combination of
    :class:`~model.index.CategoricalIndex`,
    :class:`~model.index.ConditionalCategoricalIndex`,
    :class:`~model.index.DistributionIndex`, and
    :class:`~model.index.ConditionalDistributionIndex`.

    Abstract indexes that are neither categorical nor distribution-backed
    (e.g. a plain placeholder
    :class:`~model.index.Index`) are silently excluded from the ensemble — they
    must be supplied as PARAMETER axes to
    :meth:`~simulation.evaluation.Evaluation.evaluate`.

    **Scenario dict overrides** are respected automatically.  When the scenario
    carries a ``dict[str, float]`` override for a
    :class:`~model.index.CategoricalIndex`, the override probabilities replace
    the model's declared probabilities *and* the support is automatically
    restricted to the override's keys (so only those outcomes are sampled)::

        # cv_weather model probs: good=0.33, unsettled=0.33, bad=0.34
        scenario = Scenario(model, overrides={cv_weather: {"good": 0.8, "unsettled": 0.2}})
        ensemble = CrossProductEnsemble(scenario)  # only good/unsettled, with 80/20 weights

    A ``list[str]`` override restricts to that subset of outcomes and
    renormalises the model's declared probabilities automatically::

        scenario = Scenario(model, overrides={cv_weather: ["good", "unsettled"]})
        ensemble = CrossProductEnsemble(scenario)

    **Sampling budget for distribution-backed indexes** — when a model retains
    distribution-backed indexes in the ensemble (i.e. they are not declared as
    a :attr:`~simulation.scenario.Scenario.parameter_axes`), each categorical
    combination would by default receive exactly one sample from those
    distributions, giving a total of only ``|categorical cross-product|``
    samples and high run-to-run variance.  Use *n_samples_per_combo* to draw
    more independent samples per categorical combination::

        # 3 weather × 2 seasons = 6 combos × 50 samples = 300 total scenarios
        ensemble = CrossProductEnsemble(scenario, n_samples_per_combo=50)

    Each combo's weight ``w`` is split equally among its *n_samples_per_combo*
    replicates (each replicate carries weight ``w / n_samples_per_combo``), so
    the ensemble weights still sum to 1.0.  The default value of 1 preserves
    the previous behaviour exactly.

    Parameters
    ----------
    scenario:
        Scenario whose abstract indexes are enumerated / sampled.
    max_categorical_size:
        Maximum number of samples per categorical axis.  When the support (or
        restricted subset) is larger than this threshold, the axis is Monte-Carlo
        sampled *max_categorical_size* times.
    n_samples_per_combo:
        Number of independent distribution samples to draw for each categorical
        combination.  Total ensemble size is
        ``|categorical cross-product| × n_samples_per_combo``.  Must be >= 1.
        Has no effect when all distribution-backed indexes are declared as
        parameter axes on *scenario*.
    rng:
        Optional :class:`numpy.random.Generator` for reproducibility.

    Implements :class:`AxisEnsemble`.
    """

    def __init__(
        self,
        scenario: Scenario,
        max_categorical_size: int = 20,
        n_samples_per_combo: int = 1,
        rng: np.random.Generator | None = None,
    ) -> None:
        if not isinstance(scenario, Scenario):
            raise TypeError(f"{type(self).__name__} expects a Scenario, got {type(scenario).__name__}")
        if n_samples_per_combo < 1:
            raise ValueError(f"n_samples_per_combo must be >= 1; got {n_samples_per_combo}.")

        # PARAMETER-axis indexes are already excluded by Scenario.abstract_indexes().
        abstract = list(scenario.abstract_indexes())

        # Classify abstract indexes.
        cats_unordered: list[CategoricalIndex | ConditionalCategoricalIndex] = []
        dists_unordered: list[Index] = []
        for idx in abstract:
            if isinstance(idx, CategoricalIndex | ConditionalCategoricalIndex):
                cats_unordered.append(idx)
            elif isinstance(idx, ConditionalDistributionIndex):
                dists_unordered.append(idx)
            elif isinstance(idx, DistributionIndex):
                dists_unordered.append(idx)
            # else: plain placeholder Index — skip silently

        self._categoricals = _topo_sort_categoricals(cats_unordered)
        self._distributions = _topo_sort_dists(dists_unordered)
        self._max_categorical_size = max_categorical_size
        self._scenario = scenario
        self._rng = rng

        (
            self._assignments,
            self._weights_arr,
            self.size,
            self._combo_cats,
            self._combo_weights,
        ) = self._compute_assignments(n_samples_per_combo, rng)
        self._n_combos: int = self._combo_weights.size

        self._axis = Axis("_cross_product", ENSEMBLE)

    def _compute_assignments(
        self,
        n_samples_per_combo: int,
        rng: np.random.Generator | None,
    ) -> tuple[
        dict[GenericIndex, np.ndarray],
        np.ndarray,
        int,
        dict[GenericIndex, np.ndarray],
        np.ndarray,
    ]:
        """Compute categorical/distribution assignments, weights, and total size.

        Parameters
        ----------
        n_samples_per_combo:
            Number of Monte Carlo samples per categorical combination.
        rng:
            Random generator for distribution sampling (may be ``None``).

        Returns
        -------
        (assignments, weights, S_total, combo_cats, combo_weights)
            *combo_cats* maps each categorical index to its per-combo value array
            (shape ``(S,)``); *combo_weights* is the normalised per-combo weight
            array (shape ``(S,)``).  Both are stored on the instance so that
            :meth:`draw_batch` can reuse the categorical structure.
        """
        scenario = self._scenario
        max_categorical_size = self._max_categorical_size
        categoricals = self._categoricals
        distributions = self._distributions

        # Build cross-product of categorical values.
        # Each entry: (joint_weight, {id(cat): value_str}) — id keys avoid
        # GenericIndex.__eq__ returning a graph node.
        combos: list[tuple[float, dict[int, str]]] = [(1.0, {})]

        for cat in categoricals:
            new_combos: list[tuple[float, dict[int, str]]] = []
            for w, assignments in combos:
                if isinstance(cat, ConditionalCategoricalIndex):
                    parent_values = {p.name: assignments[id(p)] for p in cat.parents}
                    outcomes = cat.outcomes_for(**parent_values)
                    values = cat.support
                else:
                    outcomes = scenario.effective_outcomes(cat) or cat.outcomes
                    values = list(outcomes.keys())
                probs = [outcomes[v] for v in values]
                for sub_w, val in _cat_samples(values, probs, max_categorical_size, rng):
                    new_combos.append((w * sub_w, {**assignments, id(cat): val}))
            combos = new_combos

        S = len(combos)
        S_total = S * n_samples_per_combo
        combo_weights = np.array([w for (w, _) in combos])
        combo_weights /= combo_weights.sum()  # normalise against FP drift
        # Each combo is replicated n_samples_per_combo times; its weight is
        # split equally across all replicates so the total still sums to 1.0.
        weights = np.repeat(combo_weights, n_samples_per_combo) / n_samples_per_combo

        # Build categorical assignment arrays (each value repeated n_samples_per_combo times).
        # Also keep the per-combo (non-repeated) arrays so draw_batch can reuse them.
        result_assignments: dict[GenericIndex, np.ndarray] = {}
        combo_cats: dict[GenericIndex, np.ndarray] = {}
        for cat in categoricals:
            cat_arr = np.array([combo[1][id(cat)] for combo in combos], dtype=object)
            combo_cats[cat] = cat_arr
            result_assignments[cat] = np.repeat(cat_arr, n_samples_per_combo)

        # Sample distribution-backed indexes (topo order — parents before children).
        for idx in distributions:
            if isinstance(idx, ConditionalDistributionIndex):
                samples = np.empty(S_total)
                for i, (_, id_keyed_cats) in enumerate(combos):
                    # Separate categorical parents (fixed for all replicates of this combo)
                    # from distribution parents (vary per replicate).
                    cat_parent_vals: dict[str, Any] = {}
                    dist_parents: list[Any] = []
                    for p in idx.parents:
                        if isinstance(p, CategoricalIndex | ConditionalCategoricalIndex):
                            if id(p) in id_keyed_cats:
                                cat_parent_vals[p.name] = id_keyed_cats[id(p)]
                            else:
                                # Parent is pinned in the scenario (not abstract) — use its concrete value.
                                cat_parent_vals[p.name] = self._scenario.overrides[p]
                        else:
                            dist_parents.append(p)
                    if not dist_parents:
                        # All parents are categorical → same distribution for every replicate;
                        # draw all n_samples_per_combo values in one vectorised call.
                        d = idx.distribution_for(**cat_parent_vals)
                        start = i * n_samples_per_combo
                        raws = (
                            d.rvs(size=n_samples_per_combo, random_state=rng)
                            if rng is not None
                            else d.rvs(size=n_samples_per_combo)
                        )
                        samples[start : start + n_samples_per_combo] = np.asarray(raws).ravel()
                    else:
                        # At least one distribution parent → distribution differs per replicate.
                        for r in range(n_samples_per_combo):
                            full_idx = i * n_samples_per_combo + r
                            parent_vals: dict[str, Any] = dict(cat_parent_vals)
                            for p in dist_parents:
                                parent_vals[p.name] = float(result_assignments[p][full_idx])
                            d = idx.distribution_for(**parent_vals)
                            samples[full_idx] = float(d.rvs(random_state=rng) if rng is not None else d.rvs())
                result_assignments[idx] = samples
            else:
                dist = scenario.effective_distribution(idx)
                assert dist is not None
                if rng is not None:
                    result_assignments[idx] = np.asarray(dist.rvs(size=S_total, random_state=rng))
                else:
                    result_assignments[idx] = np.asarray(dist.rvs(size=S_total))

        return result_assignments, weights, S_total, combo_cats, combo_weights

    def _draw_from_combos(
        self,
        n_samples_per_combo: int,
        rng: np.random.Generator,
    ) -> tuple[dict[GenericIndex, np.ndarray], np.ndarray]:
        """Re-sample distributions over the categorical combos fixed at construction.

        The categorical assignments from ``__init__`` are reused unchanged; only
        distribution-backed indexes are drawn fresh using *rng*.  This is the
        method called by :meth:`draw_batch` so that the categorical structure is
        stable across all batches drawn from the same recipe.

        Parameters
        ----------
        n_samples_per_combo:
            Number of Monte Carlo samples per categorical combination.
        rng:
            Random generator for distribution sampling (may be ``None``).

        Returns
        -------
        (assignments, weights)
        """
        S = self._n_combos
        S_total = S * n_samples_per_combo
        weights = np.repeat(self._combo_weights, n_samples_per_combo) / n_samples_per_combo

        result_assignments: dict[GenericIndex, np.ndarray] = {}

        # Categorical: reuse the per-combo arrays stored at construction.
        for cat in self._categoricals:
            result_assignments[cat] = np.repeat(self._combo_cats[cat], n_samples_per_combo)

        # Distributions: re-sample with caller's rng (topo order — parents before children).
        for idx in self._distributions:
            if isinstance(idx, ConditionalDistributionIndex):
                samples = np.empty(S_total)
                for i in range(S):
                    cat_parent_vals: dict[str, Any] = {}
                    dist_parents: list[Any] = []
                    for p in idx.parents:
                        if isinstance(p, CategoricalIndex | ConditionalCategoricalIndex):
                            cat_parent_vals[p.name] = self._combo_cats[p][i]
                        else:
                            dist_parents.append(p)
                    if not dist_parents:
                        d = idx.distribution_for(**cat_parent_vals)
                        start = i * n_samples_per_combo
                        raws = d.rvs(size=n_samples_per_combo, random_state=rng)
                        samples[start : start + n_samples_per_combo] = np.asarray(raws).ravel()
                    else:
                        for r in range(n_samples_per_combo):
                            full_idx = i * n_samples_per_combo + r
                            parent_vals: dict[str, Any] = dict(cat_parent_vals)
                            for p in dist_parents:
                                parent_vals[p.name] = float(result_assignments[p][full_idx])
                            d = idx.distribution_for(**parent_vals)
                            samples[full_idx] = float(d.rvs(random_state=rng))
                result_assignments[idx] = samples
            else:
                dist = self._scenario.effective_distribution(idx)
                assert dist is not None
                result_assignments[idx] = np.asarray(dist.rvs(size=S_total, random_state=rng))

        return result_assignments, weights

    @property
    def ensemble_axes(self) -> tuple[Axis, ...]:
        """Single ENSEMBLE axis spanning all cross-product combinations."""
        return (self._axis,)

    @property
    def ensemble_weights(self) -> tuple[np.ndarray, ...]:
        """Weight array of shape ``(S,)`` summing to 1.0."""
        return (self._weights_arr,)

    def assignments(self) -> dict[GenericIndex, np.ndarray]:
        """Return batched assignments for all enumerated / sampled indexes."""
        return self._assignments

    def __len__(self) -> int:
        """Return the total number of scenarios."""
        return self.size

    def draw_batch(self, size: int, rng: np.random.Generator, *, axis: str | None = None) -> FrozenEnsemble:
        """Draw *size* samples per categorical combination using *rng*.

        Reuses the categorical combinations fixed at construction time and
        re-samples only distribution-backed indexes with *rng*.  The
        categorical structure is therefore identical across all batches drawn
        from the same recipe, including when ``max_categorical_size`` caused
        categories to be sampled rather than fully enumerated.

        Parameters
        ----------
        size:
            Number of Monte Carlo samples to draw **per categorical combo**.
        rng:
            Caller-owned :class:`numpy.random.Generator`.  Advanced in-place.
        axis:
            Must be ``None``; :class:`CrossProductEnsemble` has a single
            ENSEMBLE axis.

        Raises
        ------
        ValueError
            If *axis* is not ``None``.
        """
        if axis is not None:
            raise ValueError(f"CrossProductEnsemble has a single ENSEMBLE axis; axis= must be None, got {axis!r}.")
        assignments, weights = self._draw_from_combos(size, rng)
        return FrozenEnsemble(
            (self._axis,),
            (weights,),
            assignments,
        )


# ---------------------------------------------------------------------------
# sample_across — weighted presence sampling
# ---------------------------------------------------------------------------


def sample_across(
    ensemble: AxisEnsemble,
    indexes: list[ConditionalDistributionIndex],
    total: int = 200,
    rng: np.random.Generator | None = None,
) -> dict[ConditionalDistributionIndex, np.ndarray]:
    """Draw weighted samples from conditional-distribution indexes across an ensemble.

    For each scenario *i* in the ensemble, draws ``max(1, round(w_i × total))``
    samples from every index, where *w_i* is the scenario weight.  The result is
    a concatenated array of approximately *total* samples per index, distributed
    according to the ensemble's marginal distribution.

    Typical use: generating scatter-dot samples for visualising the distribution
    of presence variables against a sustainability field::

        samples = sample_across(
            ensemble,
            [pv_tourists, pv_excursionists],
            total=200,
        )
        ax.scatter(samples[pv_excursionists], samples[pv_tourists])

    Parameters
    ----------
    ensemble:
        A single-axis :class:`AxisEnsemble` (e.g. :class:`CrossProductEnsemble`).
        Multi-axis ensembles are not currently supported.
    indexes:
        Conditional-distribution indexes to sample.  Their parents must be
        present in ``ensemble.assignments()``.
    total:
        Target total number of samples per index.  Actual count may differ
        slightly due to rounding.
    rng:
        Optional :class:`numpy.random.Generator` for reproducibility.

    Returns
    -------
    dict[ConditionalDistributionIndex, np.ndarray]
        Maps each index to a 1-D array of float samples.

    Raises
    ------
    ValueError
        If *ensemble* has more than one ENSEMBLE axis, or if a parent of any
        index is not present in the ensemble assignments.
    """
    if len(ensemble.ensemble_axes) != 1:
        raise ValueError(f"sample_across requires a single-axis ensemble; got {len(ensemble.ensemble_axes)} axes.")
    weights = ensemble.ensemble_weights[0]  # shape (S,)
    assignments = ensemble.assignments()

    # Validate parents upfront.
    assignment_ids = {id(k) for k in assignments}
    for idx in indexes:
        missing = [p for p in idx.parents if id(p) not in assignment_ids]
        if missing:
            names = ", ".join(getattr(p, "name", repr(p)) for p in missing)
            raise ValueError(
                f"sample_across: parent(s) {names!r} of index {idx.name!r} are not present in the ensemble assignments."
            )

    result: dict[ConditionalDistributionIndex, list[float]] = {idx: [] for idx in indexes}

    for i, w in enumerate(weights):
        nr = max(1, round(float(w) * total))
        for idx in indexes:
            parent_vals = {p.name: assignments[p][i] for p in idx.parents}
            d = idx.distribution_for(**parent_vals)
            raw = d.rvs(size=nr, random_state=rng) if rng is not None else d.rvs(size=nr)
            arr = np.asarray(raw).ravel()
            result[idx].extend(arr.tolist())

    return {idx: np.asarray(v) for idx, v in result.items()}
