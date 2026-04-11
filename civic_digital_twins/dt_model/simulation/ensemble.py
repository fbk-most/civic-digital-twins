"""Ensemble protocol and built-in ensemble implementations."""

# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np

from ..model.axis import ENSEMBLE, Axis
from ..model.index import CategoricalIndex, Distribution, GenericIndex, Index
from ..model.model import Model
from ..model.model_variant import ModelVariant

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
    model:
        The model whose abstract indexes are sampled.
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
        model: Model | ModelVariant,
        axes: list[EnsembleAxisSpec],
        default_axis: EnsembleAxisSpec | None = None,
        rng: np.random.Generator | None = None,
    ) -> None:
        abstract = list(model.abstract_indexes())
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
        self._weights: tuple[np.ndarray, ...] = tuple(
            np.full(spec.size, 1.0 / spec.size) for spec in all_specs
        )
        self._specs = all_specs
        self._assigned = assigned
        self._rng = rng

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
                    raw = [idx.sample(self._rng) for _ in range(Sj)]
                    samples = np.array(raw, dtype=object)  # shape (Sj,)
                elif isinstance(idx, Index) and isinstance(idx.value, Distribution):
                    if self._rng is not None:
                        samples = np.asarray(idx.value.rvs(size=Sj, random_state=self._rng))
                    else:
                        samples = np.asarray(idx.value.rvs(size=Sj))
                else:
                    raise ValueError(
                        f"Index {getattr(idx, 'name', repr(idx))!r} is not Distribution-backed "
                        f"or CategoricalIndex; cannot sample."
                    )

                # Reshape to (1, …, Sj, …, 1): size Sj at position j, 1 elsewhere.
                shape = [1] * M
                shape[j] = Sj
                result[idx] = samples.reshape(shape)

        return result


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
    model:
        The model whose abstract indexes are sampled.  Every abstract index
        must be either :class:`~model.index.Distribution`-backed or a
        :class:`~model.index.CategoricalIndex`; a :class:`ValueError` is
        raised at construction time otherwise.
    size:
        Number of scenarios (samples).
    rng:
        Optional :class:`numpy.random.Generator` for reproducibility.  When
        ``None``, the global NumPy random state is used.

    Raises
    ------
    ValueError
        If any abstract index of *model* is neither
        :class:`~model.index.Distribution`-backed nor a
        :class:`~model.index.CategoricalIndex`.

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

    def __init__(self, model: Model | ModelVariant, size: int, rng: np.random.Generator | None = None) -> None:
        abstract = model.abstract_indexes()
        non_samplable = [
            idx
            for idx in abstract
            if not (
                (isinstance(idx, CategoricalIndex)) or (isinstance(idx, Index) and isinstance(idx.value, Distribution))
            )
        ]
        if non_samplable:
            names = ", ".join(getattr(idx, "name", repr(idx)) for idx in non_samplable)
            raise ValueError(
                f"DistributionEnsemble requires all abstract indexes to be Distribution-backed "
                f"or CategoricalIndex; unsupported indexes: {names}"
            )
        self._model = model
        self._size = size
        self._rng = rng
        self._axis = Axis("_ensemble", ENSEMBLE)

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
        """
        abstract = self._model.abstract_indexes()
        result: dict[GenericIndex, np.ndarray] = {}
        for idx in abstract:
            if isinstance(idx, CategoricalIndex):
                raw_keys = [idx.sample(self._rng) for _ in range(self._size)]
                result[idx] = np.array(raw_keys, dtype=object)  # shape (S,)
            else:
                assert isinstance(idx, Index) and isinstance(idx.value, Distribution)
                if self._rng is not None:
                    raw = idx.value.rvs(size=self._size, random_state=self._rng)
                else:
                    raw = idx.value.rvs(size=self._size)
                result[idx] = np.asarray(raw)  # shape (S,)
        return result

    # ------------------------------------------------------------------
    # Legacy iterable interface (backward compatible)
    # ------------------------------------------------------------------

    def __iter__(self) -> Iterator[WeightedScenario]:
        """Yield *size* equally-weighted scenarios, one sample per index per scenario."""
        abstract = self._model.abstract_indexes()
        weight = 1.0 / self._size

        # Pre-sample each index: shape (size, 1) so that stacking produces
        # (size, 1) substitution arrays, which broadcast correctly against
        # timeseries of shape (T,) via numpy's (size, 1) × (T,) → (size, T).
        samples: dict[GenericIndex, np.ndarray] = {}
        for idx in abstract:
            if isinstance(idx, CategoricalIndex):
                # Sample size string keys; wrap each as a 1-element object array.
                raw_keys = [idx.sample(self._rng) for _ in range(self._size)]
                samples[idx] = np.array(raw_keys, dtype=object).reshape(self._size, 1)
            else:
                assert isinstance(idx, Index) and isinstance(idx.value, Distribution)
                if self._rng is not None:
                    raw = idx.value.rvs(size=self._size, random_state=self._rng)
                else:
                    raw = idx.value.rvs(size=self._size)
                # Wrap each sample as a 1-element array so stacking gives (S, 1).
                samples[idx] = np.asarray(raw).reshape(self._size, 1)

        for i in range(self._size):
            assignments: dict[GenericIndex, Any] = {idx: samples[idx][i] for idx in abstract}
            yield weight, assignments
