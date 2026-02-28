"""Ensemble protocol and built-in ensemble implementations."""

from __future__ import annotations

from typing import Any, Iterator, Protocol, runtime_checkable

import numpy as np

from ..model.index import Distribution, GenericIndex, Index
from ..model.model import Model

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

    def __iter__(self) -> Iterator[WeightedScenario]: ...


class DistributionEnsemble:
    """Ensemble that independently samples each :class:`~model.index.Distribution`-backed index.

    Each of the *size* scenarios draws one sample from every
    :class:`~model.index.Distribution`-backed abstract index in *model* and
    assigns equal weight ``1 / size``.

    This is the standard ensemble for models whose only source of uncertainty
    is a set of independently distributed parameters (e.g., the Bologna
    mobility example).

    Parameters
    ----------
    model:
        The model whose abstract indexes are sampled.  Every abstract index
        must be :class:`~model.index.Distribution`-backed; a
        :class:`ValueError` is raised at construction time otherwise.
    size:
        Number of scenarios (samples).
    rng:
        Optional :class:`numpy.random.Generator` for reproducibility.  When
        ``None``, the global NumPy random state is used.

    Raises
    ------
    ValueError
        If any abstract index of *model* is not :class:`~model.index.Distribution`-backed.
    """

    def __init__(self, model: Model, size: int, rng: np.random.Generator | None = None) -> None:
        abstract = model.abstract_indexes()
        non_dist = [
            idx for idx in abstract
            if not (isinstance(idx, Index) and isinstance(idx.value, Distribution))
        ]
        if non_dist:
            names = ", ".join(getattr(idx, "name", repr(idx)) for idx in non_dist)
            raise ValueError(
                f"DistributionEnsemble requires all abstract indexes to be Distribution-backed; "
                f"non-distribution indexes: {names}"
            )
        self._model = model
        self._size = size
        self._rng = rng

    def __iter__(self) -> Iterator[WeightedScenario]:
        """Yield *size* equally-weighted scenarios, one sample per index per scenario."""
        abstract = self._model.abstract_indexes()
        weight = 1.0 / self._size

        # Pre-sample each distribution: shape (size, 1) so that stacking
        # produces (size, 1) substitution arrays, which broadcast correctly
        # against timeseries of shape (T,) via numpy's (size, 1) × (T,) → (size, T).
        samples: dict[GenericIndex, np.ndarray] = {}
        for idx in abstract:
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
