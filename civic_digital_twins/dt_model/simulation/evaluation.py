"""Generic model evaluation."""

from __future__ import annotations

import numpy as np

from ..engine.frontend import graph, linearize
from ..engine.numpybackend import executor
from ..model.index import GenericIndex
from ..model.model import Model
from .ensemble import Ensemble, WeightedScenario

__all__ = ["EvaluationResult", "Evaluation"]


def _validate_scenarios(
    non_axis_abstract: list[GenericIndex],
    scenarios: list[WeightedScenario],
) -> None:
    """Raise ValueError if any non-axis abstract index is unresolved in any scenario."""
    for i, (_, assignments) in enumerate(scenarios):
        unresolved = [idx for idx in non_axis_abstract if idx not in assignments]
        if unresolved:
            names = ", ".join(getattr(idx, "name", repr(idx)) for idx in unresolved)
            raise ValueError(f"Scenario {i}: abstract index(es) not resolved: {names}")


class EvaluationResult:
    """Result of :meth:`Evaluation.evaluate`.

    Wraps the executor :class:`~executor.State` and provides typed access to
    node values and weighted marginalization over the scenario dimension.

    Parameters
    ----------
    state:
        The executor state after evaluation.
    weights:
        Scenario weights, shape ``(S,)``.
    axes:
        The axis arrays passed to :meth:`Evaluation.evaluate`, mapping each
        axis index to its 1-D value array.  Empty dict for 1-D mode.
    """

    def __init__(
        self,
        state: executor.State,
        weights: np.ndarray,
        axes: dict[GenericIndex, np.ndarray],
    ) -> None:
        self._state = state
        self._weights = weights
        self._axes = axes

    @property
    def weights(self) -> np.ndarray:
        """Scenario weights, shape ``(S,)``."""
        return self._weights

    @property
    def axes(self) -> dict[GenericIndex, np.ndarray]:
        """Axis value arrays as passed to :meth:`Evaluation.evaluate`."""
        return self._axes

    @property
    def full_shape(self) -> tuple[int, ...]:
        """Shape ``(N_0, ..., N_k, S)`` of a fully-broadcast result array."""
        return tuple(arr.size for arr in self._axes.values()) + (self._weights.size,)

    def __getitem__(self, index: GenericIndex) -> np.ndarray:
        """Return the raw result array for *index* (not yet broadcast to full shape)."""
        return self._state.values[index.node]

    def marginalize(self, index: GenericIndex) -> np.ndarray:
        """Broadcast *index*'s result to :attr:`full_shape` then contract with weights.

        Returns an array of shape ``(N_0, ..., N_k)`` — the scenario dimension
        is contracted.  In 1-D mode (no axes) returns a scalar.
        """
        arr = np.broadcast_to(self._state.values[index.node], self.full_shape)
        return np.tensordot(arr, self._weights, axes=([-1], [0]))


class Evaluation:
    """Bridge between a :class:`~dt_model.model.model.Model` and the engine.

    Given a model and a list of weighted scenarios, :meth:`evaluate` builds
    the engine substitution dict, runs :func:`executor.evaluate_nodes`, and
    returns an :class:`EvaluationResult`.

    This class knows nothing about grids, presence variables, sustainability,
    or constraints — all domain-specific logic lives in subclasses or
    vertical-specific wrappers.

    Parameters
    ----------
    model:
        The model to evaluate.
    """

    def __init__(self, model: Model) -> None:
        self.model = model

    def evaluate(
        self,
        scenarios: Ensemble,
        nodes_of_interest: list[GenericIndex] | None = None,
        *,
        axes: dict[GenericIndex, np.ndarray] | None = None,
        functions: dict[str, executor.Functor] | None = None,
    ) -> EvaluationResult:
        """Evaluate *nodes_of_interest* over the given scenarios.

        Parameters
        ----------
        scenarios:
            An :class:`~dt_model.simulation.ensemble.Ensemble` (any iterable
            of ``(weight, assignments)`` pairs).  Non-axis abstract indexes of
            the model must appear in every ``assignments`` dict.
        nodes_of_interest:
            Indexes to evaluate.  Transitive dependencies are resolved
            automatically via :func:`linearize.forest`.  Defaults to all
            indexes in the model when ``None``.
        axes:
            Optional grid axes for multi-dimensional evaluation.  Maps each
            axis :class:`~dt_model.model.index.GenericIndex` to a 1-D numpy
            array of values.  When provided:

            * Each axis index ``idx`` at position ``i`` contributes a
              substitution array of shape ``(1, …, N_i, …, 1, 1)`` where
              ``N_i = arr.size`` and the non-unit dimension is at position
              ``i`` (zero-indexed).
            * Each non-axis abstract index contributes a substitution derived
              from the scenario assignments.
            * Result arrays broadcast to shape ``(N_0, N_1, …, S)``; use
              :meth:`EvaluationResult.marginalize` to contract the scenario
              dimension.
        functions:
            Optional user-defined functions passed to the executor.  Keys are
            the names referenced by :class:`~graph.function_call` nodes;
            values are :class:`~executor.Functor` callables.

        Returns
        -------
        EvaluationResult
            Typed result wrapper.  Node values can be retrieved via
            ``result[index]``; weighted expectations via
            ``result.marginalize(index)``.

        Raises
        ------
        ValueError
            If any non-axis abstract index is not resolved in a scenario.
        """
        abstract = self.model.abstract_indexes()
        scenarios_list = list(scenarios)
        axes = axes or {}

        if nodes_of_interest is None:
            nodes_of_interest = list(self.model.indexes)

        axis_set = set(axes.keys())
        non_axis_abstract = [idx for idx in abstract if idx not in axis_set]

        _validate_scenarios(non_axis_abstract, scenarios_list)

        n_axes = len(axes)
        S = len(scenarios_list)
        c_subs: dict[graph.Node, np.ndarray] = {}

        # Axis indexes: shape (1, …, N_i, …, 1, 1) — N_i at position i.
        for i, (idx, arr) in enumerate(axes.items()):
            shape = [1] * (n_axes + 1)
            shape[i] = arr.size
            c_subs[idx.node] = arr.reshape(shape)

        # Non-axis abstract indexes: stack scenario values.
        # In grid mode (n_axes > 0) reshape to (1, …, 1, S) so they broadcast
        # against the axis dimensions.  In 1-D mode (n_axes == 0) preserve the
        # stacked shape as-is (e.g. (S, T) for timeseries-shaped values).
        for idx in non_axis_abstract:
            values = [assignments[idx] for _, assignments in scenarios_list]
            stacked = np.asarray(values)
            if n_axes > 0:
                stacked = stacked.reshape([1] * n_axes + [S])
            c_subs[idx.node] = stacked

        weights = np.array([w for w, _ in scenarios_list])
        actual_nodes = [idx.node for idx in nodes_of_interest]
        state = executor.State(c_subs, functions=functions or {})
        executor.evaluate_nodes(state, *linearize.forest(*actual_nodes))
        return EvaluationResult(state, weights, axes)
