"""Generic model evaluation."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np

from ..engine.frontend import graph, linearize
from ..engine.numpybackend import executor
from ..model.model import Model
from ..model.index import GenericIndex

WeightedScenario = tuple[float, dict[GenericIndex, Any]]
"""A weighted scenario maps each abstract index to a concrete value.

The first element is the scenario weight (probability); the second is a
mapping from each abstract index to its concrete value for this scenario.
Together a list of ``WeightedScenario`` objects defines a discrete
probability distribution over instantiations of an abstract model.
"""


class Evaluation:
    """Bridge between a :class:`~dt_model.model.model.Model` and the engine.

    Given a model and a list of weighted scenarios, :meth:`evaluate` builds
    the engine substitution dict, runs :func:`executor.evaluate_nodes`, and
    returns the resulting :class:`executor.State`.

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
        scenarios: Iterable[WeightedScenario],
        nodes_of_interest: list[graph.Node],
        *,
        axes: dict[GenericIndex, np.ndarray] | None = None,
        functions: dict[str, executor.Functor] | None = None,
    ) -> executor.State:
        """Evaluate *nodes_of_interest* over the given scenarios.

        Parameters
        ----------
        scenarios:
            An iterable of ``(weight, assignments)`` pairs.  When *axes* is
            ``None``, every abstract index of the model must appear in every
            ``assignments`` dict.  When *axes* is provided, indexes in *axes*
            are excluded from this requirement — they are provided as grid
            arrays instead.
        nodes_of_interest:
            Graph nodes to evaluate.  Transitive dependencies are resolved
            automatically via :func:`linearize.forest`.
        axes:
            Optional grid axes for multi-dimensional evaluation.  Maps each
            axis :class:`~dt_model.symbols.index.GenericIndex` to a 1-D
            numpy array of values.  When provided:

            * Each axis index ``idx`` at position ``i`` contributes a
              substitution array of shape ``(1, …, N_i, …, 1, 1)`` where
              ``N_i = arr.size`` and the non-unit dimension is at position
              ``i`` (zero-indexed).
            * Each non-axis abstract index contributes a substitution array
              of shape ``(1, …, 1, S)`` where ``S`` is the number of
              scenarios (the last dimension).
            * Result arrays therefore broadcast to shape
              ``(N_0, N_1, …, S)``; the caller marginalises over the last
              dimension using :func:`numpy.tensordot`.
        functions:
            Optional user-defined functions passed to the executor.  Keys are
            the names referenced by :class:`~graph.function_call` nodes;
            values are :class:`~executor.Functor` callables.

        Returns
        -------
        executor.State
            The engine state after evaluation.  Node values can be retrieved
            via ``state.values[node]``.

        Raises
        ------
        ValueError
            If any abstract index is not resolved in a scenario.
        """
        abstract = self.model.abstract_indexes()
        scenarios_list = list(scenarios)

        if axes is None:
            # 1-D batch mode: every abstract index must be in every scenario.
            for i, (_, assignments) in enumerate(scenarios_list):
                unresolved = [idx for idx in abstract if idx not in assignments]
                if unresolved:
                    names = ", ".join(getattr(idx, "name", repr(idx)) for idx in unresolved)
                    raise ValueError(
                        f"Scenario {i}: abstract index(es) not resolved: {names}"
                    )

            # Build substitution dict: stack scenario values into batch arrays.
            c_subs: dict[graph.Node, np.ndarray] = {}

            # One row per scenario for each abstract index.
            for idx in abstract:
                values = [assignments[idx] for _, assignments in scenarios_list]
                c_subs[idx.node] = np.asarray(values)

        else:
            # Grid mode: axis indexes are provided as dense arrays; all other
            # abstract indexes come from scenarios.
            n_axes = len(axes)
            axis_set = set(axes.keys())
            non_axis_abstract = [idx for idx in abstract if idx not in axis_set]

            # Validate: every non-axis abstract index must appear in each scenario.
            for i, (_, assignments) in enumerate(scenarios_list):
                unresolved = [idx for idx in non_axis_abstract if idx not in assignments]
                if unresolved:
                    names = ", ".join(getattr(idx, "name", repr(idx)) for idx in unresolved)
                    raise ValueError(
                        f"Scenario {i}: abstract index(es) not resolved: {names}"
                    )

            S = len(scenarios_list)
            c_subs = {}

            # Axis indexes: shape (1, …, N_i, …, 1, 1) — N_i at position i.
            for i, (idx, arr) in enumerate(axes.items()):
                shape = [1] * (n_axes + 1)
                shape[i] = arr.size
                c_subs[idx.node] = arr.reshape(shape)

            # Non-axis abstract indexes: shape (1, …, 1, S).
            for idx in non_axis_abstract:
                values = [assignments[idx] for _, assignments in scenarios_list]
                shape = [1] * n_axes + [S]
                c_subs[idx.node] = np.asarray(values).reshape(shape)

        state = executor.State(c_subs, functions=functions or {})
        executor.evaluate_nodes(state, *linearize.forest(*nodes_of_interest))
        return state
