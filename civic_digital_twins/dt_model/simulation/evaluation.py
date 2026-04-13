"""Generic model evaluation."""

import warnings
from collections.abc import Mapping

import numpy as np

from ..engine.frontend import graph, linearize
from ..engine.numpybackend import executor
from ..model.axis import ENSEMBLE, PARAMETER, Axis
from ..model.index import GenericIndex
from ..model.model import Model
from ..model.model_variant import ModelVariant
from .ensemble import AxisEnsemble, Ensemble, WeightedScenario

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


class _LegacyEnsembleAdapter:
    """Adapt ``Iterable[WeightedScenario]`` to :class:`AxisEnsemble`.

    Materialises the scenario list into batched arrays matching the
    ``AxisEnsemble`` shape contract so that the single batched evaluation
    path can handle both legacy and canonical inputs.
    """

    def __init__(
        self,
        scenarios: list[WeightedScenario],
        non_param_abstract: list[GenericIndex],
    ) -> None:
        self._axis = Axis("_ensemble", ENSEMBLE)
        self._weights = np.array([w for w, _ in scenarios])
        self._assignments: dict[GenericIndex, np.ndarray] = {}
        for idx in non_param_abstract:
            values = [assignments[idx] for _, assignments in scenarios]
            # Normalize: 1-element array assignments (common when values come
            # from DistributionEnsemble.__iter__) are unwrapped to scalars so
            # that np.asarray produces shape (S,) rather than (S, 1).
            normalized = [v.flat[0] if isinstance(v, np.ndarray) and v.size == 1 else v for v in values]
            self._assignments[idx] = np.asarray(normalized)  # shape (S,) or (S, T)

    @property
    def ensemble_axes(self) -> tuple[Axis, ...]:
        return (self._axis,)

    @property
    def ensemble_weights(self) -> tuple[np.ndarray, ...]:
        return (self._weights,)

    def assignments(self) -> Mapping[GenericIndex, np.ndarray]:
        return self._assignments


class EvaluationResult:
    """Result of :meth:`Evaluation.evaluate`.

    Wraps the executor :class:`~executor.State` and provides typed access to
    node values and weighted marginalization over ENSEMBLE and PARAMETER axes.

    Parameters
    ----------
    state:
        The executor state after evaluation.
    axis_layout:
        Maps each :class:`~dt_model.model.axis.Axis` to its numpy dimension
        position in result arrays.
    parameter_arrays:
        The value arrays passed in ``parameters=``, keyed by index.  Used by
        :meth:`parameter_values_for`.  Empty dict when no PARAMETER axes.
    axis_sizes:
        Maps each :class:`~dt_model.model.axis.Axis` to its size.
    factorized_weights:
        Per-ENSEMBLE-axis weight vectors.
    """

    def __init__(
        self,
        state: executor.State,
        axis_layout: dict[Axis, int],
        parameter_arrays: dict[GenericIndex, np.ndarray],
        axis_sizes: dict[Axis, int] | None = None,
        factorized_weights: dict[Axis, np.ndarray] | None = None,
    ) -> None:
        self._state = state
        self._axis_layout = axis_layout
        self._parameter_arrays = parameter_arrays
        self._axis_sizes: dict[Axis, int] = axis_sizes or {}
        self._factorized_weights: dict[Axis, np.ndarray] = factorized_weights or {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def weights(self) -> np.ndarray:
        """Scenario weight array.

        Returns the joint weight array (outer product of factorized per-axis
        weights).  Returns an empty array when there are no ENSEMBLE axes.
        """
        if not self._factorized_weights:
            return np.empty(0)
        joint: np.ndarray = next(iter(self._factorized_weights.values()))
        for w in list(self._factorized_weights.values())[1:]:
            joint = np.multiply.outer(joint, w)
        return joint

    @property
    def axes(self) -> dict[GenericIndex, np.ndarray]:
        """Deprecated. Use :attr:`parameter_values` instead."""
        warnings.warn(
            "'result.axes' is deprecated; use 'result.parameter_values'.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._parameter_arrays

    @property
    def parameter_values(self) -> dict[GenericIndex, np.ndarray]:
        """Parameter value arrays, keyed by the index passed in ``parameters=``."""
        return self._parameter_arrays

    def parameter_values_for(self, index: GenericIndex) -> np.ndarray:
        """Return the value array for a specific PARAMETER index.

        Parameters
        ----------
        index:
            An index that was passed in ``parameters=`` to
            :meth:`Evaluation.evaluate`.

        Raises
        ------
        KeyError
            If *index* was not a PARAMETER axis in this result.
        """
        return self._parameter_arrays[index]

    @property
    def full_shape(self) -> tuple[int, ...]:
        """Shape of a fully-broadcast result array.

        Returns ``(*PARAMETER, *ENSEMBLE)`` sizes in axis-layout order.
        """
        n_dims = len(self._axis_layout)
        if n_dims == 0:
            return ()
        shape: list[int] = [0] * n_dims
        for ax, pos in self._axis_layout.items():
            shape[pos] = self._axis_sizes[ax]
        return tuple(shape)

    def __getitem__(self, index: GenericIndex) -> np.ndarray:
        """Return the result array for *index*."""
        return np.asarray(self._state.values[index.node])

    def marginalize(self, index: GenericIndex) -> np.ndarray:
        """Contract ENSEMBLE axes using their weights and return the result.

        Uses the explicit axis layout — no shape heuristics.  After
        contracting all ENSEMBLE axes the result shape is
        ``(*PARAMETER_sizes, *DOMAIN_dims)``; trailing size-1 dims that lie
        beyond the PARAMETER positions are squeezed away (they are internal
        DOMAIN placeholders, not meaningful dimensions).
        """
        arr = np.asarray(self._state.values[index.node])

        ensemble_entries = sorted(
            [(ax, pos) for ax, pos in self._axis_layout.items() if ax.role == ENSEMBLE],
            key=lambda t: t[1],
            reverse=True,
        )

        n_params = len(self._parameter_arrays)

        if not ensemble_entries:
            return self._squeeze_domain(arr, n_params)

        for ax, pos in ensemble_entries:
            # evaluate() guarantees shape[pos] is either S (ENSEMBLE-touched
            # nodes) or 1 (injected singleton for non-touched nodes).
            # For the singleton case: weighted average of S identical copies
            # (weights summing to 1) equals the value itself — squeeze directly.
            if arr.shape[pos] == 1:
                arr = arr.squeeze(axis=pos)
            else:
                arr = np.average(arr, weights=self._factorized_weights[ax], axis=pos)

        return self._squeeze_domain(arr, n_params)

    @staticmethod
    def _squeeze_domain(arr: np.ndarray, n_params: int) -> np.ndarray:
        """Squeeze size-1 dims beyond the first *n_params* dimensions.

        PARAMETER dims (positions 0..n_params-1) are preserved even if
        size 1.  Trailing size-1 DOMAIN placeholder dims are removed so
        that pure-scalar results are returned as 0-d arrays rather than
        shape ``(1,)`` arrays.

        Known limitation — T=1 timeseries
        ----------------------------------
        When a model contains timeseries nodes, ``evaluate()`` appends a
        trailing size-1 placeholder dimension to every scalar substitution
        so that scalars broadcast correctly against ``(T,)`` arrays.  After
        ENSEMBLE contraction in :meth:`marginalize`, this placeholder is
        indistinguishable from a genuine length-1 timeseries trailing dim,
        and both are squeezed away here.  As a result,
        ``marginalize(ts)`` for a ``TimeseriesIndex`` of length 1 returns a
        0-d scalar rather than a shape-``(1,)`` array — the time axis is
        silently dropped.

        The root cause is the absence of explicit DOMAIN axis tracking: the
        engine has no way to tag "this dimension is the time axis" vs "this
        dimension is an internal broadcast placeholder".  A proper fix
        requires first-class DOMAIN axis support (see issue #157 and the
        D12/D13 design sprint).
        """
        domain_dims = tuple(i for i in range(n_params, arr.ndim) if arr.shape[i] == 1)
        if domain_dims:
            arr = np.squeeze(arr, axis=domain_dims)
        return arr


class Evaluation:
    """Bridge between a :class:`~dt_model.model.model.Model` and the engine.

    Given a model and an ensemble, :meth:`evaluate` builds the engine
    substitution dict, runs :func:`executor.evaluate_nodes`, and returns an
    :class:`EvaluationResult`.

    This class knows nothing about grids, presence variables, sustainability,
    or constraints — all domain-specific logic lives in subclasses or
    vertical-specific wrappers.

    Parameters
    ----------
    model:
        The model to evaluate.
    """

    def __init__(self, model: Model | ModelVariant) -> None:
        self.model = model

    def evaluate(
        self,
        scenarios: AxisEnsemble | Ensemble | None = None,
        nodes_of_interest: list[GenericIndex] | None = None,
        *,
        parameters: dict[GenericIndex, np.ndarray] | None = None,
        axes: dict[GenericIndex, np.ndarray] | None = None,
        ensemble: AxisEnsemble | Ensemble | None = None,
        functions: dict[str, executor.Functor] | None = None,
    ) -> EvaluationResult:
        """Evaluate *nodes_of_interest* over the given ensemble.

        Parameters
        ----------
        scenarios:
            Deprecated positional name for the ensemble argument.  Use
            ``ensemble=`` instead.  Only one of *scenarios* / *ensemble* may
            be supplied per call.
        nodes_of_interest:
            Indexes to evaluate.  Transitive dependencies are resolved
            automatically via :func:`linearize.forest`.  Defaults to all
            indexes in the model when ``None``.
        parameters:
            PARAMETER axes for multi-dimensional evaluation.  Maps each
            axis :class:`~dt_model.model.index.GenericIndex` to a 1-D numpy
            array of values.  When provided, each axis index ``idx`` at
            position ``i`` contributes a substitution array of shape
            ``(1, …, N_i, …, 1, 1)`` where ``N_i = arr.size``.
        axes:
            Deprecated alias for *parameters*.  Use ``parameters=`` instead.
        ensemble:
            The ensemble to evaluate.  Must be an :class:`AxisEnsemble`
            (canonical, batched) or a legacy ``Iterable[WeightedScenario]``
            (deprecated, emits :class:`DeprecationWarning`).  Pass ``None``
            for deterministic evaluation (no ENSEMBLE axes).
        functions:
            Optional user-defined functions passed to the executor.

        Returns
        -------
        EvaluationResult
            Typed result wrapper.

        Raises
        ------
        TypeError
            If both *scenarios* and *ensemble* are supplied, or both
            *axes* and *parameters* are supplied.
        ValueError
            If any non-parameter abstract index is not resolved in a scenario.
        """
        # --- resolve 'ensemble' from positional 'scenarios' arg ------------
        if scenarios is not None and ensemble is not None:
            raise TypeError("Cannot specify both 'scenarios' and 'ensemble'.")
        if scenarios is not None:
            warnings.warn(
                "The positional 'scenarios' argument is deprecated; use 'ensemble=' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            ensemble = scenarios

        # --- resolve 'parameters' from deprecated 'axes' arg ---------------
        if axes is not None and parameters is not None:
            raise TypeError("Cannot specify both 'axes' and 'parameters'.")
        if axes is not None:
            warnings.warn(
                "'axes' is deprecated; use 'parameters=' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            parameters = axes

        parameters = parameters or {}

        if nodes_of_interest is None:
            nodes_of_interest = list(self.model.indexes)

        abstract = self.model.abstract_indexes()
        param_set = set(parameters.keys())
        non_param_abstract = [idx for idx in abstract if idx not in param_set]

        # --- adapt legacy Iterable[WeightedScenario] to AxisEnsemble ------
        if ensemble is not None and not isinstance(ensemble, AxisEnsemble):
            warnings.warn(
                "Passing an iterable of WeightedScenario to 'evaluate()' is deprecated. "
                "Use an AxisEnsemble (e.g. DistributionEnsemble) instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            scenarios_list = list(ensemble)
            if scenarios_list:
                _validate_scenarios(non_param_abstract, scenarios_list)
                ensemble = _LegacyEnsembleAdapter(scenarios_list, non_param_abstract)
            else:
                ensemble = None  # empty list → deterministic

        # Pre-compute linearization and timeseries detection once.
        # These are reused for executor.evaluate_nodes() and BFS normalisation below.
        actual_nodes = [idx.node for idx in nodes_of_interest]
        linearized_nodes = linearize.forest(*actual_nodes)
        _has_timeseries = any(
            isinstance(node, (graph.timeseries_constant, graph.timeseries_placeholder)) for node in linearized_nodes
        )

        n_params = len(parameters)
        axis_layout: dict[Axis, int] = {}
        axis_sizes: dict[Axis, int] = {}
        factorized_weights: dict[Axis, np.ndarray] = {}
        c_subs: dict[graph.Node, np.ndarray] = {}
        param_nodes: list[graph.Node] = []

        # PARAMETER axes — positions 0..n_params-1.
        for i, (idx, arr) in enumerate(parameters.items()):
            ax = Axis(getattr(idx, "name", f"param_{i}"), PARAMETER)
            axis_layout[ax] = i
            axis_sizes[ax] = arr.size
            shape = [1] * n_params
            shape[i] = arr.size
            c_subs[idx.node] = arr.reshape(shape)
            param_nodes.append(idx.node)

        n_ensemble = 0
        ens_subs: dict[graph.Node, np.ndarray] = {}

        if ensemble is not None:
            ens_assignments = ensemble.assignments()
            n_ensemble = len(ensemble.ensemble_axes)
            for j, (ax, w) in enumerate(zip(ensemble.ensemble_axes, ensemble.ensemble_weights)):
                axis_layout[ax] = n_params + j
                axis_sizes[ax] = w.size
                factorized_weights[ax] = w
            for idx, batched in ens_assignments.items():
                # Prepend n_params PARAMETER singletons.
                # When the model contains timeseries nodes, also append a
                # trailing 1 for scalar (non-timeseries) assignments so they
                # broadcast with timeseries (T,) nodes:
                # (S, 1) × (T,) → (S, T).
                param_singletons = (1,) * n_params
                target = param_singletons + batched.shape
                if _has_timeseries and batched.ndim == n_ensemble:
                    target = target + (1,)
                ens_subs[idx.node] = np.reshape(batched, target)

        # Extend PARAMETER subs with trailing singleton dims:
        # - one per ENSEMBLE axis (so PARAMETER arrays broadcast against ENSEMBLE arrays),
        # - plus one extra timeseries placeholder when the graph has timeseries nodes
        #   (so (N, …, 1) broadcasts against a bare (T,) timeseries).
        extra_ts = 1 if _has_timeseries else 0
        trailing_singletons = (1,) * (n_ensemble + extra_ts)
        if trailing_singletons:
            for node in param_nodes:
                c_subs[node] = c_subs[node].reshape(c_subs[node].shape + trailing_singletons)

        c_subs.update(ens_subs)

        # Snapshot the substituted node keys before the executor mutates state.values
        # (executor.State takes c_subs by reference and adds all computed nodes into it).
        substituted_nodes: set[graph.Node] = set(c_subs)

        state = executor.State(c_subs, functions=functions or {})
        executor.evaluate_nodes(state, *linearized_nodes)

        # Normalise result arrays: every actual node must have shape
        # (*PARAMETER, *ENSEMBLE, *domain) with explicit singletons where
        # the node does not vary along an axis.
        #
        # All-or-nothing property: a node is either
        #   (a) downstream of some substitution → executor already produced
        #       the correct shape via numpy broadcasting; or
        #   (b) not downstream of any substitution → natural shape with zero
        #       leading dims (scalar () or bare timeseries (T,)).
        #
        # A single reshape prepending (n_total - arr.ndim) singletons handles
        # both subcases of (b), eliminating the need for a per-axis loop.
        n_full = n_params + n_ensemble
        n_total = n_full + extra_ts

        if n_total > 0:
            # all_touched: nodes transitively downstream of any substitution.
            # Use the pre-executor snapshot so that constant nodes evaluated by
            # the executor don't accidentally appear as "substituted".
            all_touched: set[graph.Node] = set(substituted_nodes)
            for node in linearized_nodes:
                if node in all_touched:
                    continue
                if any(dep in all_touched for dep in linearize._get_dependencies(node)):
                    all_touched.add(node)

            for node in actual_nodes:
                if node in all_touched or node not in state.values:
                    continue  # already has correct shape via substitution/broadcasting
                arr = np.asarray(state.values[node])
                # All-or-nothing invariant: untouched nodes must have natural
                # shape (zero leading dims: scalar or bare timeseries).
                assert arr.ndim in ({0, 1} if _has_timeseries else {0}), (
                    f"Untouched node {getattr(node, 'name', repr(node))!r}: "
                    f"unexpected ndim={arr.ndim} (has_timeseries={_has_timeseries})"
                )
                # Inject (n_total - arr.ndim) leading singletons.
                # Scalars get the full n_total prefix (including the timeseries
                # placeholder when _has_timeseries); timeseries nodes get n_full
                # singletons and keep their trailing (T,) dimension.
                n_inject = n_total - arr.ndim
                state.values[node] = arr.reshape((1,) * n_inject + arr.shape)

        # Post-normalisation invariant: every actual node must carry the
        # correct number of leading axis dims and valid sizes at each
        # axis position.  Guarded by __debug__ to avoid the loop overhead
        # when running with python -O.
        if __debug__:
            for node in actual_nodes:
                if node not in state.values:
                    continue
                arr = np.asarray(state.values[node])
                assert arr.ndim == n_total, (
                    f"Post-norm: node {getattr(node, 'name', repr(node))!r} ndim={arr.ndim}, expected {n_total}"
                )
                for ax, pos in axis_layout.items():
                    assert arr.shape[pos] in {1, axis_sizes[ax]}, (
                        f"Post-norm: node {getattr(node, 'name', repr(node))!r} "
                        f"axis {ax.name!r} at pos {pos}: shape[{pos}]={arr.shape[pos]}, "
                        f"expected 1 or {axis_sizes[ax]}"
                    )

        return EvaluationResult(
            state,
            axis_layout,
            parameters,
            axis_sizes=axis_sizes,
            factorized_weights=factorized_weights,
        )
