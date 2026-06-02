# SPDX-License-Identifier: Apache-2.0
"""Incremental and asynchronous evaluation handles for the engine control layer.

:class:`EvaluationHandle` wraps a pre-built :class:`~simulation.plan.EvaluationPlan`
together with the first evaluation result and provides :meth:`~EvaluationHandle.extend`
to grow the ensemble with additional Monte Carlo samples, merging new results into the
accumulated result without discarding prior computation.

:class:`AsyncEvaluationHandle` is a non-blocking variant: the evaluation runs on a
background thread and the handle exposes :meth:`~AsyncEvaluationHandle.poll` and
:meth:`~AsyncEvaluationHandle.get` for status checking and result retrieval.
Once the future resolves, :meth:`~AsyncEvaluationHandle.extend` works identically
to the synchronous base class.

Obtain instances via :meth:`EvaluationHandle.from_evaluation` (synchronous) or
:meth:`AsyncEvaluationHandle.from_evaluation` (asynchronous) rather than constructing
these classes directly.
"""

from __future__ import annotations

import concurrent.futures
from typing import Any

import numpy as np

from ..engine.frontend import graph
from ..engine.numpybackend import executor
from ..model.axis import ENSEMBLE, PARAMETER, Axis
from ..model.index import GenericIndex
from .ensemble import BatchDrawable, DistributionEnsemble, FrozenEnsemble
from .evaluation import Evaluation, EvaluationResult
from .plan import EvaluationPlan

# ---------------------------------------------------------------------------
# Module-level thread-pool executor (shared across all async evaluations)
# ---------------------------------------------------------------------------

# Allocated lazily on first use so that importing this module does not spawn
# threads for callers who never use async evaluation.
# Uses a ThreadPoolExecutor: the GIL is released during NumPy computation, so
# the main thread remains responsive while evaluation runs in the background.
_default_executor: concurrent.futures.ThreadPoolExecutor | None = None


def _get_default_executor() -> concurrent.futures.ThreadPoolExecutor:
    """Return (and lazily create) the module-level default ThreadPoolExecutor."""
    global _default_executor
    if _default_executor is None:
        _default_executor = concurrent.futures.ThreadPoolExecutor()
    return _default_executor


__all__ = ["AsyncEvaluationHandle", "EvaluationHandle"]


def _merge_results(
    r1: EvaluationResult,
    r2: EvaluationResult,
    plan: EvaluationPlan,
    *,
    merge_axis_name: str | None = None,
) -> EvaluationResult:
    """Merge two :class:`~simulation.evaluation.EvaluationResult` instances.

    Both results must have been produced by the same plan and with the same
    PARAMETER axes.  The merge concatenates node values along one ENSEMBLE
    axis and combines that axis's weights as a size-proportional mixture so
    that non-uniform weight schemes (e.g.
    :class:`~simulation.ensemble.CrossProductEnsemble`) are preserved.
    All other ENSEMBLE axes and their factorised weights pass through
    unchanged from *r1*.

    Parameters
    ----------
    r1:
        The accumulated result from previous evaluations.
    r2:
        The result of the latest :meth:`~simulation.evaluation.Evaluation.execute_plan`
        call.
    plan:
        The shared evaluation plan.  Used to enumerate the nodes of interest
        that must appear in the merged state.
    merge_axis_name:
        Name of the ENSEMBLE axis to grow.  Required when either result has
        more than one ENSEMBLE axis; ignored (but validated) for single-axis
        results.

    Returns
    -------
    EvaluationResult
        A new result whose node arrays are concatenated along the chosen
        ENSEMBLE axis (or kept as singletons when both inputs are singleton
        on that axis).

    Raises
    ------
    ValueError
        If either result has no ENSEMBLE axis, if *merge_axis_name* is absent
        when multiple ENSEMBLE axes are present, if the named axis is missing
        or at different positions in the two results, if the fixed ENSEMBLE
        axes differ, or if the PARAMETER axis layouts are incompatible.
    """
    # --- Collect ENSEMBLE axes ---
    ens_1 = [(ax, pos) for ax, pos in r1._axis_layout.items() if ax.role == ENSEMBLE]
    ens_2 = [(ax, pos) for ax, pos in r2._axis_layout.items() if ax.role == ENSEMBLE]

    if not ens_1 or not ens_2:
        raise ValueError(
            "_merge_results requires both results to have at least one ENSEMBLE axis; "
            f"got {len(ens_1)} in r1 and {len(ens_2)} in r2."
        )

    # --- Identify the growing axis ---
    if len(ens_1) == 1 and len(ens_2) == 1:
        grow_ax_1, ens_pos = ens_1[0]
        grow_ax_2, ens_pos2 = ens_2[0]
        if ens_pos != ens_pos2:
            raise ValueError(  # pragma: no cover
                f"ENSEMBLE axis position mismatch: r1 at dim {ens_pos}, r2 at dim {ens_pos2}."
            )
    else:
        if merge_axis_name is None:
            names = sorted(ax.name for ax, _ in ens_1)
            raise ValueError(
                f"_merge_results: both results have multiple ENSEMBLE axes {names}; "
                "specify merge_axis_name= to indicate which axis to grow."
            )
        grow_entry_1 = next(((ax, pos) for ax, pos in ens_1 if ax.name == merge_axis_name), None)
        grow_entry_2 = next(((ax, pos) for ax, pos in ens_2 if ax.name == merge_axis_name), None)
        if grow_entry_1 is None:
            raise ValueError(f"_merge_results: no ENSEMBLE axis named {merge_axis_name!r} in r1.")
        if grow_entry_2 is None:
            raise ValueError(f"_merge_results: no ENSEMBLE axis named {merge_axis_name!r} in r2.")
        grow_ax_1, ens_pos = grow_entry_1
        grow_ax_2, ens_pos2 = grow_entry_2
        if ens_pos != ens_pos2:
            raise ValueError(
                f"Growing ENSEMBLE axis {merge_axis_name!r} is at dim {ens_pos} in r1 but dim {ens_pos2} in r2."
            )

        # Validate the fixed ENSEMBLE axes match by name, position, and size.
        def _fixed_ens_sig(axes: list[tuple[Axis, int]], sizes: dict[Axis, int]) -> frozenset[tuple[str, int, int]]:
            return frozenset((ax.name, pos, sizes[ax]) for ax, pos in axes if ax.name != merge_axis_name)

        if _fixed_ens_sig(ens_1, r1._axis_sizes) != _fixed_ens_sig(ens_2, r2._axis_sizes):
            raise ValueError("_merge_results: fixed ENSEMBLE axis layouts differ between r1 and r2.")

    S1: int = r1._axis_sizes[grow_ax_1]
    S2: int = r2._axis_sizes[grow_ax_2]

    # --- Validate that PARAMETER axes are compatible ---
    def _param_sig(layout: dict[Axis, int], sizes: dict[Axis, int]) -> frozenset[tuple[str, object, int, int]]:
        return frozenset((ax.name, ax.role, pos, sizes[ax]) for ax, pos in layout.items() if ax.role != ENSEMBLE)

    if _param_sig(r1._axis_layout, r1._axis_sizes) != _param_sig(r2._axis_layout, r2._axis_sizes):
        raise ValueError(
            "_merge_results requires both results to have identical PARAMETER axis layouts. "
            "Ensure both were built from the same plan with the same 'parameters=' dict."
        )

    # --- Merge node values along the growing ENSEMBLE axis ---
    merged_values: dict[graph.Node, np.ndarray] = {}
    for idx in plan.nodes_of_interest:
        node = idx.node
        if __debug__ and (node not in r1._state.values or node not in r2._state.values):
            raise RuntimeError(  # pragma: no cover
                f"_merge_results: node {getattr(node, 'name', repr(node))!r} from "
                "plan.nodes_of_interest is missing in one of the results. "
                "This is a bug — both results must be produced by the same plan."
            )
        v1 = np.asarray(r1._state.values[node])
        v2 = np.asarray(r2._state.values[node])

        while v1.ndim <= ens_pos:
            v1 = v1[np.newaxis]  # pragma: no cover
        while v2.ndim <= ens_pos:
            v2 = v2[np.newaxis]  # pragma: no cover

        if v1.shape[ens_pos] == 1 and v2.shape[ens_pos] == 1 and np.array_equal(v1, v2):
            merged_values[node] = v1
        else:
            if v1.shape[ens_pos] == 1:
                bcast = v1.shape[:ens_pos] + (S1,) + v1.shape[ens_pos + 1 :]
                v1 = np.broadcast_to(v1, bcast).copy()
            if v2.shape[ens_pos] == 1:
                bcast = v2.shape[:ens_pos] + (S2,) + v2.shape[ens_pos + 1 :]
                v2 = np.broadcast_to(v2, bcast).copy()
            merged_values[node] = np.concatenate([v1, v2], axis=ens_pos)

    # --- Build merged axis metadata ---
    # Fresh Axis object for the grown dimension (identity-based keys; old object
    # would carry a stale size and must not be reused).
    merged_grow_ax = Axis(grow_ax_1.name, ENSEMBLE)
    merged_axis_layout: dict[Axis, int] = {
        **{ax: pos for ax, pos in r1._axis_layout.items() if ax is not grow_ax_1},
        merged_grow_ax: ens_pos,
    }
    merged_axis_sizes: dict[Axis, int] = {
        **{ax: sz for ax, sz in r1._axis_sizes.items() if ax is not grow_ax_1},
        merged_grow_ax: S1 + S2,
    }

    # Size-proportional mixture for the growing axis; fixed ENSEMBLE axes
    # keep their factorised weights from r1 unchanged.
    w1 = r1._factorized_weights[grow_ax_1]
    w2 = r2._factorized_weights[grow_ax_2]
    alpha = S1 / (S1 + S2)
    merged_factorized_weights: dict[Axis, np.ndarray] = {
        **{ax: w for ax, w in r1._factorized_weights.items() if ax is not grow_ax_1},
        merged_grow_ax: np.concatenate([w1 * alpha, w2 * (1.0 - alpha)]),
    }

    merged_state = executor.State(merged_values)
    return EvaluationResult(
        merged_state,
        merged_axis_layout,
        r1._parameter_arrays,
        axis_sizes=merged_axis_sizes,
        factorized_weights=merged_factorized_weights,
        named_axis_values=r1._named_axis_values or None,
    )


def _merge_results_param_extend(
    r1: EvaluationResult,
    r2: EvaluationResult,
    plan: EvaluationPlan,
    param_idx: GenericIndex,
) -> EvaluationResult:
    """Merge two results by extending the PARAMETER axis for *param_idx*.

    Both results must have been produced by the **same ensemble** (identical
    ENSEMBLE axis size and position) and the same fixed PARAMETER axes.
    *r1*'s PARAMETER axis for *param_idx* has size ``P1``; *r2*'s has size
    ``P2``.  The merged result has size ``P1 + P2``.

    Parameters
    ----------
    r1:
        The accumulated result with the original parameter values.
    r2:
        The result evaluated at the extra parameter values using the same
        ensemble as *r1*.
    plan:
        The shared evaluation plan.
    param_idx:
        The index whose PARAMETER axis is being extended.

    Returns
    -------
    EvaluationResult
        A new result with node arrays concatenated along *param_idx*'s axis.

    Raises
    ------
    ValueError
        If either result lacks an ENSEMBLE axis, if the ENSEMBLE sizes differ,
        if *param_idx* has no PARAMETER axis in *r1*, or if the fixed
        PARAMETER axes do not match.
    """
    param_name: str = getattr(param_idx, "name", repr(param_idx))

    # --- Validate ENSEMBLE axes ---
    ens_1 = [(ax, pos) for ax, pos in r1._axis_layout.items() if ax.role == ENSEMBLE]
    ens_2 = [(ax, pos) for ax, pos in r2._axis_layout.items() if ax.role == ENSEMBLE]
    if len(ens_1) != 1 or len(ens_2) != 1:
        raise ValueError(
            "_merge_results_param_extend requires exactly one ENSEMBLE axis in each result; "
            f"got {len(ens_1)} in r1 and {len(ens_2)} in r2."
        )
    ax_ens, ens_pos = ens_1[0]
    _, ens_pos2 = ens_2[0]
    if ens_pos != ens_pos2:
        raise ValueError(f"ENSEMBLE axis position mismatch: r1 at dim {ens_pos}, r2 at dim {ens_pos2}.")
    S = r1._axis_sizes[ax_ens]
    S2_check = r2._axis_sizes[ens_2[0][0]]
    if S != S2_check:
        raise ValueError(
            f"_merge_results_param_extend requires identical ENSEMBLE sizes; got {S} vs {S2_check}. "
            "Both results must be produced by the same ensemble."
        )

    # --- Locate the growing PARAMETER axis in r1 ---
    grow_ax_1: Axis | None = next(
        (ax for ax, _ in r1._axis_layout.items() if ax.role == PARAMETER and ax.name == param_name),
        None,
    )
    if grow_ax_1 is None:
        raise ValueError(
            f"_merge_results_param_extend: no PARAMETER axis named {param_name!r} in r1. "
            "extra_parameters must contain indexes already present in the initial parameters= dict."
        )
    param_pos = r1._axis_layout[grow_ax_1]
    P1 = r1._axis_sizes[grow_ax_1]

    grow_ax_2: Axis | None = next(
        (ax for ax, _ in r2._axis_layout.items() if ax.role == PARAMETER and ax.name == param_name),
        None,
    )
    if grow_ax_2 is None:
        raise ValueError(f"_merge_results_param_extend: no PARAMETER axis named {param_name!r} in r2.")
    P2 = r2._axis_sizes[grow_ax_2]

    # --- Validate fixed PARAMETER axes ---
    def _fixed_sig(layout: dict[Axis, int], sizes: dict[Axis, int]) -> frozenset[tuple[str, int, int]]:
        return frozenset(
            (ax.name, pos, sizes[ax]) for ax, pos in layout.items() if ax.role == PARAMETER and ax.name != param_name
        )

    if _fixed_sig(r1._axis_layout, r1._axis_sizes) != _fixed_sig(r2._axis_layout, r2._axis_sizes):
        raise ValueError("_merge_results_param_extend: fixed PARAMETER axis layouts differ between r1 and r2.")

    # --- Concatenate node arrays along the growing PARAMETER axis ---
    merged_values: dict[graph.Node, np.ndarray] = {}
    for noi in plan.nodes_of_interest:
        node = noi.node
        v1 = np.asarray(r1._state.values[node])
        v2 = np.asarray(r2._state.values[node])

        # Ensure both arrays are at least (param_pos + 1)-dimensional.
        while v1.ndim <= param_pos:
            v1 = v1[np.newaxis]  # pragma: no cover
        while v2.ndim <= param_pos:
            v2 = v2[np.newaxis]  # pragma: no cover

        if v1.shape[param_pos] == 1 and v2.shape[param_pos] == 1:
            # Node is constant across the PARAMETER axis (e.g. ensemble-only
            # inputs that don't vary with the swept parameter).  Keep singleton.
            merged_values[node] = v1
        else:
            merged_values[node] = np.concatenate([v1, v2], axis=param_pos)

    # --- Build merged axis metadata ---
    # Fresh Axis identity for the grown dimension (old object would carry stale size).
    merged_grow_ax = Axis(param_name, PARAMETER)
    merged_axis_layout: dict[Axis, int] = {}
    merged_axis_sizes: dict[Axis, int] = {}
    for ax, pos in r1._axis_layout.items():
        if ax is grow_ax_1:
            merged_axis_layout[merged_grow_ax] = pos
            merged_axis_sizes[merged_grow_ax] = P1 + P2
        else:
            merged_axis_layout[ax] = pos
            merged_axis_sizes[ax] = r1._axis_sizes[ax]

    # Factorised weights: ENSEMBLE axis unchanged.
    merged_factorized_weights: dict[Axis, np.ndarray] = dict(r1._factorized_weights)

    # Concatenate the growing parameter's value array.
    merged_parameter_arrays: dict[GenericIndex, np.ndarray] = dict(r1._parameter_arrays)
    if param_idx in r1._parameter_arrays and param_idx in r2._parameter_arrays:
        merged_parameter_arrays[param_idx] = np.concatenate(
            [r1._parameter_arrays[param_idx], r2._parameter_arrays[param_idx]]
        )

    merged_state = executor.State(merged_values)
    return EvaluationResult(
        merged_state,
        merged_axis_layout,
        merged_parameter_arrays,
        axis_sizes=merged_axis_sizes,
        factorized_weights=merged_factorized_weights,
        named_axis_values=r1._named_axis_values or None,
    )


class EvaluationHandle:
    """Incremental evaluation handle for growing an ensemble in steps.

    Wraps a pre-built :class:`~simulation.plan.EvaluationPlan` together with
    its first evaluation result.  Each call to :meth:`extend` draws additional
    Monte Carlo samples, executes the plan against the new ensemble, and merges
    the resulting arrays into the accumulated result.

    All sample draws share the same :class:`numpy.random.Generator` so that
    the full sequence of samples is reproducible from a single seed.

    .. note::

        :meth:`extend` delegates ensemble growth to the stored
        *ensemble_recipe* via the :class:`~simulation.ensemble.BatchDrawable`
        protocol.  Any ensemble type that implements :meth:`draw_batch
        <simulation.ensemble.BatchDrawable.draw_batch>` can serve as the
        recipe — :class:`~simulation.ensemble.DistributionEnsemble`,
        :class:`~simulation.ensemble.CrossProductEnsemble`, and
        :class:`~simulation.ensemble.PartitionedEnsemble` all do.

    Obtain an instance via :meth:`EvaluationHandle.from_evaluation` rather than
    constructing this class directly.

    Parameters
    ----------
    evaluation:
        The :class:`~simulation.evaluation.Evaluation` that owns the plan.
    plan:
        The pre-built evaluation plan.
    result:
        The initial :class:`~simulation.evaluation.EvaluationResult`, or
        ``None`` when the result is not yet available (async path).
    rng:
        Shared random number generator.  Reused by every :meth:`extend` call.
    parameters:
        The PARAMETER axis dict passed to the initial execution (array-valued
        entries and callable-valued entries combined).
    parameter_axes:
        Named PARAMETER axes dict passed to the initial execution (from
        ``parameter_axes=``).  ``None`` when correlated axes were not used.
    ensemble:
        Frozen replay of the initial ensemble's sample draws, used by
        :meth:`extend` when *extra_parameters* is supplied.  ``None`` when
        the handle was constructed without an ensemble (e.g. in tests).
    ensemble_recipe:
        The live :class:`~simulation.ensemble.BatchDrawable` ensemble that was
        used to draw the initial samples.  Stored so that
        :meth:`_extend_ensemble_axis` can ask it to draw more samples without
        naming any concrete ensemble class or reaching into private evaluation
        state.  ``None`` when the handle was constructed without an ensemble.
    functions:
        Optional user-defined functions passed through to the executor.
    backend:
        The computation backend (currently only ``NumpyBackend`` is supported).
    """

    def __init__(
        self,
        *,
        evaluation: Evaluation,
        plan: EvaluationPlan,
        result: EvaluationResult | None,
        rng: np.random.Generator,
        parameters: dict[GenericIndex, np.ndarray],
        parameter_axes: dict[str, np.ndarray] | None = None,
        ensemble: FrozenEnsemble | None = None,
        ensemble_recipe: BatchDrawable | None = None,
        functions: dict[str, executor.Functor] | None = None,
        backend: type[executor.NumpyBackend] = executor.NumpyBackend,
    ) -> None:
        self._evaluation = evaluation
        self._plan = plan
        self._result = result
        self._rng = rng
        self._parameters = parameters
        self._parameter_axes = parameter_axes
        self._ensemble = ensemble
        self._ensemble_recipe = ensemble_recipe
        self._functions = functions
        self._backend = backend

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def result(self) -> EvaluationResult:
        """The current accumulated :class:`~simulation.evaluation.EvaluationResult`.

        Raises
        ------
        RuntimeError
            If the result has not yet been set (async path before resolution).
        """
        if self._result is None:
            raise RuntimeError("result is not yet available.")
        return self._result

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_evaluation(
        cls,
        evaluation: Evaluation,
        initial_ensemble_size: int,
        nodes_of_interest: list[GenericIndex] | None = None,
        *,
        parameters: dict[GenericIndex, Any] | None = None,
        parameter_axes: dict[str, np.ndarray] | None = None,
        strategy: str = "monolithic",
        rng: np.random.Generator | None = None,
        functions: dict[str, executor.Functor] | None = None,
        backend: type[executor.NumpyBackend] = executor.NumpyBackend,
    ) -> "EvaluationHandle":
        """Build a plan, run an initial ensemble, and return an incremental handle.

        The returned :class:`EvaluationHandle` holds the first result and can be
        extended with additional Monte Carlo samples via :meth:`extend` without
        discarding prior results.

        All sample draws (initial and extended) share the same
        :class:`numpy.random.Generator`, making the full sequence reproducible
        from a single seed.

        Parameters
        ----------
        evaluation:
            The :class:`~simulation.evaluation.Evaluation` to run.
        initial_ensemble_size:
            Number of scenarios in the first evaluation batch.
        nodes_of_interest:
            Indexes to evaluate.  Defaults to all indexes in the model.
        parameters:
            PARAMETER axes for multi-dimensional evaluation.
        parameter_axes:
            Named PARAMETER axes for correlated sweeps.
        strategy:
            Plan build strategy (``"monolithic"`` or ``"regional"``).
        rng:
            Random number generator for reproducibility.  When ``None``, a
            fresh :func:`numpy.random.default_rng` is created automatically.
        functions:
            Optional user-defined functions passed to the executor.
        backend:
            The computation backend (currently only ``NumpyBackend``).

        Returns
        -------
        EvaluationHandle
            Incremental handle wrapping the first result.
        """
        parameters = parameters or {}
        if rng is None:
            rng = np.random.default_rng()

        plan = evaluation.build_plan(nodes_of_interest, strategy=strategy)
        dist_ensemble = DistributionEnsemble(
            evaluation._scenario, initial_ensemble_size, rng=rng, exclude=frozenset(parameters)
        )
        # Draw samples once and freeze them so parameter-extension can reuse the
        # same scenarios without advancing the RNG a second time.
        ensemble = dist_ensemble.draw_batch(initial_ensemble_size, rng)
        result = evaluation.execute_plan(
            plan, ensemble, parameters=parameters, parameter_axes=parameter_axes, functions=functions, backend=backend
        )
        return cls(
            evaluation=evaluation,
            plan=plan,
            result=result,
            rng=rng,
            parameters=parameters,
            parameter_axes=parameter_axes,
            ensemble=ensemble,
            ensemble_recipe=dist_ensemble,
            functions=functions,
            backend=backend,
        )

    # ------------------------------------------------------------------
    # Private helpers — one primitive operation each
    # ------------------------------------------------------------------

    def _extend_ensemble_axis(self, ensemble_size: int, axis: str | None = None) -> None:
        """Draw *ensemble_size* new scenarios and merge along an ENSEMBLE axis.

        Parameters
        ----------
        ensemble_size:
            Number of new Monte Carlo samples (or samples per combo for
            :class:`~simulation.ensemble.CrossProductEnsemble`).
        axis:
            For multi-axis ensembles (e.g.
            :class:`~simulation.ensemble.PartitionedEnsemble`): name of the
            ENSEMBLE axis to grow.  ``None`` for single-axis ensembles.
        """
        assert self._ensemble_recipe is not None, (
            "EvaluationHandle._extend_ensemble_axis() requires an ensemble_recipe. "
            "Obtain the handle via EvaluationHandle.from_evaluation() to enable ensemble extension."
        )
        new_batch = self._ensemble_recipe.draw_batch(ensemble_size, self._rng, axis=axis)
        if axis is not None and self._ensemble is not None and len(self._ensemble.ensemble_axes) > 1:
            # Multi-axis: pair fresh samples for the target axis with the existing
            # frozen samples for all other axes before executing the plan.
            execution_ensemble = self._ensemble.with_replaced_axis(axis, new_batch)
        else:
            execution_ensemble = new_batch
        new_result = self._evaluation.execute_plan(
            self._plan,
            execution_ensemble,
            parameters=self._parameters,
            parameter_axes=self._parameter_axes,
            functions=self._functions,
            backend=self._backend,
        )
        assert self._result is not None
        self._result = _merge_results(self._result, new_result, self._plan, merge_axis_name=axis)
        if self._ensemble is not None:
            if axis is not None and len(self._ensemble.ensemble_axes) > 1:
                self._ensemble = self._ensemble.concat_along(axis, new_batch)
            else:
                self._ensemble = self._ensemble.concat(new_batch)

    def _extend_param_axis(self, param_idx: GenericIndex, extra_vals: np.ndarray) -> None:
        """Re-run the stored ensemble at *extra_vals* and merge along the PARAMETER axis."""
        assert self._result is not None
        assert self._ensemble is not None
        r_extra = self._evaluation.execute_plan(
            self._plan,
            self._ensemble,
            parameters={**self._parameters, param_idx: extra_vals},
            parameter_axes=self._parameter_axes,
            functions=self._functions,
            backend=self._backend,
        )
        self._result = _merge_results_param_extend(self._result, r_extra, self._plan, param_idx)
        self._parameters = {
            **self._parameters,
            param_idx: np.concatenate([self._parameters[param_idx], extra_vals]),
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extend(
        self,
        ensemble_size: int = 0,
        *,
        extra_ensemble: dict[str, int] | None = None,
        extra_parameters: dict[GenericIndex, np.ndarray] | None = None,
    ) -> EvaluationResult:
        """Grow the ensemble and/or the PARAMETER grid and merge.

        ENSEMBLE extension
        -----------------
        Two mutually exclusive forms are accepted:

        * **Shorthand** — ``ensemble_size=N`` (positional).  Draws *N* new
          Monte Carlo samples on the single ENSEMBLE axis of the recipe.
          Raises :class:`ValueError` when the recipe has more than one axis;
          use *extra_ensemble* instead::

              handle.extend(50)

        * **Explicit dict** — ``extra_ensemble={"axis_name": N, ...}``.
          Extends one or more named ENSEMBLE axes, processed sequentially.
          Each entry calls :meth:`_extend_ensemble_axis` and the results are
          merged with the appropriate :meth:`_merge_results` call.  Use this
          form for :class:`~simulation.ensemble.PartitionedEnsemble`::

              handle.extend(extra_ensemble={"unc": 5, "default": 3})

        Supplying both *ensemble_size* > 0 and *extra_ensemble* at the same
        time raises :class:`ValueError`.

        PARAMETER extension
        ------------------
        ``extra_parameters={idx: extra_array, ...}`` re-runs the plan over
        new parameter values using the **same** frozen sample draws and merges
        along each PARAMETER axis.  Every key must already be present in the
        ``parameters=`` dict passed to
        :meth:`~simulation.evaluation.Evaluation.evaluate_incremental`.
        Multiple entries are processed one axis at a time::

            handle.extend(extra_parameters={x1: np.array([0.9, 1.0])})

        Combined
        --------
        ENSEMBLE and PARAMETER extension may be combined freely::

            # single-axis recipe + parameter extension
            handle.extend(50, extra_parameters={x1: vals})

            # PartitionedEnsemble: extend one axis + parameter extension
            handle.extend(
                extra_ensemble={"unc": 5},
                extra_parameters={x1: vals},
            )

        Parameters
        ----------
        ensemble_size:
            Shorthand for single-axis ENSEMBLE extension.  Mutually exclusive
            with *extra_ensemble*.
        extra_ensemble:
            Explicit multi-axis ENSEMBLE extension dict ``{axis_name: N}``.
            Mutually exclusive with *ensemble_size* > 0.
        extra_parameters:
            Dict ``{idx: extra_array, ...}`` extending the PARAMETER sweep for
            each *idx*.  Every key must already be present in the
            ``parameters=`` dict passed to
            :meth:`EvaluationHandle.from_evaluation`.

        Returns
        -------
        EvaluationResult
            The updated accumulated result (same object as :attr:`result`
            after the merge).

        Raises
        ------
        ValueError
            If both *ensemble_size* > 0 and *extra_ensemble* are supplied, if
            any key in *extra_ensemble* is not a known ENSEMBLE axis name, or
            if any key in *extra_parameters* is not in the original
            ``parameters=`` dict.
        RuntimeError
            If the handle has no stored ensemble and *extra_parameters* is
            supplied.
        """
        if ensemble_size > 0 and extra_ensemble is not None:
            raise ValueError(
                "extend(): ensemble_size and extra_ensemble are mutually exclusive. "
                "Use ensemble_size for single-axis recipes; use extra_ensemble for "
                "named-axis (PartitionedEnsemble) extension."
            )
        if extra_parameters is None and extra_ensemble is None and ensemble_size <= 0:
            return self._result  # type: ignore[return-value]

        assert self._result is not None, (
            "EvaluationHandle.extend() called with _result=None — "
            "this is a bug: either the handle was constructed incorrectly or "
            "AsyncEvaluationHandle.extend() failed to call _resolve() first."
        )

        if extra_parameters is not None:
            bad = [k for k in extra_parameters if k not in self._parameters]
            if bad:
                names = ", ".join(repr(getattr(k, "name", repr(k))) for k in bad)
                raise ValueError(
                    f"extend(extra_parameters=): {names} not in the original parameters= dict. "
                    "Only existing parameter axes can be extended."
                )
            if self._ensemble is None:
                raise RuntimeError(
                    "EvaluationHandle has no stored ensemble; extra_parameters extension is unavailable. "
                    "Obtain the handle via evaluate_incremental() to enable this feature."
                )

        if ensemble_size > 0:
            self._extend_ensemble_axis(ensemble_size, axis=None)
        if extra_ensemble is not None:
            for axis_name, size in extra_ensemble.items():
                self._extend_ensemble_axis(size, axis=axis_name)
        if extra_parameters is not None:
            for idx, vals in extra_parameters.items():
                self._extend_param_axis(idx, np.asarray(vals))

        return self._result


class AsyncEvaluationHandle(EvaluationHandle):
    """Non-blocking evaluation handle backed by a :class:`concurrent.futures.Future`.

    The evaluation runs on a background thread.  Use :meth:`poll` to check
    completion without blocking, or :meth:`get` to wait for the result.
    Once the future resolves, :meth:`extend` works identically to
    :class:`EvaluationHandle`.

    See the :class:`EvaluationHandle` class note for the
    :class:`~simulation.ensemble.BatchDrawable` contract that the
    *ensemble_recipe* must satisfy.

    Obtain an instance via :meth:`AsyncEvaluationHandle.from_evaluation` rather than
    constructing this class directly.

    Parameters
    ----------
    future:
        A :class:`concurrent.futures.Future` that will resolve to an
        :class:`~simulation.evaluation.EvaluationResult`.
    evaluation:
        The :class:`~simulation.evaluation.Evaluation` that owns the plan.
    plan:
        The pre-built evaluation plan.
    rng:
        Shared random number generator reused by :meth:`extend`.
    parameters:
        The PARAMETER axis dict passed to the initial execution.
    parameter_axes:
        Named PARAMETER axes dict passed to the initial execution.
    ensemble:
        Frozen replay of the initial ensemble's sample draws (see
        :class:`EvaluationHandle`).
    ensemble_recipe:
        The live ensemble used to draw the initial samples (see
        :class:`EvaluationHandle`).
    functions:
        Optional user-defined functions passed through to the executor.
    backend:
        The computation backend.
    """

    def __init__(
        self,
        *,
        future: concurrent.futures.Future[EvaluationResult],
        evaluation: Evaluation,
        plan: EvaluationPlan,
        rng: np.random.Generator,
        parameters: dict[GenericIndex, np.ndarray],
        parameter_axes: dict[str, np.ndarray] | None,
        ensemble: FrozenEnsemble | None,
        ensemble_recipe: BatchDrawable | None,
        functions: dict[str, executor.Functor] | None,
        backend: type[executor.NumpyBackend],
    ) -> None:
        super().__init__(
            evaluation=evaluation,
            plan=plan,
            result=None,  # not yet available; resolved lazily by _resolve()
            rng=rng,
            parameters=parameters,
            parameter_axes=parameter_axes,
            ensemble=ensemble,
            ensemble_recipe=ensemble_recipe,
            functions=functions,
            backend=backend,
        )
        self._future = future

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_evaluation(
        cls,
        evaluation: Evaluation,
        initial_ensemble_size: int,
        nodes_of_interest: list[GenericIndex] | None = None,
        *,
        parameters: dict[GenericIndex, Any] | None = None,
        parameter_axes: dict[str, np.ndarray] | None = None,
        strategy: str = "monolithic",
        rng: np.random.Generator | None = None,
        functions: dict[str, executor.Functor] | None = None,
        backend: type[executor.NumpyBackend] = executor.NumpyBackend,
        pool: concurrent.futures.Executor | None = None,
    ) -> "AsyncEvaluationHandle":
        """Submit an evaluation to a background thread and return immediately.

        Mirrors :meth:`EvaluationHandle.from_evaluation` but runs the initial
        :meth:`~simulation.evaluation.Evaluation.execute_plan` call on a thread
        from *pool* (or the module-level default
        :class:`~concurrent.futures.ThreadPoolExecutor`) so that the caller is
        not blocked.  The returned handle can be polled for status or awaited
        for its result.

        Once the future resolves, :meth:`extend` works identically to
        :class:`EvaluationHandle`.

        Parameters
        ----------
        evaluation:
            The :class:`~simulation.evaluation.Evaluation` to run.
        initial_ensemble_size:
            Number of scenarios in the first evaluation batch.
        nodes_of_interest:
            Indexes to evaluate.  Defaults to all indexes in the model.
        parameters:
            PARAMETER axes for multi-dimensional evaluation.
        parameter_axes:
            Named PARAMETER axes for correlated sweeps.
        strategy:
            Plan build strategy (``"monolithic"`` or ``"regional"``).
        rng:
            Random number generator for reproducibility.  When ``None``, a
            fresh :func:`numpy.random.default_rng` is created automatically.
        functions:
            Optional user-defined functions passed to the executor.
        backend:
            The computation backend (currently only ``NumpyBackend``).
        pool:
            :class:`concurrent.futures.Executor` to submit the work to.
            Defaults to a module-level :class:`~concurrent.futures.ThreadPoolExecutor`
            shared across calls (created lazily on first use).

        Returns
        -------
        AsyncEvaluationHandle
            Handle wrapping the in-flight future.  Call :meth:`get` to block
            for the result or :meth:`poll` to check without blocking.
        """
        parameters = parameters or {}
        if rng is None:
            rng = np.random.default_rng()

        plan = evaluation.build_plan(nodes_of_interest, strategy=strategy)
        dist_ensemble = DistributionEnsemble(
            evaluation._scenario, initial_ensemble_size, rng=rng, exclude=frozenset(parameters)
        )
        # Draw samples in the main thread before submitting so the frozen batch
        # can be shared safely with the background thread.
        ensemble = dist_ensemble.draw_batch(initial_ensemble_size, rng)
        _exec = pool or _get_default_executor()
        future: concurrent.futures.Future[EvaluationResult] = _exec.submit(
            evaluation.execute_plan,
            plan,
            ensemble,
            parameters=parameters,
            parameter_axes=parameter_axes,
            functions=functions,
            backend=backend,
        )
        return cls(
            future=future,
            evaluation=evaluation,
            plan=plan,
            rng=rng,
            parameters=parameters,
            parameter_axes=parameter_axes,
            ensemble=ensemble,
            ensemble_recipe=dist_ensemble,
            functions=functions,
            backend=backend,
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _resolve(self) -> EvaluationResult:
        """Block until the future completes and return the cached result.

        Idempotent: subsequent calls return the cached value without
        re-blocking.
        """
        if self._result is None:
            self._result = self._future.result()
        return self._result

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def future(self) -> concurrent.futures.Future["EvaluationResult"]:
        """The underlying :class:`concurrent.futures.Future`.

        Provides direct access to the raw future so that protocol-layer
        wrappers (e.g. :class:`~simulation.runner.ModelRunHandle`) can hold
        the future without accessing private state.

        Returns
        -------
        concurrent.futures.Future[EvaluationResult]
            The future that will resolve to the evaluation result.
        """
        return self._future

    @property
    def result(self) -> EvaluationResult:
        """The evaluation result.

        Raises
        ------
        RuntimeError
            If the background evaluation has not yet completed.  Use
            :meth:`poll` or :meth:`get` to wait for completion.
        """
        if not self._future.done():
            raise RuntimeError(
                "AsyncEvaluationHandle: evaluation is still running. "
                "Call .get() to wait for completion or .poll() to check status."
            )
        return self._resolve()

    def poll(self) -> tuple[bool, EvaluationResult | None]:
        """Non-blocking status check.

        Returns
        -------
        tuple[bool, EvaluationResult | None]
            ``(True, result)`` if the evaluation is complete;
            ``(False, None)`` if it is still running.
        """
        if not self._future.done():
            return False, None
        return True, self._resolve()

    def get(self) -> EvaluationResult:
        """Block until the evaluation completes and return the result.

        Returns
        -------
        EvaluationResult
            The completed evaluation result.  Subsequent calls return the
            cached result immediately.
        """
        return self._resolve()

    def extend(
        self,
        ensemble_size: int = 0,
        *,
        extra_ensemble: dict[str, int] | None = None,
        extra_parameters: dict[GenericIndex, np.ndarray] | None = None,
    ) -> EvaluationResult:
        """Extend the ensemble after the background evaluation completes.

        Delegates to :meth:`EvaluationHandle.extend` after verifying that
        the future has resolved.

        Raises
        ------
        RuntimeError
            If the background evaluation has not yet completed.
        """
        if not self._future.done():
            raise RuntimeError(
                "AsyncEvaluationHandle: cannot extend before the evaluation completes. Call .get() or .poll() first."
            )
        self._resolve()  # populate self._result before super().extend() reads it
        return super().extend(
            ensemble_size, extra_ensemble=extra_ensemble, extra_parameters=extra_parameters
        )
