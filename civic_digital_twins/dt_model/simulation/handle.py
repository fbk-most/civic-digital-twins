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

Obtain instances via
:meth:`~simulation.evaluation.Evaluation.evaluate_incremental` (synchronous) or
:meth:`~simulation.evaluation.Evaluation.submit_evaluate` (asynchronous) rather than
constructing these classes directly.
"""

from __future__ import annotations

import concurrent.futures
from collections.abc import Mapping

import numpy as np

from ..engine.frontend import graph
from ..engine.numpybackend import executor
from ..model.axis import ENSEMBLE, PARAMETER, Axis
from ..model.index import GenericIndex

# Imported at runtime to avoid circular imports:
#   evaluation.py → handle.py  (local import inside evaluate_incremental)
#   handle.py → evaluation.py  (module-level imports below are fine because
#                                evaluation.py does not import handle.py at
#                                module level)
from .evaluation import Evaluation, EvaluationResult
from .plan import EvaluationPlan

__all__ = ["AsyncEvaluationHandle", "EvaluationHandle"]


class _ReplayEnsemble:
    """AxisEnsemble backed by pre-computed (frozen) sample arrays — no RNG draws.

    Returned by :func:`_replay_from` and stored on :class:`EvaluationHandle` so
    that parameter-grid extension can re-run the plan over the *same* scenarios
    without advancing the RNG a second time.
    """

    def __init__(
        self,
        axis: Axis,
        weights: np.ndarray,
        cached_assignments: dict[GenericIndex, np.ndarray],
    ) -> None:
        self._axis = axis
        self._weights = weights
        self._cached_assignments = cached_assignments

    @property
    def ensemble_axes(self) -> tuple[Axis, ...]:
        return (self._axis,)

    @property
    def ensemble_weights(self) -> tuple[np.ndarray, ...]:
        return (self._weights,)

    def assignments(self) -> Mapping[GenericIndex, np.ndarray]:
        return self._cached_assignments

    def concat(self, other: _ReplayEnsemble) -> _ReplayEnsemble:
        """Return a new replay ensemble with concatenated samples and proportional weights."""
        S1 = self._weights.size
        S2 = other._weights.size
        alpha = S1 / (S1 + S2)
        merged_weights = np.concatenate([self._weights * alpha, other._weights * (1.0 - alpha)])
        merged_assignments = {
            idx: np.concatenate([self._cached_assignments[idx], other._cached_assignments[idx]])
            for idx in self._cached_assignments
        }
        return _ReplayEnsemble(Axis("_ensemble", ENSEMBLE), merged_weights, merged_assignments)


def _replay_from_dist(dist_ensemble: object) -> _ReplayEnsemble:
    """Draw samples from *dist_ensemble* once and wrap them in a :class:`_ReplayEnsemble`."""
    return _ReplayEnsemble(
        dist_ensemble.ensemble_axes[0],  # type: ignore[union-attr]
        dist_ensemble.ensemble_weights[0],  # type: ignore[union-attr]
        dict(dist_ensemble.assignments()),  # type: ignore[union-attr]
    )


def _merge_results(
    r1: EvaluationResult,
    r2: EvaluationResult,
    plan: EvaluationPlan,
) -> EvaluationResult:
    """Merge two :class:`~simulation.evaluation.EvaluationResult` instances.

    Both results must have been produced by the same plan and with the same
    PARAMETER axes.  The merge concatenates node values along the single
    ENSEMBLE axis and combines weights as a size-proportional mixture:
    each scenario's weight is scaled by ``S_i / (S1 + S2)`` so that the
    merged weights still sum to 1 and non-uniform weight schemes (e.g.
    :class:`~simulation.ensemble.CrossProductEnsemble`) are preserved.

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

    Returns
    -------
    EvaluationResult
        A new result whose node arrays are concatenated along the ENSEMBLE axis.

    Raises
    ------
    ValueError
        If either result has no ENSEMBLE axis, if their ENSEMBLE axes are at
        different positions, or if the PARAMETER axis layouts are incompatible.
    NotImplementedError
        If either result has more than one ENSEMBLE axis (multi-axis merging is
        not supported in v0.10.0).
    """
    # --- Locate the single ENSEMBLE axis in each result ---
    ens_1 = [(ax, pos) for ax, pos in r1._axis_layout.items() if ax.role == ENSEMBLE]
    ens_2 = [(ax, pos) for ax, pos in r2._axis_layout.items() if ax.role == ENSEMBLE]

    if not ens_1 or not ens_2:
        raise ValueError(
            "_merge_results requires both results to have exactly one ENSEMBLE axis; "
            f"got {len(ens_1)} in r1 and {len(ens_2)} in r2."
        )
    if len(ens_1) != 1 or len(ens_2) != 1:
        raise NotImplementedError(
            "Merging results with multiple ENSEMBLE axes is not supported in v0.10.0. "
            "Use a single-axis DistributionEnsemble."
        )

    ax1, ens_pos = ens_1[0]
    ax2, ens_pos2 = ens_2[0]
    if ens_pos != ens_pos2:
        raise ValueError(  # pragma: no cover
            f"ENSEMBLE axis position mismatch: r1 has ensemble at dim {ens_pos}, r2 has it at dim {ens_pos2}."
        )

    S1: int = r1._axis_sizes[ax1]
    S2: int = r2._axis_sizes[ax2]

    # --- Validate that PARAMETER axes are compatible ---
    # Both results must come from the same plan executed with identical parameters.
    # Axis equality is identity-based, so we compare by (name, role, position, size).
    def _param_sig(layout: dict[Axis, int], sizes: dict[Axis, int]) -> frozenset[tuple[str, object, int, int]]:
        return frozenset((ax.name, ax.role, pos, sizes[ax]) for ax, pos in layout.items() if ax.role != ENSEMBLE)

    if _param_sig(r1._axis_layout, r1._axis_sizes) != _param_sig(r2._axis_layout, r2._axis_sizes):
        raise ValueError(
            "_merge_results requires both results to have identical PARAMETER axis layouts. "
            "Ensure both were built from the same plan with the same 'parameters=' dict."
        )

    # --- Merge node values along the ENSEMBLE axis ---
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

        # Ensure both arrays are at least (ens_pos + 1)-dimensional.
        while v1.ndim <= ens_pos:
            v1 = v1[np.newaxis]  # pragma: no cover
        while v2.ndim <= ens_pos:
            v2 = v2[np.newaxis]  # pragma: no cover

        if v1.shape[ens_pos] == 1 and v2.shape[ens_pos] == 1 and np.array_equal(v1, v2):
            merged_values[node] = v1
        else:
            # Expand singleton dims before concatenation so shapes match.
            if v1.shape[ens_pos] == 1:
                bcast = v1.shape[:ens_pos] + (S1,) + v1.shape[ens_pos + 1 :]
                v1 = np.broadcast_to(v1, bcast).copy()
            if v2.shape[ens_pos] == 1:
                bcast = v2.shape[:ens_pos] + (S2,) + v2.shape[ens_pos + 1 :]
                v2 = np.broadcast_to(v2, bcast).copy()
            merged_values[node] = np.concatenate([v1, v2], axis=ens_pos)

    # --- Build merged axis metadata ---
    # Create a fresh Axis object for the combined ensemble dimension
    # (Axis equality is identity-based; we must not reuse ax1 or ax2 as dict keys
    # since their sizes differ from the merged size).
    merged_ens_axis = Axis("_ensemble", ENSEMBLE)
    merged_axis_layout: dict[Axis, int] = {
        **{ax: pos for ax, pos in r1._axis_layout.items() if ax.role != ENSEMBLE},
        merged_ens_axis: ens_pos,
    }
    merged_axis_sizes: dict[Axis, int] = {
        **{ax: sz for ax, sz in r1._axis_sizes.items() if ax.role != ENSEMBLE},
        merged_ens_axis: S1 + S2,
    }

    # Size-proportional mixture: each partial result contributes weight
    # proportional to its scenario count, preserving non-uniform schemes.
    w1 = r1._factorized_weights[ax1]
    w2 = r2._factorized_weights[ax2]
    alpha = S1 / (S1 + S2)
    merged_weights = np.concatenate([w1 * alpha, w2 * (1.0 - alpha)])
    merged_factorized_weights: dict[Axis, np.ndarray] = {merged_ens_axis: merged_weights}

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
        raise ValueError(
            f"ENSEMBLE axis position mismatch: r1 at dim {ens_pos}, r2 at dim {ens_pos2}."
        )
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
        raise ValueError(
            f"_merge_results_param_extend: no PARAMETER axis named {param_name!r} in r2."
        )
    P2 = r2._axis_sizes[grow_ax_2]

    # --- Validate fixed PARAMETER axes ---
    def _fixed_sig(layout: dict[Axis, int], sizes: dict[Axis, int]) -> frozenset[tuple[str, int, int]]:
        return frozenset(
            (ax.name, pos, sizes[ax])
            for ax, pos in layout.items()
            if ax.role == PARAMETER and ax.name != param_name
        )

    if _fixed_sig(r1._axis_layout, r1._axis_sizes) != _fixed_sig(r2._axis_layout, r2._axis_sizes):
        raise ValueError(
            "_merge_results_param_extend: fixed PARAMETER axis layouts differ between r1 and r2."
        )

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

        Both the initial evaluation (via
        :meth:`~simulation.evaluation.Evaluation.evaluate_incremental`) and
        every subsequent :meth:`extend` call use
        :class:`~simulation.ensemble.DistributionEnsemble` to generate
        scenarios.  This means the model's abstract indexes must all be either
        :class:`~model.index.Distribution`-backed or
        :class:`~model.index.CategoricalIndex`.  Models that require
        :class:`~simulation.ensemble.CrossProductEnsemble` or
        :class:`~simulation.ensemble.PartitionedEnsemble` cannot be used with
        this handle; use
        :meth:`~simulation.evaluation.Evaluation.execute_plan` directly
        instead.

    Obtain an instance via
    :meth:`~simulation.evaluation.Evaluation.evaluate_incremental` rather than
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
        ensemble: _ReplayEnsemble | None = None,
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
    # Private helpers — one primitive operation each
    # ------------------------------------------------------------------

    def _extend_ensemble_axis(self, ensemble_size: int) -> None:
        """Draw *ensemble_size* new scenarios and merge along the ENSEMBLE axis."""
        from .ensemble import DistributionEnsemble

        new_dist = DistributionEnsemble(
            self._evaluation._scenario, ensemble_size, rng=self._rng, exclude=frozenset(self._parameters)
        )
        new_replay = _replay_from_dist(new_dist)
        new_result = self._evaluation.execute_plan(
            self._plan, new_replay,
            parameters=self._parameters, parameter_axes=self._parameter_axes,
            functions=self._functions, backend=self._backend,
        )
        assert self._result is not None
        self._result = _merge_results(self._result, new_result, self._plan)
        if self._ensemble is not None:
            self._ensemble = self._ensemble.concat(new_replay)

    def _extend_param_axis(self, param_idx: GenericIndex, extra_vals: np.ndarray) -> None:
        """Re-run the stored ensemble at *extra_vals* and merge along the PARAMETER axis."""
        assert self._result is not None
        assert self._ensemble is not None
        r_extra = self._evaluation.execute_plan(
            self._plan, self._ensemble,
            parameters={**self._parameters, param_idx: extra_vals},
            parameter_axes=self._parameter_axes,
            functions=self._functions, backend=self._backend,
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
        extra_parameters: dict[GenericIndex, np.ndarray] | None = None,
    ) -> EvaluationResult:
        """Grow the ensemble and/or the PARAMETER grid and merge.

        *ensemble_size* draws additional Monte Carlo samples via
        :class:`~simulation.ensemble.DistributionEnsemble` and merges along
        the ENSEMBLE axis.  *extra_parameters* re-runs the plan over new
        parameter values using the **same** sample draws and merges along the
        PARAMETER axis.  Both may be supplied together; multiple entries in
        *extra_parameters* are processed one axis at a time.

        Parameters
        ----------
        ensemble_size:
            Number of new Monte Carlo scenarios to evaluate.  When ``<= 0``
            and *extra_parameters* is ``None``, this is a no-op.
        extra_parameters:
            Dict ``{idx: extra_array, ...}`` extending the PARAMETER sweep for
            each *idx*.  Every key must already be present in the
            ``parameters=`` dict passed to
            :meth:`~simulation.evaluation.Evaluation.evaluate_incremental`.

        Returns
        -------
        EvaluationResult
            The updated accumulated result (same object as :attr:`result`
            after the merge).

        Raises
        ------
        ValueError
            If any key in *extra_parameters* is not in the original
            ``parameters=`` dict.
        RuntimeError
            If the handle has no stored ensemble (constructed without one).
        """
        if extra_parameters is None and ensemble_size <= 0:
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
            self._extend_ensemble_axis(ensemble_size)
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
    :class:`~simulation.ensemble.DistributionEnsemble` constraint that applies
    to both the initial evaluation and every :meth:`extend` call.

    Obtain an instance via
    :meth:`~simulation.evaluation.Evaluation.submit_evaluate` rather than
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
        ensemble: _ReplayEnsemble | None,
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
            functions=functions,
            backend=backend,
        )
        self._future = future

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
        return super().extend(ensemble_size, extra_parameters=extra_parameters)
