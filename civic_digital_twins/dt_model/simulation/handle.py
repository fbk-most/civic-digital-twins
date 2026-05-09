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

import numpy as np

from ..engine.frontend import graph
from ..engine.numpybackend import executor
from ..model.axis import ENSEMBLE, Axis
from ..model.index import GenericIndex

# Imported at runtime to avoid circular imports:
#   evaluation.py → handle.py  (local import inside evaluate_incremental)
#   handle.py → evaluation.py  (module-level imports below are fine because
#                                evaluation.py does not import handle.py at
#                                module level)
from .evaluation import Evaluation, EvaluationResult
from .plan import EvaluationPlan

__all__ = ["AsyncEvaluationHandle", "EvaluationHandle"]


def _merge_results(
    r1: EvaluationResult,
    r2: EvaluationResult,
    plan: EvaluationPlan,
) -> EvaluationResult:
    """Merge two :class:`~simulation.evaluation.EvaluationResult` instances.

    Both results must have been produced by the same plan and with the same
    PARAMETER axes.  The merge concatenates node values along the single
    ENSEMBLE axis and renormalises weights to a uniform distribution over
    the combined scenario count.

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
        A new result whose node arrays are concatenated along the ENSEMBLE axis
        (or kept as singletons when both inputs are singleton on that axis).

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

        if v1.shape[ens_pos] == 1 and v2.shape[ens_pos] == 1:
            # Node does not vary along ENSEMBLE in either batch — it is constant
            # (e.g. a parameter-only or fully-concrete node).  Both arrays must
            # be equal since they come from the same deterministic computation.
            if __debug__ and not np.array_equal(v1, v2):
                raise AssertionError(  # pragma: no cover
                    f"_merge_results: singleton node {getattr(node, 'name', repr(node))!r} "
                    f"has different values in r1 ({v1!r}) and r2 ({v2!r}). "
                    "This indicates a non-deterministic computation or mismatched plans."
                )
            merged_values[node] = v1
        else:
            # Expand singleton dims before concatenation so shapes match.
            if v1.shape[ens_pos] == 1:
                bcast = v1.shape[:ens_pos] + (S1,) + v1.shape[ens_pos + 1 :]  # pragma: no cover
                v1 = np.broadcast_to(v1, bcast).copy()  # pragma: no cover
            if v2.shape[ens_pos] == 1:
                bcast = v2.shape[:ens_pos] + (S2,) + v2.shape[ens_pos + 1 :]  # pragma: no cover
                v2 = np.broadcast_to(v2, bcast).copy()  # pragma: no cover
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

    # Uniform weights renormalised over the combined ensemble.
    merged_weights = np.full(S1 + S2, 1.0 / (S1 + S2))
    merged_factorized_weights: dict[Axis, np.ndarray] = {merged_ens_axis: merged_weights}

    merged_state = executor.State(merged_values)
    return EvaluationResult(
        merged_state,
        merged_axis_layout,
        r1._parameter_arrays,
        axis_sizes=merged_axis_sizes,
        factorized_weights=merged_factorized_weights,
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
        The PARAMETER axis dict passed to the initial execution.
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
        functions: dict[str, executor.Functor] | None,
        backend: type[executor.NumpyBackend],
    ) -> None:
        self._evaluation = evaluation
        self._plan = plan
        self._result = result
        self._rng = rng
        self._parameters = parameters
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

    def extend(
        self,
        ensemble_size: int = 0,
        *,
        extra_parameters: dict[GenericIndex, np.ndarray] | None = None,
    ) -> EvaluationResult:
        """Grow the ensemble by *ensemble_size* additional samples and merge.

        Draws ``ensemble_size`` new scenarios from
        :class:`~simulation.ensemble.DistributionEnsemble` using the shared RNG
        (see class-level note for the ensemble-type constraint), executes the
        plan, and merges the result via :func:`_merge_results`.  The merged
        result replaces :attr:`result`.

        Parameters
        ----------
        ensemble_size:
            Number of new Monte Carlo scenarios to evaluate.  When ``<= 0``
            and *extra_parameters* is ``None``, this is a no-op.
        extra_parameters:
            Not yet implemented in v0.10.0; raises :exc:`NotImplementedError`
            when provided.  PARAMETER-grid extension (evaluating at additional
            parameter values) requires a different merging strategy and is
            deferred to a future release.

        Returns
        -------
        EvaluationResult
            The updated accumulated result (same object as :attr:`result`
            after the merge).

        Raises
        ------
        NotImplementedError
            When *extra_parameters* is not ``None`` (deferred in v0.10.0).
        """
        if extra_parameters is not None:
            raise NotImplementedError(
                "Parameter-grid extension (extra_parameters) is not yet implemented "
                "in v0.10.0. PARAMETER-grid extension requires a different merging "
                "strategy and is deferred to a future release."
            )
        if ensemble_size <= 0:
            return self._result  # type: ignore[return-value]  # None only on async path before resolve

        assert self._result is not None, (
            "EvaluationHandle.extend() called with _result=None — "
            "this is a bug: either the handle was constructed incorrectly or "
            "AsyncEvaluationHandle.extend() failed to call _resolve() first."
        )

        from .ensemble import DistributionEnsemble

        new_ensemble = DistributionEnsemble(self._plan.model, ensemble_size, rng=self._rng)
        new_result = self._evaluation.execute_plan(
            self._plan,
            new_ensemble,
            parameters=self._parameters,
            functions=self._functions,
            backend=self._backend,
        )
        self._result = _merge_results(self._result, new_result, self._plan)
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
        functions: dict[str, executor.Functor] | None,
        backend: type[executor.NumpyBackend],
    ) -> None:
        super().__init__(
            evaluation=evaluation,
            plan=plan,
            result=None,  # not yet available; resolved lazily by _resolve()
            rng=rng,
            parameters=parameters,
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
        NotImplementedError
            When *extra_parameters* is not ``None`` (deferred in v0.10.0).
        """
        if not self._future.done():
            raise RuntimeError(
                "AsyncEvaluationHandle: cannot extend before the evaluation completes. Call .get() or .poll() first."
            )
        self._resolve()  # populate self._result before super().extend() reads it
        return super().extend(ensemble_size, extra_parameters=extra_parameters)
