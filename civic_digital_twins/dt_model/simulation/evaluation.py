"""Generic model evaluation."""
# SPDX-License-Identifier: Apache-2.0

import concurrent.futures
import warnings
from collections.abc import Mapping
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .handle import AsyncEvaluationHandle, EvaluationHandle

from ..engine.frontend import graph, linearize
from ..engine.numpybackend import executor
from ..model.axis import ENSEMBLE, PARAMETER, Axis
from ..model.index import GenericIndex
from ..model.model import Model
from ..model.model_variant import ModelVariant
from .ensemble import AxisEnsemble, DistributionEnsemble, Ensemble, WeightedScenario
from .plan import EvaluationPlan, Region, RegionGuard

__all__ = ["EvaluationResult", "Evaluation"]

# Module-level default executor for submit_evaluate().
# Uses a ThreadPoolExecutor so that the GIL is released during NumPy computation
# and the main thread remains responsive while evaluation runs in the background.
# Process pools are deferred to v0.11 (out of scope for this milestone).
_DEFAULT_EXECUTOR: concurrent.futures.ThreadPoolExecutor = concurrent.futures.ThreadPoolExecutor()


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

    Given a model (or Scenario), :meth:`build_plan` encodes the DAG navigation
    strategy as an :class:`~simulation.plan.EvaluationPlan`, and
    :meth:`execute_plan` runs it against a given ensemble and parameter grid,
    returning an :class:`EvaluationResult`.  :meth:`evaluate` is a thin
    convenience wrapper that calls both in sequence.

    This class knows nothing about grids, presence variables, sustainability,
    or constraints — all domain-specific logic lives in subclasses or
    vertical-specific wrappers.

    Parameters
    ----------
    model:
        The model to evaluate.  Also accepts Scenario-like objects (any object
        with a ``.model`` attribute of type :class:`~.model.model.Model` or
        :class:`~.model.model_variant.ModelVariant`) at runtime; the type
        annotation will be updated when ``Scenario`` (parameter-axes.md D1b)
        is implemented.
    """

    def __init__(self, model: Model | ModelVariant) -> None:
        # Accept Scenario-like objects at runtime via duck typing
        # (parameter-axes.md D1b — Scenario not yet implemented).
        # Using an untyped intermediate avoids a spurious Pyright
        # "condition is always True" error on the isinstance check.
        _arg: object = model
        if not isinstance(_arg, (Model, ModelVariant)):
            extracted = getattr(_arg, "model", None)
            if not isinstance(extracted, (Model, ModelVariant)):
                raise TypeError(
                    f"Evaluation() expects a Model, ModelVariant, or Scenario-like "
                    f"object with a '.model' attribute; got {type(_arg).__name__!r}."
                )
            self.model: Model | ModelVariant = extracted
        else:
            self.model = _arg

    # ------------------------------------------------------------------
    # Plan construction
    # ------------------------------------------------------------------

    def build_plan(
        self,
        nodes_of_interest: list[GenericIndex] | None = None,
        *,
        strategy: str = "monolithic",
    ) -> EvaluationPlan:
        """Build an :class:`~simulation.plan.EvaluationPlan` for this model.

        Encodes the DAG partitioning strategy — how the computation graph
        is split into :class:`~simulation.plan.Region` instances and in what order
        they execute — independently of the execution inputs (ensemble,
        parameters).  The returned plan can be reused across multiple
        :meth:`execute_plan` calls with different ensembles or parameter grids.

        Parameters
        ----------
        nodes_of_interest:
            Indexes to evaluate.  Transitive dependencies are resolved
            automatically via :func:`~engine.frontend.linearize.forest`.
            Defaults to all indexes in the model when ``None``.
        strategy:
            DAG partitioning strategy.

            ``"monolithic"`` (default)
                One region containing all linearised nodes.  Always available.
            ``"regional"``
                Split at :class:`~engine.frontend.graph.variant_selector`
                boundaries into multiple regions: one unconditional shared
                region for the pre-selector sub-graph, one guarded region per
                variant branch, and one unconditional merge region.  Current
                limitations: exactly one top-level
                :class:`~engine.frontend.graph.variant_selector` (nested
                variants require recursive partitioning); a single-axis
                ensemble (multi-axis scenario masking requires tensor
                fancy-indexing); and a selector that does not vary along any
                PARAMETER axis (a PARAMETER-varying selector implies a
                different scenario partition per parameter combination,
                requiring per-parameter scatter/gather).  In all these cases
                use ``strategy='monolithic'``.

        Returns
        -------
        EvaluationPlan
            An :class:`~simulation.plan.EvaluationPlan` ready for
            :meth:`execute_plan`.

        Raises
        ------
        ValueError
            If *strategy* is not a recognised value.
        NotImplementedError
            If *strategy* is ``"regional"`` and the model contains more than
            one :class:`~engine.frontend.graph.variant_selector` node (nested
            variants require recursive partitioning, not yet supported).
        """
        if nodes_of_interest is None:
            nodes_of_interest = list(self.model.indexes)

        actual_nodes = [idx.node for idx in nodes_of_interest]
        linearized_nodes = linearize.forest(*actual_nodes)
        has_timeseries = any(
            isinstance(node, (graph.timeseries_constant, graph.timeseries_placeholder)) for node in linearized_nodes
        )

        if strategy == "monolithic":
            return EvaluationPlan(
                model=self.model,
                nodes_of_interest=tuple(nodes_of_interest),
                regions=(Region(nodes=tuple(linearized_nodes), has_timeseries=has_timeseries),),
                dependencies=(frozenset(),),
            )
        if strategy == "regional":
            # ---------------------------------------------------------------
            # Regional partitioning — step-by-step
            #
            # Example model throughout:
            #   mode  = CategoricalIndex("mode", {"bike": 0.3, "train": 0.7})
            #   mv    = ModelVariant("Transport",
            #               {"bike": BikeModel(cap_bike),
            #                "train": TrainModel(cap_train)},
            #               selector=mode)
            #
            # ModelVariant.__init__ builds exactly one variant_selector node
            # (vs) and one exclusive_multi_clause_where node per output field
            # (mcw_tp, mcw_em).  The graph looks like:
            #
            #   mode.node ──┬── eq_bike  (mode.node == "bike")
            #               └── eq_train (mode.node == "train")
            #
            #   cap_bike.node  ──> tp_bike.node  ───┐
            #   em_bike.node   ─────────────────────┤
            #                                       ├──> vs
            #   cap_train.node ──> tp_train.node  ──┤
            #   em_train.node  ─────────────────────┘
            #
            #   vs, eq_bike, tp_bike.node, eq_train, tp_train.node
            #                                       ──> mcw_tp
            #   vs, eq_bike, em_bike.node, eq_train, em_train.node
            #                                       ──> mcw_em
            #
            # vs carries three attributes used by this function:
            #   vs.selector_node  = mode.node
            #   vs.branch_map     = {"bike":  [tp_bike.node,  em_bike.node],
            #                        "train": [tp_train.node, em_train.node]}
            #   vs.merge_nodes    = [mcw_tp, mcw_em]
            # ---------------------------------------------------------------

            # Step 1 — locate the single variant_selector.
            # linearize.forest already traversed the full graph; vs is the
            # unique structural sentinel introduced by ModelVariant.
            vs_nodes = [n for n in linearized_nodes if isinstance(n, graph.variant_selector)]
            if not vs_nodes:
                raise ValueError(
                    "build_plan(strategy='regional') requires a ModelVariant with a "
                    "runtime selector. No variant_selector found — use strategy='monolithic'."
                )
            if len(vs_nodes) > 1:
                raise NotImplementedError(
                    f"build_plan(strategy='regional') supports exactly one top-level "
                    f"variant_selector; found {len(vs_nodes)}. "
                    "Nested variants will be supported in a future release."
                )
            vs = vs_nodes[0]

            # Step 2 — collect condition nodes from the merge clauses.
            # Each mcw node has clauses = [(eq_bike, tp_bike.node), ...]; the
            # condition nodes (eq_bike, eq_train) depend only on mode.node and
            # string constants, so they belong in the shared region.
            #   all_cond_nodes = [eq_bike, eq_train, eq_bike, eq_train]
            #                     ^- from mcw_tp -^  ^- from mcw_em -^
            # (duplicates are harmless; forest() deduplicates via its visited set)
            all_cond_nodes = [
                cond
                for mcw in vs.merge_nodes
                if isinstance(mcw, graph.exclusive_multi_clause_where)
                for cond, _ in mcw.clauses
            ]
            # Step 3 — shared region node set.
            # forest(mode.node, eq_bike, eq_train) traverses backwards from each
            # root and returns all transitive dependencies in topological order.
            # Result: {mode.node, constant("bike"), constant("train"),
            #          eq_bike, eq_train}
            # (plus any upstream deps of mode.node, e.g. its placeholder node)
            shared_node_set = set(linearize.forest(vs.selector_node, *all_cond_nodes))

            # Step 4 — per-branch node sets.
            # forest(*branch_outputs, boundary=shared_node_set) traverses
            # backwards from each branch's output nodes but stops when it
            # hits a node already in shared_node_set.
            #   branch_map["bike"]  roots: [tp_bike.node, em_bike.node]
            #   → traversal stops at mode.node (in shared) → adds:
            #     {cap_bike.node, tp_bike.node, em_bike.node}
            #   Subtract shared_node_set (boundary nodes can appear in both):
            #     branch_node_sets["bike"]  = {cap_bike.node, tp_bike.node, em_bike.node}
            #     branch_node_sets["train"] = {cap_train.node, tp_train.node, em_train.node}
            branch_node_sets: dict[str, set[graph.Node]] = {}
            for key, branch_outputs in vs.branch_map.items():
                branch_ns = set(linearize.forest(*branch_outputs, boundary=shared_node_set))
                branch_ns -= shared_node_set
                branch_node_sets[key] = branch_ns

            # Step 5 — merge region node set (complement).
            # Everything in linearized_nodes that is neither shared nor a
            # branch internal belongs to the merge region:
            #   merge_node_set = {vs, constant(nan), mcw_tp, mcw_em}
            # Note: constant(nan) is the default_value of each mcw and has no
            # deps of its own, so forest() never places it in shared or any
            # branch; the complement captures it here automatically.
            already_assigned: set[graph.Node] = shared_node_set.copy()
            for bns in branch_node_sets.values():
                already_assigned |= bns
            merge_node_set = set(linearized_nodes) - already_assigned

            def _is_ts(nodes: tuple[graph.Node, ...]) -> bool:
                return any(isinstance(n, (graph.timeseries_constant, graph.timeseries_placeholder)) for n in nodes)

            # Extract nodes in topological order for each region (preserving order from linearized_nodes).
            shared_nodes_topo = tuple(n for n in linearized_nodes if n in shared_node_set)
            merge_nodes_topo = tuple(n for n in linearized_nodes if n in merge_node_set)

            shared_region = Region(nodes=shared_nodes_topo, has_timeseries=_is_ts(shared_nodes_topo))

            branch_regions: list[Region] = []
            for key in vs.branch_map:
                bns = tuple(n for n in linearized_nodes if n in branch_node_sets[key])
                branch_regions.append(
                    Region(
                        nodes=bns,
                        has_timeseries=_is_ts(bns),
                        guard=RegionGuard(selector_node=vs.selector_node, branch_key=key),
                    )
                )

            merge_region = Region(nodes=merge_nodes_topo, has_timeseries=_is_ts(merge_nodes_topo))

            # Build the DAG: shared(0) → branches(1..N) → merge(N+1).
            n_branches = len(branch_regions)
            all_regions = (shared_region, *branch_regions, merge_region)
            all_deps: tuple[frozenset[int], ...] = (
                frozenset(),  # shared: no predecessors
                *[frozenset({0}) for _ in branch_regions],  # each branch depends on shared
                frozenset(range(n_branches + 1)),  # merge depends on shared + all branches
            )

            return EvaluationPlan(
                model=self.model,
                nodes_of_interest=tuple(nodes_of_interest),
                regions=all_regions,
                dependencies=all_deps,
            )
        raise ValueError(f"Unknown strategy {strategy!r}. Expected 'monolithic' or 'regional'.")

    # ------------------------------------------------------------------
    # Plan execution
    # ------------------------------------------------------------------

    def execute_plan(
        self,
        plan: EvaluationPlan,
        ensemble: AxisEnsemble | None = None,
        *,
        parameters: dict[GenericIndex, np.ndarray] | None = None,
        functions: dict[str, executor.Functor] | None = None,
        backend: type[executor.NumpyBackend] = executor.NumpyBackend,
    ) -> EvaluationResult:
        """Execute a pre-built plan against a given ensemble and parameter grid.

        Parameters
        ----------
        plan:
            The plan to execute, built via :meth:`build_plan`.
        ensemble:
            The ensemble to evaluate.  Must be an :class:`AxisEnsemble`
            (canonical, batched) or ``None`` for deterministic evaluation.
            Legacy ``Iterable[WeightedScenario]`` inputs must be adapted
            before this call (done automatically by :meth:`evaluate`).
        parameters:
            PARAMETER axes for multi-dimensional evaluation.  Maps each
            axis :class:`~dt_model.model.index.GenericIndex` to a 1-D numpy
            array of values.  When ``None``, defaults to an empty dict (no
            PARAMETER axes).
        functions:
            Optional user-defined functions passed to the executor.  Wrap
            callables with :meth:`~executor.NumpyBackend.adapt` before passing.
        backend:
            The computation backend to use.  Currently only
            :class:`~executor.NumpyBackend` is supported (the default).

        Returns
        -------
        EvaluationResult
            Typed result wrapper.

        Raises
        ------
        NotImplementedError
            If *plan* contains a guarded region and the ensemble has more than
            one axis: multi-axis scenario masking requires tensor
            fancy-indexing, which is not yet implemented.  Use
            ``strategy='monolithic'`` or a single-axis
            :class:`~simulation.ensemble.DistributionEnsemble` instead.
        """
        parameters = parameters or {}
        return self._execute_plan(
            plan,
            ensemble,
            parameters=parameters,
            functions=functions,
            backend=backend,
        )

    def _execute_plan(
        self,
        plan: EvaluationPlan,
        ensemble: AxisEnsemble | None,
        *,
        parameters: dict[GenericIndex, np.ndarray],
        functions: dict[str, executor.Functor] | None,
        backend: type[executor.NumpyBackend],
    ) -> EvaluationResult:
        """Execute an :class:`~simulation.plan.EvaluationPlan`."""
        actual_nodes = [idx.node for idx in plan.nodes_of_interest]
        _has_timeseries = any(r.has_timeseries for r in plan.regions)

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
                # Prepend n_params PARAMETER singletons so ENSEMBLE arrays
                # broadcast correctly against the (*PARAMETER, *ENSEMBLE) layout.
                # When the model contains timeseries nodes, also append a
                # trailing 1 for scalar (non-timeseries) assignments so they
                # broadcast with timeseries (T,) nodes: (S, 1) × (T,) → (S, T).
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
        # Snapshot the substituted node keys before the executor mutates state.values.
        # executor.State takes c_subs by reference and adds every computed node into
        # the same dict; capturing the keys now lets the shape-normalisation step
        # below distinguish pre-supplied substitutions from executor-computed values.
        substituted_nodes: set[graph.Node] = set(c_subs)

        if backend is not executor.NumpyBackend:
            raise NotImplementedError(f"Backend {backend!r} is not supported; only NumpyBackend is available.")

        n_full = n_params + n_ensemble
        n_total = n_full + extra_ts

        # Validate ensemble cardinality when guarded regions are present.
        has_guarded = any(r.guard is not None for r in plan.regions)
        if has_guarded and n_ensemble != 1:
            raise NotImplementedError(
                "Regional execution with a multi-axis or absent ensemble is not yet "
                "supported. Use strategy='monolithic' or a single-axis ensemble."
            )
        # Total scenario count (used for scatter-back in guarded regions).
        n_S: int = axis_sizes[list(ensemble.ensemble_axes)[0]] if (has_guarded and ensemble is not None) else 0

        state = executor.State(c_subs, functions=functions or {})

        # Execute regions in topological order.
        for region in plan.regions:
            if region.guard is None:
                # Unconditional region — evaluate for all scenarios.
                executor.evaluate_nodes(state, *region.nodes)
            else:
                # Guarded region — evaluate only for the matching scenario subset.
                guard = region.guard
                sel_val = np.asarray(state.values[guard.selector_node])
                # Guard: the selector must not vary along any PARAMETER axis.
                # If it does, the 1-D scenario mask below would only reflect
                # PARAMETER position 0, silently producing wrong results for
                # every other PARAMETER combination.  Detect and reject early.
                if n_params > 0 and any(sel_val.shape[i] > 1 for i in range(n_params)):
                    raise NotImplementedError(
                        "Regional execution is not supported when the variant selector "
                        "depends on PARAMETER axes (selector shape "
                        f"{sel_val.shape!r} has non-singleton PARAMETER dims). "
                        "The scenario partition would differ per parameter combination, "
                        "requiring per-parameter scatter/gather — not yet implemented. "
                        "Use strategy='monolithic' instead."
                    )
                # sel_val shape: (1,)*n_params + (S,) + (1,)*extra_ts
                # Extract 1-D boolean mask over the S axis.
                s_idx: tuple[int | slice, ...] = (0,) * n_params + (slice(None),) + (0,) * extra_ts
                mask_1d: np.ndarray = sel_val[s_idx] == guard.branch_key  # shape (S,)
                branch_idx: np.ndarray = np.where(mask_1d)[0]

                if len(branch_idx) == 0:
                    # No scenarios fall into this branch. Pre-initialize branch
                    # nodes to NaN arrays so the merge region's np.select can
                    # reference them (branch conditions are all False, so NaN
                    # values are never selected, but the arrays must exist).
                    nan_shape = (1,) * n_params + (n_S,) + (1,) * extra_ts
                    for node in region.nodes:
                        if node not in state.values:
                            state.values[node] = np.full(nan_shape, np.nan, dtype=float)
                            substituted_nodes.add(node)  # prevent spurious shape-norm
                    continue

                # Build a branch-local state:
                # - inherits all values already computed in the main state
                # - overrides ENSEMBLE subs for abstract indexes local to this branch
                branch_values: dict[graph.Node, np.ndarray] = dict(state.values)
                branch_region_nodes = set(region.nodes)
                for ens_node, ens_arr in ens_subs.items():
                    if ens_node in branch_region_nodes:
                        branch_values[ens_node] = np.take(ens_arr, branch_idx, axis=n_params)

                branch_state = executor.State(branch_values, functions=functions or {})
                executor.evaluate_nodes(branch_state, *region.nodes)

                # Scatter branch results back into the main state as full-S arrays,
                # with NaN at positions that do not belong to this branch.
                for node in region.nodes:
                    if node not in branch_state.values or node in state.values:
                        continue
                    val_k = np.asarray(branch_state.values[node])
                    # Constant nodes (no dependency on ensemble subs) evaluate to
                    # scalars (ndim=0) or small arrays (ndim<=n_params).  Expand
                    # them so the scatter indexing logic below can assume at least
                    # n_params+1 dimensions with the ensemble axis at position n_params.
                    if val_k.ndim <= n_params:
                        leading = (1,) * (n_params + 1 - val_k.ndim)
                        val_k = val_k.reshape(leading + val_k.shape)
                        # Broadcast the singleton ensemble dim to len(branch_idx).
                        bcast_shape = val_k.shape[:n_params] + (len(branch_idx),) + val_k.shape[n_params + 1 :]
                        val_k = np.broadcast_to(val_k, bcast_shape).copy()
                    full_shape = list(val_k.shape)
                    full_shape[n_params] = n_S
                    full_val = np.full(full_shape, np.nan, dtype=float)
                    idx_expand = branch_idx.reshape((1,) * n_params + (-1,) + (1,) * (val_k.ndim - n_params - 1))
                    np.put_along_axis(full_val, np.broadcast_to(idx_expand, val_k.shape), val_k, axis=n_params)
                    state.values[node] = full_val
                    substituted_nodes.add(node)  # mark as correctly shaped; skip shape-norm

        # All nodes in topological order (for touched-set computation).
        all_region_nodes = [n for r in plan.regions for n in r.nodes]

        # Shape normalisation: every actual node must end up with shape
        # (*PARAMETER, *ENSEMBLE, *domain) with explicit singletons where
        # the node does not vary along an axis.
        #
        # All-or-nothing property: a node is either
        #   (a) downstream of some substitution → the executor already produced
        #       the correct shape via numpy broadcasting; or
        #   (b) not downstream of any substitution → natural shape with zero
        #       leading dims (scalar () or bare timeseries (T,)).
        #
        # A single reshape prepending (n_total - arr.ndim) singletons handles
        # both subcases of (b), eliminating the need for a per-axis loop.
        # Nodes scattered back from guarded regions are added to substituted_nodes
        # so they are treated as case (a) and skipped here.
        if n_total > 0:
            # all_touched: nodes transitively downstream of any substitution.
            # Use the pre-executor snapshot (substituted_nodes) so that constant
            # nodes evaluated by the executor don't accidentally appear as
            # substituted and bypass the reshape.
            all_touched: set[graph.Node] = set(substituted_nodes)
            for node in all_region_nodes:
                if node in all_touched:
                    continue
                if any(dep in all_touched for dep in linearize._get_dependencies(node)):
                    all_touched.add(node)

            for node in actual_nodes:
                if node in all_touched or node not in state.values:
                    continue
                arr = np.asarray(state.values[node])
                assert arr.ndim in ({0, 1} if _has_timeseries else {0}), (
                    f"Untouched node {getattr(node, 'name', repr(node))!r}: "
                    f"unexpected ndim={arr.ndim} (has_timeseries={_has_timeseries})"
                )
                n_inject = n_total - arr.ndim
                state.values[node] = arr.reshape((1,) * n_inject + arr.shape)

        if __debug__:
            for node in actual_nodes:
                assert node in state.values, (
                    f"Post-norm: node {getattr(node, 'name', repr(node))!r} missing from "
                    f"state after evaluate_nodes — this is a bug in the executor"
                )
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

    def evaluate_incremental(
        self,
        initial_ensemble_size: int,
        nodes_of_interest: list[GenericIndex] | None = None,
        *,
        parameters: dict[GenericIndex, np.ndarray] | None = None,
        strategy: str = "monolithic",
        rng: np.random.Generator | None = None,
        functions: dict[str, executor.Functor] | None = None,
        backend: type[executor.NumpyBackend] = executor.NumpyBackend,
    ) -> "EvaluationHandle":
        """Build a plan, run an initial ensemble, and return an incremental handle.

        The returned :class:`~simulation.handle.EvaluationHandle` holds the first
        result and can be extended with additional Monte Carlo samples via
        :meth:`~simulation.handle.EvaluationHandle.extend` without discarding
        prior results.

        All sample draws (initial and extended) share the same
        :class:`numpy.random.Generator`, making the full sequence reproducible
        from a single seed.

        Parameters
        ----------
        initial_ensemble_size:
            Number of scenarios in the first evaluation batch.
        nodes_of_interest:
            Indexes to evaluate.  Defaults to all indexes in the model.
        parameters:
            PARAMETER axes for multi-dimensional evaluation.  Passed through
            to every :meth:`execute_plan` call on the handle.
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
        from .handle import EvaluationHandle  # local import avoids circular dependency

        parameters = parameters or {}
        if rng is None:
            rng = np.random.default_rng()

        plan = self.build_plan(nodes_of_interest, strategy=strategy)
        ensemble = DistributionEnsemble(self.model, initial_ensemble_size, rng=rng)
        result = self.execute_plan(plan, ensemble, parameters=parameters, functions=functions, backend=backend)
        return EvaluationHandle(
            evaluation=self,
            plan=plan,
            result=result,
            rng=rng,
            parameters=parameters,
            functions=functions,
            backend=backend,
        )

    def submit_evaluate(
        self,
        initial_ensemble_size: int,
        nodes_of_interest: list[GenericIndex] | None = None,
        *,
        parameters: dict[GenericIndex, np.ndarray] | None = None,
        strategy: str = "monolithic",
        rng: np.random.Generator | None = None,
        functions: dict[str, executor.Functor] | None = None,
        backend: type[executor.NumpyBackend] = executor.NumpyBackend,
        exec: concurrent.futures.Executor | None = None,
    ) -> "AsyncEvaluationHandle":
        """Submit an evaluation to a background thread and return immediately.

        Mirrors :meth:`evaluate_incremental` but runs the initial
        :meth:`execute_plan` call on a thread from *exec* (or the module-level
        :data:`_DEFAULT_EXECUTOR`) so that the caller is not blocked.  The
        returned :class:`~simulation.handle.AsyncEvaluationHandle` can be
        polled for status or awaited for its result.

        Once the future resolves, :meth:`~simulation.handle.AsyncEvaluationHandle.extend`
        works identically to :class:`~simulation.handle.EvaluationHandle`.

        Parameters
        ----------
        initial_ensemble_size:
            Number of scenarios in the first evaluation batch.
        nodes_of_interest:
            Indexes to evaluate.  Defaults to all indexes in the model.
        parameters:
            PARAMETER axes for multi-dimensional evaluation.
        strategy:
            Plan build strategy (``"monolithic"`` or ``"regional"``).
        rng:
            Random number generator for reproducibility.  When ``None``, a
            fresh :func:`numpy.random.default_rng` is created automatically.
        functions:
            Optional user-defined functions passed to the executor.
        backend:
            The computation backend (currently only ``NumpyBackend``).
        exec:
            :class:`concurrent.futures.Executor` to submit the work to.
            Defaults to the module-level :data:`_DEFAULT_EXECUTOR`
            (a :class:`~concurrent.futures.ThreadPoolExecutor`).

        Returns
        -------
        AsyncEvaluationHandle
            Handle wrapping the in-flight future.  Call
            :meth:`~simulation.handle.AsyncEvaluationHandle.get` to block for
            the result or
            :meth:`~simulation.handle.AsyncEvaluationHandle.poll` to check
            without blocking.
        """
        from .handle import AsyncEvaluationHandle  # local import avoids circular dependency

        parameters = parameters or {}
        if rng is None:
            rng = np.random.default_rng()

        plan = self.build_plan(nodes_of_interest, strategy=strategy)
        ensemble = DistributionEnsemble(self.model, initial_ensemble_size, rng=rng)
        _exec = exec or _DEFAULT_EXECUTOR
        future: concurrent.futures.Future[EvaluationResult] = _exec.submit(
            self.execute_plan,
            plan,
            ensemble,
            parameters=parameters,
            functions=functions,
            backend=backend,
        )
        return AsyncEvaluationHandle(
            future=future,
            evaluation=self,
            plan=plan,
            rng=rng,
            parameters=parameters,
            functions=functions,
            backend=backend,
        )

    def evaluate(
        self,
        scenarios: AxisEnsemble | Ensemble | None = None,
        nodes_of_interest: list[GenericIndex] | None = None,
        *,
        parameters: dict[GenericIndex, np.ndarray] | None = None,
        axes: dict[GenericIndex, np.ndarray] | None = None,
        ensemble: AxisEnsemble | Ensemble | None = None,
        functions: dict[str, executor.Functor] | None = None,
        backend: type[executor.NumpyBackend] = executor.NumpyBackend,
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
            Optional user-defined functions passed to the executor.  Wrap
            callables with :meth:`~executor.NumpyBackend.adapt` before passing.
        backend:
            The computation backend to use.  Currently only
            :class:`~executor.NumpyBackend` is supported (the default).

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

        # --- build plan and execute ---
        plan = self.build_plan(nodes_of_interest)
        return self.execute_plan(plan, ensemble, parameters=parameters, functions=functions, backend=backend)
