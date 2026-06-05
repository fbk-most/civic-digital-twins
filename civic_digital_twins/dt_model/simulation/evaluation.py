"""Generic model evaluation."""
# SPDX-License-Identifier: Apache-2.0

import inspect
import warnings
from collections.abc import Callable, Mapping
from typing import Any, assert_never

import numpy as np

from ..engine.frontend import graph, linearize
from ..engine.numpybackend import executor
from ..model.axis import DOMAIN, ENSEMBLE, PARAMETER, Axis
from ..model.index import GenericIndex
from ..model.model import Model
from ..model.model_variant import ModelVariant
from .ensemble import AxisEnsemble, Ensemble, WeightedScenario
from .plan import EvaluationPlan, Region, RegionGuard
from .scenario import Scenario

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
        Anonymous PARAMETER-axis arrays from ``parameters=`` (array-valued
        entries only; callable-backed indexes are not included).  Used by
        :meth:`parameter_values_for`.  Empty dict when no anonymous PARAMETER
        axes.
    axis_sizes:
        Maps each :class:`~dt_model.model.axis.Axis` to its size.
    factorized_weights:
        Per-ENSEMBLE-axis weight vectors.
    named_axis_values:
        Raw 1-D arrays for named axes declared via ``parameter_axes=``, keyed
        by axis name.  Empty dict when ``parameter_axes=`` was not used.
    """

    def __init__(
        self,
        state: executor.State,
        axis_layout: dict[Axis, int],
        parameter_arrays: dict[GenericIndex, np.ndarray],
        axis_sizes: dict[Axis, int] | None = None,
        factorized_weights: dict[Axis, np.ndarray] | None = None,
        named_axis_values: dict[str, np.ndarray] | None = None,
    ) -> None:
        self._state = state
        self._axis_layout = axis_layout
        self._parameter_arrays = parameter_arrays
        self._axis_sizes: dict[Axis, int] = axis_sizes or {}
        self._factorized_weights: dict[Axis, np.ndarray] = factorized_weights or {}
        self._named_axis_values: dict[str, np.ndarray] = named_axis_values or {}

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

    @property
    def named_axis_values(self) -> dict[str, np.ndarray]:
        """Raw 1-D arrays for axes declared via ``parameter_axes=``, keyed by name."""
        return self._named_axis_values

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
        """Shape of a fully-broadcast result array in axis-layout order."""
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

    def _contract_ensemble(self, index: GenericIndex) -> np.ndarray:
        """Contract all ENSEMBLE axes and return the ``(*P, *D)`` array.

        For each ENSEMBLE axis (in descending position order so earlier
        squeezes do not shift later positions): if the axis size is 1 the
        singleton is squeezed away directly; otherwise a weighted average is
        taken.  The result shape is ``(*PARAMETER, *DOMAIN)`` — all DOMAIN
        dimensions are preserved regardless of size.
        """
        arr = np.asarray(self._state.values[index.node])
        for ax, pos in sorted(
            ((a, p) for a, p in self._axis_layout.items() if a.role == ENSEMBLE),
            key=lambda t: t[1],
            reverse=True,
        ):
            arr = (
                arr.squeeze(axis=pos)
                if arr.shape[pos] == 1
                else np.average(arr, weights=self._factorized_weights[ax], axis=pos)
            )
        return arr

    def expected_value(self, index: GenericIndex) -> np.ndarray:
        """Return the typed result for *index* after contracting ENSEMBLE axes.

        Contracts all ENSEMBLE axes (weighted average or singleton squeeze),
        then drops size-1 DOMAIN dimensions that *index* does not carry in its
        :attr:`~model.index.GenericIndex.output_axes`:

        - A :class:`~dt_model.model.index.TimeseriesIndex` carries
          ``Axis("time", DOMAIN)``; result shape is ``(*PARAMETER, T)``.
        - A plain :class:`~dt_model.model.index.Index` formula carries no
          DOMAIN axes; size-1 DOMAIN dims are squeezed away; result shape is
          ``(*PARAMETER,)``.

        This is the primary result-extraction method for user code and
        vertical applications.  Use :meth:`_contract_ensemble` directly if
        you need the full ``(*PARAMETER, *DOMAIN)`` shape.
        """
        arr = self._contract_ensemble(index)
        n_params = sum(1 for ax in self._axis_layout if ax.role == PARAMETER)
        domain_axes = sorted(
            [(ax, pos) for ax, pos in self._axis_layout.items() if ax.role == DOMAIN],
            key=lambda t: t[1],
        )
        stray = tuple(
            n_params + i
            for i, (ax, _) in enumerate(domain_axes)
            if ax not in index.output_axes and arr.shape[n_params + i] == 1
        )
        if stray:
            arr = np.squeeze(arr, axis=stray)
        return arr

    def marginalize(self, index: GenericIndex) -> np.ndarray:
        """Use :meth:`expected_value` instead — ``marginalize()`` is deprecated.

        Currently equivalent to ``expected_value(index)``.
        """
        warnings.warn(
            "EvaluationResult.marginalize() is deprecated. Use expected_value() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.expected_value(index)


class Evaluation:
    """Bridge between a :class:`~simulation.scenario.Scenario` and the engine.

    Given a scenario (or a model, deprecated), :meth:`build_plan` encodes the
    DAG navigation strategy as an :class:`~simulation.plan.EvaluationPlan`, and
    :meth:`execute_plan` runs it against a given ensemble and parameter grid,
    returning an :class:`EvaluationResult`.  :meth:`evaluate` is a thin
    convenience wrapper that calls both in sequence.

    This class knows nothing about grids, presence variables, sustainability,
    or constraints — all domain-specific logic lives in subclasses or
    vertical-specific wrappers.

    Parameters
    ----------
    scenario_or_model:
        A :class:`~simulation.scenario.Scenario` (canonical) or, deprecated,
        a :class:`~model.model.Model` / :class:`~model.model_variant.ModelVariant`
        which is auto-wrapped in ``Scenario(model)`` with a
        :class:`DeprecationWarning`.
    """

    def __init__(self, scenario_or_model: Scenario | Model | ModelVariant) -> None:
        scenario: Scenario
        model: Model | ModelVariant
        if isinstance(scenario_or_model, Scenario):
            scenario = scenario_or_model
            model = scenario_or_model.model
        elif isinstance(scenario_or_model, (Model, ModelVariant)):
            warnings.warn(
                "Passing a Model or ModelVariant directly to Evaluation() is deprecated and will be removed "
                "in a future version. Wrap it in Scenario(model) first: Evaluation(Scenario(model)).",
                DeprecationWarning,
                stacklevel=2,
            )
            model = scenario_or_model
            scenario = Scenario(model)
        else:
            assert_never(scenario_or_model)
        self._scenario = scenario
        self.model: Model | ModelVariant = model

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
                raise NotImplementedError(  # pragma: no cover
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
        parameters: dict[GenericIndex, Any] | None = None,
        parameter_axes: dict[str, np.ndarray] | None = None,
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
            Per-index value sources.  Each entry maps a
            :class:`~dt_model.model.index.GenericIndex` to either a 1-D numpy
            array (anonymous PARAMETER axis — current behaviour) or a callable
            (correlated value computed from named axes declared in
            *parameter_axes*).  Callables receive broadcast-ready shaped arrays
            for each named axis whose name appears in the callable's signature;
            parameters with defaults whose names are not axis names are ignored.
            Callables are only valid when *parameter_axes* is also provided.
        parameter_axes:
            Named PARAMETER axes for correlated sweeps.  Maps axis name to a
            1-D numpy array of axis values.  Named axes occupy the leading
            dimensions of result arrays (before anonymous PARAMETER axes).
            Access the raw arrays via :attr:`~EvaluationResult.named_axis_values`
            on the returned result.
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
            parameter_axes=parameter_axes,
            functions=functions,
            backend=backend,
        )

    def _execute_plan(
        self,
        plan: EvaluationPlan,
        ensemble: AxisEnsemble | None,
        *,
        parameters: dict[GenericIndex, Any],
        parameter_axes: dict[str, np.ndarray] | None,
        functions: dict[str, executor.Functor] | None,
        backend: type[executor.NumpyBackend],
    ) -> EvaluationResult:
        """Execute an :class:`~simulation.plan.EvaluationPlan`."""
        actual_nodes = [idx.node for idx in plan.nodes_of_interest]
        _raw_scenario_subs = self._scenario.base_substitutions()
        # Filter out entries where base_substitutions() wrapped a graph.Node as a numpy
        # object array — this happens for formula-based Index instances whose `.value`
        # is an existing graph node rather than a concrete scalar or array.  Such nodes
        # do not need value injection: the executor handles them via their own evaluation
        # chain (or via graph.placeholder.default_value if set).
        scenario_subs: dict[graph.Node, np.ndarray] = {
            node: val
            for node, val in _raw_scenario_subs.items()
            if not (isinstance(val, np.ndarray) and val.ndim == 0 and val.dtype.kind == "O")
        }
        _has_timeseries = any(r.has_timeseries for r in plan.regions)

        # Separate callable entries (correlated axes) from plain array entries.
        parameter_axes = parameter_axes or {}
        callable_params: dict[GenericIndex, Callable[..., Any]] = {}
        array_params: dict[GenericIndex, np.ndarray] = {}
        for idx, val in parameters.items():
            if callable(val):
                callable_params[idx] = val
            else:
                array_params[idx] = np.asarray(val)
        if callable_params and not parameter_axes:
            names = ", ".join(repr(getattr(idx, "name", repr(idx))) for idx in callable_params)
            raise ValueError(
                f"Callable values in parameters= require parameter_axes= to be provided "
                f"(indexes with callable values: {names})."
            )

        # Validate that parameters= does not contain constant-node indexes.
        # ConstIndex and ConstTimeseriesIndex bake their value into a graph.constant
        # node; the executor evaluates constant nodes directly and never consults
        # state.values, so substituting them has no effect.  Placeholder-backed
        # indexes (Index, DistributionIndex, TimeseriesIndex — with or without a
        # default value) are fine because the executor does read their state entry.
        from ..model.index import ConstIndex, ConstTimeseriesIndex

        const_params = [idx for idx in parameters if isinstance(idx, (ConstIndex, ConstTimeseriesIndex))]
        if const_params:
            names = ", ".join(repr(getattr(idx, "name", repr(idx))) for idx in const_params)
            raise ValueError(
                f"The following indexes passed in parameters= are constant-node indexes "
                f"whose values are baked into the computation graph and cannot be "
                f"overridden at evaluate time: {names}. "
                "Use Index(name, value) or Index(name, None) for sweep parameters."
            )

        # Membership check: every parameters= key's node must appear in the plan's
        # computation graph (i.e., in the linearised nodes across all regions).
        # An index whose node is absent from the plan would contribute a PARAMETER axis
        # to the result layout but would never influence any computation — a silent bug.
        # Note: we check node identity in the plan rather than model.indexes membership
        # because a valid use-case is passing an index that is only referenced in a
        # selector formula (and therefore not declared in model.indexes but still
        # present in the linearised graph).
        all_plan_nodes = {n for r in plan.regions for n in r.nodes}
        orphan_params = [idx for idx in parameters if idx.node not in all_plan_nodes]
        if orphan_params:
            names = ", ".join(repr(getattr(idx, "name", repr(idx))) for idx in orphan_params)
            raise ValueError(
                f"parameters= for model {plan.model.name!r}: {names} "
                f"{'is' if len(orphan_params) == 1 else 'are'} not part of this "
                "model's computation. Passing "
                f"{'it' if len(orphan_params) == 1 else 'them'} would add result "
                "dimensions that have no effect on any output value. Check that "
                "you are using index objects that belong to this model."
            )

        k = len(parameter_axes)  # named PARAMETER axes
        m = len(array_params)  # anonymous PARAMETER axes
        n_params = k + m
        axis_layout: dict[Axis, int] = {}
        axis_sizes: dict[Axis, int] = {}
        factorized_weights: dict[Axis, np.ndarray] = {}
        c_subs: dict[graph.Node, np.ndarray] = {}
        param_nodes: list[graph.Node] = []  # anonymous array param nodes
        callable_nodes: list[graph.Node] = []  # callable-backed nodes (no new axis)

        # Named PARAMETER axes — positions 0..k-1.
        # Build broadcast-ready shaped arrays (singleton at every position except own).
        named_shaped: dict[str, np.ndarray] = {}
        for i, (name, arr) in enumerate(parameter_axes.items()):
            ax = Axis(name, PARAMETER)
            axis_layout[ax] = i
            axis_sizes[ax] = arr.size
            shape = [1] * k
            shape[i] = arr.size
            named_shaped[name] = arr.reshape(shape)

        # Callable entries — substitute standard model indexes using named axis arrays.
        # Each callable receives the same broadcast-ready shaped arrays that _execute_plan
        # would supply to a formula node in the equivalent traditional model.
        for idx, fn in callable_params.items():
            sig = inspect.signature(fn)
            kwargs: dict[str, np.ndarray] = {}
            has_var_keyword = False
            for param_name, param in sig.parameters.items():
                if param.kind == inspect.Parameter.VAR_POSITIONAL:
                    raise TypeError(
                        f"Callable for index {getattr(idx, 'name', repr(idx))!r} uses *args; "
                        "use named keyword parameters instead."
                    )
                if param.kind == inspect.Parameter.VAR_KEYWORD:
                    has_var_keyword = True
                    break
                if param_name in named_shaped:
                    kwargs[param_name] = named_shaped[param_name]
                elif param.default is inspect.Parameter.empty:
                    raise ValueError(
                        f"Callable for index {getattr(idx, 'name', repr(idx))!r}: "
                        f"required parameter {param_name!r} is not a declared named axis."
                    )
                # else: has a default and not an axis name → uses its default
            if has_var_keyword:
                kwargs = dict(named_shaped)
            c_subs[idx.node] = np.asarray(fn(**kwargs))
            callable_nodes.append(idx.node)

        # Anonymous array PARAMETER axes — positions k..k+m-1.
        for j, (idx, arr) in enumerate(array_params.items()):
            ax = Axis(getattr(idx, "name", f"param_{k + j}"), PARAMETER)
            axis_layout[ax] = k + j
            axis_sizes[ax] = arr.size
            shape = [1] * n_params
            shape[k + j] = arr.size
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

        # Extend substitutions with trailing singleton dims for broadcasting:
        # - anonymous param nodes: shape (*P,) needs n_ensemble + extra_ts singletons.
        # - callable-backed nodes: shape (*named_P,) needs m + n_ensemble + extra_ts singletons.
        extra_ts = 1 if _has_timeseries else 0
        anon_trailing = (1,) * (n_ensemble + extra_ts)
        if anon_trailing:
            for node in param_nodes:
                c_subs[node] = c_subs[node].reshape(c_subs[node].shape + anon_trailing)
        callable_trailing = (1,) * (m + n_ensemble + extra_ts)
        if callable_trailing:
            for node in callable_nodes:
                c_subs[node] = c_subs[node].reshape(c_subs[node].shape + callable_trailing)

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

        # Guarded regions operate over the full leading evaluation layout:
        # (*PARAMETER, *ENSEMBLE).  The helpers below gather/scatter arbitrary
        # leading-axis coordinates, so selectors may vary along PARAMETER axes
        # and ensembles may span multiple ENSEMBLE axes.
        leading_axes = tuple(ax for ax, pos in sorted(axis_layout.items(), key=lambda item: item[1]) if pos < n_full)
        leading_shape = tuple(axis_sizes[ax] for ax in leading_axes)
        n_leading = int(np.prod(leading_shape, dtype=np.int64)) if leading_shape else 1

        def _has_domain_axis(node: graph.Node) -> bool:
            return any(ax.role == DOMAIN for ax in node.output_axes)

        def _normalise_leading(node: graph.Node, value: Any) -> np.ndarray:
            """Return *value* with explicit leading singleton axes when needed."""
            arr = np.asarray(value)
            if n_full == 0 or isinstance(node, graph.variant_selector):
                return arr
            has_domain = _has_domain_axis(node)
            # A raw DOMAIN-only value, e.g. a timeseries constant with shape (T,),
            # has no explicit PARAMETER/ENSEMBLE dimensions yet.  Prepend them
            # even when T accidentally equals a leading-axis size.
            if has_domain and arr.ndim == len(node.output_axes):
                return arr.reshape((1,) * n_full + arr.shape)
            if arr.ndim >= n_full and all(arr.shape[i] in {1, leading_shape[i]} for i in range(n_full)):
                return arr
            return arr.reshape((1,) * n_full + arr.shape)

        def _broadcast_to_leading(node: graph.Node, value: Any) -> np.ndarray:
            """Broadcast *value* to ``leading_shape + trailing_shape``."""
            arr = _normalise_leading(node, value)
            if n_full == 0 or isinstance(node, graph.variant_selector):
                return arr
            target = tuple(leading_shape[i] if arr.shape[i] == 1 else arr.shape[i] for i in range(n_full))
            return np.broadcast_to(arr, target + arr.shape[n_full:])

        def _leading_mask(selector_node: graph.Node, selector_value: Any, branch_key: str) -> np.ndarray:
            """Return a boolean mask over the full ``(*PARAMETER, *ENSEMBLE)`` layout."""
            sel = _broadcast_to_leading(selector_node, selector_value)
            if n_full == 0:
                mask = np.asarray(sel == branch_key)
                if mask.shape != ():
                    if any(dim > 1 for dim in mask.shape):
                        raise NotImplementedError(
                            "Regional execution does not support selectors with non-singleton DOMAIN axes."
                        )
                    mask = mask.reshape(())
                return mask
            trailing = sel.shape[n_full:]
            if trailing:
                if any(dim > 1 for dim in trailing):
                    raise NotImplementedError(
                        "Regional execution does not support selectors with non-singleton DOMAIN axes."
                    )
                sel = sel.reshape(sel.shape[:n_full])
            return np.broadcast_to(sel == branch_key, leading_shape)

        def _gather_leading(node: graph.Node, value: Any, flat_idx: np.ndarray) -> np.ndarray:
            """Gather selected leading-axis coordinates into a branch-local first axis."""
            if n_full == 0 or isinstance(node, graph.variant_selector):
                return np.asarray(value)
            arr = _broadcast_to_leading(node, value)
            flat = arr.reshape((n_leading,) + arr.shape[n_full:])
            return np.take(flat, flat_idx, axis=0)

        def _branch_fill_value(dtype: np.dtype) -> tuple[np.dtype, Any]:
            """Return an output dtype and inactive-branch fill value for *dtype*."""
            if dtype.kind in {"f", "c"}:
                return dtype, np.nan
            if dtype.kind == "b":
                return dtype, False
            if dtype.kind in {"i", "u"}:
                return dtype, 0
            return np.dtype(object), None

        def _scatter_leading(node: graph.Node, value: Any, flat_idx: np.ndarray) -> np.ndarray:
            """Scatter a branch-local value back into the full leading layout."""
            arr = np.asarray(value)
            if n_full == 0:
                return arr
            k = int(flat_idx.size)
            if arr.ndim == 0:
                arr = np.broadcast_to(arr, (k,)).copy()
            elif arr.shape[0] == 1 and k != 1:
                arr = np.broadcast_to(arr, (k,) + arr.shape[1:]).copy()
            elif arr.shape[0] != k:
                # DOMAIN-only values produced inside the branch (shape (T,)) are
                # invariant across selected leading coordinates.
                if _has_domain_axis(node):
                    arr = np.broadcast_to(arr, (k,) + arr.shape).copy()
                else:
                    raise ValueError(
                        f"Regional scatter for node {getattr(node, 'name', repr(node))!r}: "
                        f"branch result first dimension {arr.shape[0]} does not match selected size {k}."
                    )
            if extra_ts and not _has_domain_axis(node) and arr.ndim == 1:
                arr = arr.reshape(arr.shape + (1,))
            out_dtype, fill_value = _branch_fill_value(arr.dtype)
            full_flat = np.full((n_leading,) + arr.shape[1:], fill_value, dtype=out_dtype)
            full_flat[flat_idx] = arr.astype(out_dtype, copy=False)
            return full_flat.reshape(leading_shape + arr.shape[1:])

        def _empty_branch_value() -> np.ndarray:
            """Create a broadcast-compatible inactive value for an unselected branch."""
            trailing = (1,) if extra_ts else ()
            return np.full(leading_shape + trailing, np.nan, dtype=float)

        # Coverage validation (D_valid): every abstract index must have a value source.
        # We check only abstract indexes (value=None or Distribution-backed) — NOT
        # concrete-value placeholder nodes that may be "orphan" (not in model.indexes).
        # Concrete-value orphan placeholders are handled via graph.placeholder.default_value
        # or graph.timeseries_placeholder.default_values auto-populated at creation time.
        abstract_nodes = {idx.node for idx in self._scenario.abstract_indexes()}
        covered = set(scenario_subs.keys()) | set(c_subs.keys())
        uncovered_abstract = abstract_nodes - covered
        if uncovered_abstract:
            node_to_idx = {idx.node: idx for idx in self._scenario.abstract_indexes()}
            names = sorted(getattr(node_to_idx.get(n, n), "name", repr(n)) for n in uncovered_abstract)
            raise ValueError(
                f"The following abstract indexes are not covered by Scenario, parameters=, or ensemble: "
                f"{', '.join(repr(n) for n in names)}"
            )

        # Overlap check: parameters= and Scenario.overrides must not overlap.
        param_idx_ids = {id(idx) for idx in parameters.keys()}
        override_idx_ids = {id(idx) for idx in self._scenario._overrides.keys()}
        overlap_ids = param_idx_ids & override_idx_ids
        if overlap_ids:
            overlapping = [idx for idx in parameters.keys() if id(idx) in overlap_ids]
            names_str = ", ".join(repr(getattr(idx, "name", repr(idx))) for idx in overlapping)
            raise ValueError(f"The following indexes appear in both parameters= and Scenario.overrides: {names_str}")

        state = executor.State(
            {**scenario_subs, **c_subs},
            functions=functions or {},
            node_functions=getattr(plan.model, "_node_functions", {}),
        )

        # Execute regions in topological order.
        for region in plan.regions:
            if region.guard is None:
                # Unconditional region — evaluate for all scenarios.
                executor.evaluate_nodes(state, *region.nodes)
            else:
                # Guarded region — evaluate only for coordinates whose selector
                # value matches this branch.  The mask spans the entire leading
                # layout, so selectors may vary along PARAMETER and/or multiple
                # ENSEMBLE axes.
                guard = region.guard
                sel_val = state.values[guard.selector_node]
                mask = _leading_mask(guard.selector_node, sel_val, guard.branch_key)
                flat_mask = np.asarray(mask).reshape(-1)
                branch_idx = np.flatnonzero(flat_mask)

                if branch_idx.size == 0:
                    # No coordinates fall into this branch.  Pre-initialize
                    # missing branch nodes with broadcast-compatible inactive
                    # arrays so merge-region np.select can still reference them.
                    for node in region.nodes:
                        if node not in state.values:
                            state.values[node] = _empty_branch_value()
                            substituted_nodes.add(node)  # prevent spurious shape-norm
                    continue

                # Build a branch-local state by gathering every already-known
                # value to the matching leading coordinates.  Constants and
                # structural variant_selector sentinels are carried unchanged.
                branch_values: dict[graph.Node, np.ndarray] = {
                    node: _gather_leading(node, value, branch_idx) for node, value in state.values.items()
                }

                branch_state = executor.State(
                    branch_values,
                    functions=functions or {},
                    node_functions=getattr(plan.model, "_node_functions", {}),
                )
                executor.evaluate_nodes(branch_state, *region.nodes)

                # Scatter branch results back into full leading-shape arrays,
                # filling inactive coordinates with branch-neutral placeholders.
                for node in region.nodes:
                    if node not in branch_state.values or node in state.values:
                        continue
                    state.values[node] = _scatter_leading(node, branch_state.values[node], branch_idx)
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

        # DOMAIN axis tracking: register Axis("time", DOMAIN) in axis_layout
        # so that every result dimension is named.  T is read post-execution
        # because abstract TimeseriesIndex nodes are filled at evaluate time.
        # Assumption: T is uniform across all PARAMETER configurations (T is a
        # structural property of the model, not a function of parameter values).
        if _has_timeseries:
            ts_nodes = [
                n
                for r in plan.regions
                for n in r.nodes
                if isinstance(n, (graph.timeseries_constant, graph.timeseries_placeholder))
            ]
            T = int(np.asarray(state.values[ts_nodes[0]]).shape[-1])
            if __debug__:
                for ts_n in ts_nodes:
                    T_n = int(np.asarray(state.values[ts_n]).shape[-1])
                    assert T_n == T, (
                        f"Non-uniform timeseries length: node "
                        f"{getattr(ts_n, 'name', repr(ts_n))!r} has T={T_n}, "
                        f"expected T={T}. T must be constant across all PARAMETER "
                        f"configurations (it is a structural model property)."
                    )
            time_axis = Axis("time", DOMAIN)
            axis_layout[time_axis] = n_full
            axis_sizes[time_axis] = T
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
            array_params,
            axis_sizes=axis_sizes,
            factorized_weights=factorized_weights,
            named_axis_values=parameter_axes if parameter_axes else None,
        )

    def evaluate(
        self,
        scenarios: AxisEnsemble | Ensemble | None = None,
        nodes_of_interest: list[GenericIndex] | None = None,
        *,
        parameters: dict[GenericIndex, Any] | None = None,
        parameter_axes: dict[str, np.ndarray] | None = None,
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
            Per-index value sources.  Each entry maps a
            :class:`~dt_model.model.index.GenericIndex` to either a 1-D numpy
            array (anonymous PARAMETER axis — current behaviour) or a callable
            (correlated value computed from named axes declared in
            *parameter_axes*).  Callables receive broadcast-ready shaped arrays
            for each named axis whose name appears in the callable's signature;
            parameters with defaults whose names are not axis names are ignored.
            Callables are only valid when *parameter_axes* is also provided.
        parameter_axes:
            Named PARAMETER axes for correlated sweeps.  Maps axis name to a
            1-D numpy array of axis values.  Named axes occupy the leading
            dimensions of result arrays (before anonymous PARAMETER axes).
            Access the raw arrays via :attr:`~EvaluationResult.named_axis_values`
            on the returned result.
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

        # Enforce: scenario.parameter_axes must all be covered by parameters=.
        if self._scenario.parameter_axes:
            param_set_check = set(parameters.keys())
            missing = [idx for idx in self._scenario.parameter_axes if idx not in param_set_check]
            if missing:
                names = ", ".join(repr(getattr(idx, "name", repr(idx))) for idx in missing)
                raise ValueError(
                    f"Scenario declares {names} as parameter_axes but "
                    f"{'it was' if len(missing) == 1 else 'they were'} not supplied in "
                    "parameters=. Pass their values via "
                    "Evaluation.evaluate(parameters={idx: values, ...})."
                )

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
        return self.execute_plan(
            plan, ensemble, parameters=parameters, parameter_axes=parameter_axes, functions=functions, backend=backend
        )
