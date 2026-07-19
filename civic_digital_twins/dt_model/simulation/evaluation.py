"""Generic model evaluation."""
# SPDX-License-Identifier: Apache-2.0

import inspect
from collections.abc import Callable
from typing import Any

import numpy as np

from ..axes import PARAMETER, TIME_AXIS, Axis
from ..engine.frontend import graph, linearize
from ..engine.numpybackend import executor
from ..model.index import GenericIndex
from ..model.model import Model
from ..model.model_variant import ModelVariant
from .axis_layout import AxisLayout
from .ensemble import AxisEnsemble
from .plan import EvaluationPlan, Region, RegionGuard
from .region_execution import RegionArrayOps
from .scenario import Scenario

__all__ = ["EvaluationResult", "Evaluation"]


class EvaluationResult:
    """Result of :meth:`Evaluation.evaluate`.

    Wraps the executor :class:`~executor.State` and provides typed access to
    node values and weighted marginalization over ENSEMBLE and PARAMETER axes.

    Parameters
    ----------
    state:
        The executor state after evaluation.
    axis_layout:
        The :class:`~simulation.axis_layout.AxisLayout` of result arrays.
        A ``{Axis: position}`` dict (with sizes in *axis_sizes*) is also
        accepted transitionally and converted internally.
    parameter_arrays:
        Anonymous PARAMETER-axis arrays from ``parameters=`` (array-valued
        entries only; callable-backed indexes are not included).  Used by
        :meth:`parameter_values_for`.  Empty dict when no anonymous PARAMETER
        axes.
    axis_sizes:
        Maps each :class:`~dt_model.axes.Axis` to its size.  Only used (and
        required) with the transitional dict form of *axis_layout*.
    factorized_weights:
        Per-ENSEMBLE-axis weight vectors.
    named_axis_values:
        Raw 1-D arrays for named axes declared via ``parameter_axes=``, keyed
        by axis name.  Empty dict when ``parameter_axes=`` was not used.
    """

    def __init__(
        self,
        state: executor.State,
        axis_layout: AxisLayout | dict[Axis, int],
        parameter_arrays: dict[GenericIndex, np.ndarray],
        axis_sizes: dict[Axis, int] | None = None,
        factorized_weights: dict[Axis, np.ndarray] | None = None,
        named_axis_values: dict[str, np.ndarray] | None = None,
    ) -> None:
        self._state = state
        if isinstance(axis_layout, AxisLayout):
            self._layout = axis_layout
        else:
            # Transitional dict form (handle.py, runner.py); to be removed
            # once all constructors pass an AxisLayout.
            self._layout = AxisLayout.from_positions(axis_layout, axis_sizes or {})
        self._parameter_arrays = parameter_arrays
        self._factorized_weights: dict[Axis, np.ndarray] = factorized_weights or {}
        self._named_axis_values: dict[str, np.ndarray] = named_axis_values or {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def layout(self) -> AxisLayout:
        """The :class:`~simulation.axis_layout.AxisLayout` of result arrays."""
        return self._layout

    # Transitional views over the layout, kept while handle.py and runner.py
    # still consume the dict form; to be removed at the end of the refactor.

    @property
    def _axis_layout(self) -> dict[Axis, int]:
        return self._layout.positions

    @property
    def _axis_sizes(self) -> dict[Axis, int]:
        return self._layout.sizes

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
    def factorized_weights(self) -> dict[Axis, np.ndarray]:
        """Per-ENSEMBLE-axis weight vectors, keyed by axis.

        The joint scenario weight array (outer product of these vectors) is
        available via :attr:`weights`.
        """
        return self._factorized_weights

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
        return self._layout.full_shape

    def __getitem__(self, index: GenericIndex) -> np.ndarray:
        """Return the result array for *index*."""
        return np.asarray(self._state.values[index.node])

    def _contract_ensemble(self, index: GenericIndex) -> np.ndarray:
        """Contract all ENSEMBLE axes and return the ``(*P, *D)`` array.

        Delegates to :meth:`~simulation.axis_layout.AxisLayout.contract_ensemble`
        with this result's factorized weights.  The result shape is
        ``(*PARAMETER, *DOMAIN)`` — all DOMAIN dimensions are preserved
        regardless of size.
        """
        arr = np.asarray(self._state.values[index.node])
        return self._layout.contract_ensemble(arr, self._factorized_weights)

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
        return self._layout.drop_stray_domain(arr, index.output_axes)


class Evaluation:
    """Bridge between a :class:`~simulation.scenario.Scenario` and the engine.

    Given a scenario, :meth:`build_plan` encodes the
    DAG navigation strategy as an :class:`~simulation.plan.EvaluationPlan`, and
    :meth:`execute_plan` runs it against a given ensemble and parameter grid,
    returning an :class:`EvaluationResult`.  :meth:`evaluate` is a thin
    convenience wrapper that calls both in sequence.

    This class knows nothing about grids, presence variables, sustainability,
    or constraints — all domain-specific logic lives in subclasses or
    vertical-specific wrappers.

    Parameters
    ----------
    scenario:
        The :class:`~simulation.scenario.Scenario` to evaluate.
    """

    def __init__(self, scenario: Scenario) -> None:
        if not isinstance(scenario, Scenario):
            raise TypeError(f"{type(self).__name__} expects a Scenario, got {type(scenario).__name__}")
        self._scenario = scenario
        self.model: Model | ModelVariant = scenario.model

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
                Recursively split at
                :class:`~engine.frontend.graph.variant_selector` boundaries.
                At each nesting level: one shared region (pre-selector nodes,
                guarded by all ancestor branch keys), one guarded region per
                variant branch, and one merge region.  Both flat and nested
                :class:`~model.model_variant.ModelVariant` graphs are
                supported.

        Returns
        -------
        EvaluationPlan
            An :class:`~simulation.plan.EvaluationPlan` ready for
            :meth:`execute_plan`.

        Raises
        ------
        ValueError
            If *strategy* is not a recognised value.
        ValueError
            If *strategy* is ``"regional"`` and no
            :class:`~engine.frontend.graph.variant_selector` node exists
            (use ``strategy='monolithic'`` for plain models).
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
            # Regional partitioning — recursive algorithm
            #
            # Single-level example (used in inline comments):
            #   mode  = CategoricalIndex("mode", {"bike": 0.3, "train": 0.7})
            #   mv    = ModelVariant("Transport",
            #               {"bike": BikeModel(cap_bike),
            #                "train": TrainModel(cap_train)},
            #               selector=mode)
            #
            # ModelVariant.__init__ builds one variant_selector node (vs) and
            # one exclusive_multi_clause_where node per output field (mcw).
            #
            # vs carries:
            #   vs.selector_node  = mode.node
            #   vs.branch_map     = {"bike":  [tp_bike.node,  em_bike.node],
            #                        "train": [tp_train.node, em_train.node]}
            #   vs.merge_nodes    = [mcw_tp, mcw_em]
            #
            # For nested ModelVariants, the "car" branch in the example above
            # could itself be a ModelVariant("Policy", {"strict": ..., "loose":
            # ...}, selector=policy).  In that case, linearize.forest produces
            # inner_vs before outer_vs in topological order (outer_vs depends
            # on inner_mcw which depends on inner_vs), so choosing the LAST
            # variant_selector in topological order always picks the outermost.
            # ---------------------------------------------------------------

            if not any(isinstance(n, graph.variant_selector) for n in linearized_nodes):
                raise ValueError(
                    "build_plan(strategy='regional') requires a ModelVariant with a "
                    "runtime selector. No variant_selector found — use strategy='monolithic'."
                )

            def _is_ts(nodes: tuple[graph.Node, ...]) -> bool:
                return any(isinstance(n, (graph.timeseries_constant, graph.timeseries_placeholder)) for n in nodes)

            collected_regions: list[Region] = []
            collected_deps: list[frozenset[int]] = []

            def _partition(
                scope: set[graph.Node],
                inherited_guards: tuple[RegionGuard, ...],
                scope_entry: frozenset[int],
            ) -> frozenset[int]:
                """Recursively partition *scope* into regions.

                Parameters
                ----------
                scope:
                    Set of computation-graph nodes to partition at this level.
                inherited_guards:
                    Guard chain accumulated from outer nesting levels.  Every
                    region emitted in this scope carries these guards.
                scope_entry:
                    Region indices that must complete before the first region
                    in this scope can execute (i.e., the "input frontier").

                Returns
                -------
                frozenset[int]
                    The "output frontier" of this scope: the index of the merge
                    region (or the single flat region in the base case).  Used
                    by the caller to wire the parent merge's dependencies.
                """
                # Find all variant_selectors within this scope.
                # Topological order is preserved because linearized_nodes was
                # produced by linearize.forest; the last vs is the outermost.
                vs_in_scope = [n for n in linearized_nodes if n in scope and isinstance(n, graph.variant_selector)]

                if not vs_in_scope:
                    # Base case: no further variant structure.  One flat region.
                    topo = tuple(n for n in linearized_nodes if n in scope)
                    if not topo:
                        return scope_entry  # pragma: no cover
                    idx = len(collected_regions)
                    collected_regions.append(Region(nodes=topo, has_timeseries=_is_ts(topo), guards=inherited_guards))
                    collected_deps.append(scope_entry)
                    return frozenset({idx})

                vs = vs_in_scope[-1]  # outermost = last in topological order

                # Shared region: selector node + condition nodes from MCW clauses.
                # These depend only on the selector placeholder and string constants,
                # so they are safe to evaluate for all coordinates where
                # inherited_guards hold (before the branch split).
                cond_nodes = [
                    cond
                    for mcw in vs.merge_nodes
                    if isinstance(mcw, graph.exclusive_multi_clause_where)
                    for cond, _ in mcw.clauses
                ]
                shared_ns = set(linearize.forest(vs.selector_node, *cond_nodes)) & scope
                shared_topo = tuple(n for n in linearized_nodes if n in shared_ns)

                shared_idx = len(collected_regions)
                collected_regions.append(
                    Region(nodes=shared_topo, has_timeseries=_is_ts(shared_topo), guards=inherited_guards)
                )
                collected_deps.append(scope_entry)

                branch_entry = scope_entry | {shared_idx}
                scope_indices: set[int] = {shared_idx}

                # Per-branch node sets: traverse from branch outputs back to the
                # shared boundary, then subtract shared nodes (boundary nodes must
                # not appear in both shared and branch regions).
                branch_node_sets: dict[str, set[graph.Node]] = {}
                for key, branch_outputs in vs.branch_map.items():
                    bns = set(linearize.forest(*branch_outputs, boundary=shared_ns)) - shared_ns
                    bns &= scope
                    branch_node_sets[key] = bns

                # Recurse into each branch with one additional guard.
                for key in vs.branch_map:
                    branch_guard = RegionGuard(selector_node=vs.selector_node, branch_key=key)
                    frontier = _partition(
                        scope=branch_node_sets[key],
                        inherited_guards=(*inherited_guards, branch_guard),
                        scope_entry=branch_entry,
                    )
                    scope_indices |= frontier

                # Merge region: everything in scope not claimed by shared or branches.
                # This includes vs itself and the mcw output nodes.
                already = shared_ns | set().union(*branch_node_sets.values())
                merge_ns = scope - already
                merge_topo = tuple(n for n in linearized_nodes if n in merge_ns)

                merge_idx = len(collected_regions)
                collected_regions.append(
                    Region(nodes=merge_topo, has_timeseries=_is_ts(merge_topo), guards=inherited_guards)
                )
                collected_deps.append(frozenset(scope_indices))
                return frozenset({merge_idx})

            _partition(
                scope=set(linearized_nodes),
                inherited_guards=(),
                scope_entry=frozenset(),
            )

            return EvaluationPlan(
                model=self.model,
                nodes_of_interest=tuple(nodes_of_interest),
                regions=tuple(collected_regions),
                dependencies=tuple(collected_deps),
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
            The :class:`AxisEnsemble` to evaluate, or ``None`` for
            deterministic evaluation.
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
        param_axis_entries: list[tuple[Axis, int]] = []  # named axes first, then anonymous
        ens_axis_entries: list[tuple[Axis, int]] = []
        factorized_weights: dict[Axis, np.ndarray] = {}
        c_subs: dict[graph.Node, np.ndarray] = {}
        param_nodes: list[graph.Node] = []  # anonymous array param nodes
        callable_nodes: list[graph.Node] = []  # callable-backed nodes (no new axis)

        # Named PARAMETER axes — positions 0..k-1.
        # Build broadcast-ready shaped arrays (singleton at every position except own).
        named_shaped: dict[str, np.ndarray] = {}
        for i, (name, arr) in enumerate(parameter_axes.items()):
            param_axis_entries.append((Axis(name, PARAMETER), arr.size))
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
            param_axis_entries.append((Axis(getattr(idx, "name", f"param_{k + j}"), PARAMETER), arr.size))
            shape = [1] * n_params
            shape[k + j] = arr.size
            c_subs[idx.node] = arr.reshape(shape)
            param_nodes.append(idx.node)

        n_ensemble = 0
        ens_subs: dict[graph.Node, np.ndarray] = {}

        if ensemble is not None:
            ens_assignments = ensemble.assignments()
            n_ensemble = len(ensemble.ensemble_axes)
            for ax, w in zip(ensemble.ensemble_axes, ensemble.ensemble_weights):
                ens_axis_entries.append((ax, w.size))
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

        # Leading evaluation layout: (*PARAMETER, *ENSEMBLE).  The canonical
        # role ordering is enforced by the AxisLayout constructor; the DOMAIN
        # time axis is appended after execution, once T is known.  Guarded
        # regions gather/scatter arbitrary leading-axis coordinates via
        # RegionArrayOps, so selectors may vary along PARAMETER axes and
        # ensembles may span multiple ENSEMBLE axes.
        leading_layout = AxisLayout.build(parameters=param_axis_entries, ensemble=ens_axis_entries)
        n_total = leading_layout.n_leading + extra_ts
        region_ops = RegionArrayOps(leading_layout, has_timeseries=_has_timeseries)

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
            if not region.guards:
                # Unconditional region — evaluate for all scenarios.
                executor.evaluate_nodes(state, *region.nodes)
            else:
                # Guarded region — evaluate only for coordinates where every
                # guard's selector matches its branch key (AND of all masks).
                # For single-level plans region.guards has one element; for
                # nested variants it has one entry per nesting level.
                mask = np.ones(leading_layout.leading_shape, dtype=bool)
                for guard in region.guards:
                    sel_val = state.values[guard.selector_node]
                    guard_mask = region_ops.selector_mask(guard.selector_node, sel_val, guard.branch_key)
                    mask = mask & np.asarray(guard_mask)
                flat_mask = np.asarray(mask).reshape(-1)
                branch_idx = np.flatnonzero(flat_mask)

                if branch_idx.size == 0:
                    # No coordinates fall into this branch.  Pre-initialize
                    # missing branch nodes with broadcast-compatible inactive
                    # arrays so merge-region np.select can still reference them.
                    for node in region.nodes:
                        if node not in state.values:
                            state.values[node] = region_ops.empty_branch_value()
                            substituted_nodes.add(node)  # prevent spurious shape-norm
                    continue

                # Build a branch-local state by gathering every already-known
                # value to the matching leading coordinates.  Constants and
                # structural variant_selector sentinels are carried unchanged.
                branch_values: dict[graph.Node, np.ndarray] = {
                    node: region_ops.gather(node, value, branch_idx) for node, value in state.values.items()
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
                    state.values[node] = region_ops.scatter(node, branch_state.values[node], branch_idx)
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

        # DOMAIN axis tracking: append TIME_AXIS to the layout so that every
        # result dimension is named.  T is read post-execution because
        # abstract TimeseriesIndex nodes are filled at evaluate time.
        # Assumption: T is uniform across all PARAMETER configurations (T is a
        # structural property of the model, not a function of parameter values).
        result_layout = leading_layout
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
            result_layout = leading_layout.with_axis_appended(TIME_AXIS, T)
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
                assert result_layout.compatible_with(arr.shape), (
                    f"Post-norm: node {getattr(node, 'name', repr(node))!r} "
                    f"shape {arr.shape} is incompatible with layout {result_layout!r} "
                    f"(each dim must be 1 or the axis size)"
                )

        return EvaluationResult(
            state,
            result_layout,
            array_params,
            factorized_weights=factorized_weights,
            named_axis_values=parameter_axes if parameter_axes else None,
        )

    def evaluate(
        self,
        nodes_of_interest: list[GenericIndex] | None = None,
        *,
        parameters: dict[GenericIndex, Any] | None = None,
        parameter_axes: dict[str, np.ndarray] | None = None,
        ensemble: AxisEnsemble | None = None,
        functions: dict[str, executor.Functor] | None = None,
        backend: type[executor.NumpyBackend] = executor.NumpyBackend,
    ) -> EvaluationResult:
        """Evaluate *nodes_of_interest* over the given ensemble.

        Parameters
        ----------
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
        ensemble:
            The :class:`AxisEnsemble` to evaluate, or ``None`` for
            deterministic evaluation (no ENSEMBLE axes).
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
        ValueError
            If any non-parameter abstract index is not resolved in a scenario.
        """
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

        # --- build plan and execute ---
        plan = self.build_plan(nodes_of_interest)
        return self.execute_plan(
            plan, ensemble, parameters=parameters, parameter_axes=parameter_axes, functions=functions, backend=backend
        )
