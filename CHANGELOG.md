# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

**Model I/O contract**

- `Model.__init__` accepts `inputs=`, `outputs=`, and `expose=` keyword
  arguments.  Each is an instance of an inner `@dataclass` (`Inputs`, `Outputs`,
  `Expose`) declared on the subclass with typed fields.  `Model` inspects them
  via `dataclasses.fields()` to build the proxies and derive `indexes`
  automatically — no flat index list required.
- **Three access levels**:
  1. `model.outputs.<field>` / `model.inputs.<field>` — contractual interface,
     declared via `Outputs` / `Inputs` inner dataclasses.  Stable across versions.
  2. `model.expose.<field>` — inspectable but not contractual, declared via the
     optional `Expose` inner dataclass.  May change between versions.
  3. Local variables inside `__init__` — internal to the engine graph only;
     not accessible from outside.
- **List and dict field values**: dataclass fields may hold a single
  `GenericIndex`, a `list[GenericIndex]`, or a `dict[str, GenericIndex]`.
  Iteration, `len()`, and `in` flatten these to scalar indexes only, preserving
  the existing engine contract.  Field access returns the raw value.
- **`indexes` derived automatically**: the flat `model.indexes` list is built by
  collecting and deduplicating all scalar `GenericIndex` values from `inputs`,
  `outputs`, and `expose` (first-seen order).
- **Construction-time validation**: every entry in `inputs` and `outputs` must
  appear in `indexes` (identity check); declaring the same entry twice is an
  error.  A descriptive `ValueError` is raised in all cases.
- **Legacy `indexes=` path deprecated**: passing `indexes` explicitly emits a
  `DeprecationWarning`.  The legacy path is preserved for backward compatibility
  and will be removed in a future version.
- **`IOProxy`** — read-only proxy exposing declared fields via attribute access,
  iteration, `len()`, and `in` membership.  `IOProxy` is generic (`IOProxy[DC]`);
  `__getattr__` returns `Any` so typed field access flows through to the caller's
  expected type without `cast()`.

**Bologna mobility example — modular rewrite**

- `mobility_bologna.py` decomposed into three sub-models with explicit typed
  interfaces:
  - `InflowModel` — policy-modified inflow and payment statistics.
  - `TrafficModel` — baseline and modified circulating traffic.
  - `EmissionsModel` — baseline and modified NOx emissions.
- `BolognaModel` wires the three sub-models via constructor arguments; `Expose`
  collects all sub-model indexes for the engine plus named timeseries fields for
  plotting.
- `compute_kpis` updated to use `m.outputs.*`.
- `__main__` updated to use `m.expose.*` for plot data; graphs saved via
  `fig.savefig()` (headless-safe).

## [0.7.0] - 2026-03-15

### Added

**Distribution indexes**

- `DistributionIndex(name, distribution, params)` — a single, distribution-agnostic
  index class that replaces the three distribution-specific classes removed below.
  `distribution` is any callable (e.g. a `scipy.stats` frozen-distribution factory)
  that accepts `**params` and returns a `Distribution`-conformant object; `params` is
  a `dict[str, Any]` forwarded verbatim to it, so scipy validates the values at
  construction time.  `DistributionIndex.params` supports full replacement
  (`idx.params = {"loc": 0, "scale": 1}`) and partial update via the Python
  dict-merge operator (`idx.params |= {"loc": 200}`).

**Engine layer — axis reduction operators**

- `project_using_min(node, axis)` — minimum value reduction along an axis.
- `project_using_max(node, axis)` — maximum value reduction along an axis.
- `project_using_std(node, axis)` — standard deviation reduction along an axis.
- `project_using_var(node, axis)` — variance reduction along an axis.
- `project_using_median(node, axis)` — median reduction along an axis.
- `project_using_prod(node, axis)` — product reduction along an axis.
- `project_using_any(node, axis)` — logical OR reduction along an axis.
- `project_using_all(node, axis)` — logical AND reduction along an axis.
- `project_using_count_nonzero(node, axis)` — count non-zero elements along an axis.
- `project_using_quantile(node, q, axis)` — quantile/percentile reduction along an axis;
  requires a quantile level `q` in the range [0, 1].

**Model layer — convenience methods**

All new axis reduction operators have corresponding convenience methods on `GenericIndex`:
- `GenericIndex.min(axis=-1)`, `GenericIndex.max(axis=-1)`, `GenericIndex.std(axis=-1)`,
  `GenericIndex.var(axis=-1)`, `GenericIndex.median(axis=-1)`, `GenericIndex.prod(axis=-1)`,
  `GenericIndex.any(axis=-1)`, `GenericIndex.all(axis=-1)`, `GenericIndex.count_nonzero(axis=-1)`,
  and `GenericIndex.quantile(q, axis=-1)`.

### Deprecated

**Python versions**

- Python 3.11 is deprecated and will be removed in a future version. Please upgrade to Python 3.12 or later.

### Removed

**Distribution indexes - Breaking changes**

- `UniformDistIndex` — use `DistributionIndex("x", scipy.stats.uniform, {"loc": 0, "scale": 1})`.
- `LognormDistIndex` — use `DistributionIndex("x", scipy.stats.lognorm, {"loc": 0, "scale": 1, "s": 0.5})`.
- `TriangDistIndex` — use `DistributionIndex("x", scipy.stats.triang, {"loc": 0, "scale": 1, "c": 0.5})`.

**Engine layer — breaking changes**

- `executor.evaluate` — use `executor.evaluate_single_node` or `executor.evaluate_nodes` as appropriate.

## [0.6.0] - 2026-03-01

### Added

**Model / simulation layer**

- `Model(name, indexes)` — replaces `AbstractModel` / `InstantiatedModel`.
  `Model.abstract_indexes()` returns indexes that require external values;
  `Model.is_instantiated()` returns `True` when all indexes are concrete.
- `Evaluation(model)` — generic evaluation bridge.
  `.evaluate(scenarios, nodes_of_interest=None, *, axes=None, functions=None)`
  consumes an `Ensemble`, builds the engine substitution dict, and returns an
  `EvaluationResult`.
- `EvaluationResult` — typed wrapper for evaluation output.
  `result[idx]` returns the raw array; `result.marginalize(idx)` computes the
  weighted expectation over the scenario dimension.  Properties: `weights`,
  `axes`, `full_shape`.
- `Evaluation.evaluate(axes=…)` grid mode — sweeps axis indexes over dense
  arrays while the ensemble provides non-axis index values.  Result arrays
  have shape `(N₀, …, Nₖ, S)`.
- `WeightedScenario = tuple[float, dict[GenericIndex, Any]]` — canonical type
  alias for a weighted model instantiation.
- `Ensemble` (Protocol) — structural protocol satisfied by any iterable of
  `WeightedScenario` tuples.
- `DistributionEnsemble(model, size, rng=None)` — samples each
  distribution-backed abstract index independently and yields `size`
  equally-weighted scenarios.  Raises `ValueError` if any abstract index is
  not distribution-backed.

**Engine layer**

- `TimeseriesIndex(name, values)` — time-indexed quantity supporting fixed
  array, placeholder, and formula modes.
- `graph.timeseries_constant(values, name)` and
  `graph.timeseries_placeholder(name)` graph nodes.
- `graph.piecewise(*clauses)` — conditional expression node; each clause is
  `(expr, cond)`.  Replaces `sympyke.Piecewise`.
- `graph.negate(node)` — unary negation node; supports `GenericIndex.__neg__`.
- `graph.expand_dims(node, axis)` and `graph.squeeze(node, axis)` — axis
  management nodes.
- `graph.project_using_sum(node, axis=-1)` and
  `graph.project_using_mean(node, axis=-1)` — axis reduction nodes that
  always preserve the reduced axis as size 1.

**Examples and documentation**

- `examples/mobility_bologna/` — Bologna mobility example using the direct
  pattern (`DistributionEnsemble` + `Evaluation`).
- `examples/overtourism_molveno/` — Molveno overtourism example using the
  vertical extension pattern (`OvertourismModel`, `OvertourismEnsemble`, grid
  evaluation, sustainability field).
- `docs/getting-started.md` — step-by-step tutorial covering both usage patterns.
- `docs/design/dd-cdt-engine.md` — design document for the engine layer.
- `docs/design/dd-cdt-model.md` — design document for the model/simulation layer.

### Changed

- `graph.project_using_sum` / `graph.project_using_mean` now **always**
  preserve the reduced axis as a size-1 dimension (keepdims semantics).
  Previously the axis was collapsed (e.g. shape `(3, 3)` reduced to `(3,)`
  along axis 0); now the result shape is `(1, 3)`.
- `Index(name, value)` — `cvs` keyword argument removed (was unused metadata).
- `TimeseriesIndex(name, values)` — `cvs` keyword argument removed.
- `examples/` restructured into two sub-packages:
  `examples/mobility_bologna/` and `examples/overtourism_molveno/`.
- Design document renamed from `dd-000-engine.md` to `dd-cdt-engine.md`.

### Removed

**Model / simulation layer — breaking changes**

- `AbstractModel` — use `Model`.
- `InstantiatedModel` — use `WeightedScenario`-based `Evaluation.evaluate()`.
- `Evaluation` (old signature `Evaluation(inst, ensemble)` with
  `evaluate_grid()` / `evaluate_usage()`) — use `Evaluation(model).evaluate(scenarios, axes=…)`.
- `SustainabilityEvaluation` — sustainability field is now computed as
  explicit post-processing in `examples/overtourism_molveno/overtourism_molveno.py`.
- `dt_model.ensemble` shim package.
- `SymIndex` — use formula-mode `Index(name, node)` instead.
- `ContextVariable`, `CategoricalContextVariable`,
  `UniformCategoricalContextVariable`, `ContinuousContextVariable`,
  `PresenceVariable`, `Constraint` — moved from the core library to
  `examples/overtourism_molveno/overtourism_metamodel.py`.
- `sympyke` module (`dt_model.internal.sympyke`) — removed entirely.
  Use `graph.piecewise()` and `GenericIndex.__eq__` directly.
- `dt_model.symbols` subpackage — index types now live in `dt_model.model.index`.
- `dt_model.internal` subpackage — removed.

**Engine layer — breaking changes**

- `graph.reduce_sum` / `graph.reduce_mean` — use
  `graph.project_using_sum` / `graph.project_using_mean`.
- `graph.function` alias — use `graph.function_call`.
- `graph.timeseries_constant.times` attribute.
- `executor.evaluate_dag`, `executor.evaluate_trees`,
  `executor.evaluate_single_tree` — use `executor.evaluate_nodes`.
- `engine/frontend/forest.py` and `engine/frontend/ir.py` — moved to the
  experimental JIT branch; no longer part of the public surface.
- `numpybackend/jit.py` — renamed to `numpybackend/numpy_ast.py`
  (internal; experimental, unmerged).

## [0.5.0] - 2025-07-14

[Unreleased]: https://github.com/fbk-most/civic-digital-twins/compare/v0.7.0...HEAD
[0.7.0]: https://github.com/fbk-most/civic-digital-twins/compare/v0.6.0...v0.7.0
[0.6.0]: https://github.com/fbk-most/civic-digital-twins/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/fbk-most/civic-digital-twins/releases/tag/v0.5.0
