<!-- SPDX-License-Identifier: Apache-2.0 -->

# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `EvaluationResult.expected_value(idx)` — canonical weighted expectation over
  the ensemble and parameter dimensions, replacing `marginalize()`.  Uses
  `GenericIndex.output_axes` to determine which dimensions are DOMAIN (kept) vs
  ENSEMBLE/PARAMETER (contracted); correctly handles T=1 timeseries that were
  previously indistinguishable from scalars.
- `GenericIndex.output_axes` — ordered tuple of `Axis` objects describing the
  dimensions of the index's result array (`*PARAMETER, *ENSEMBLE, *DOMAIN`
  convention).
- `dt_model/axes.py` — shared module owning `Axis`, `AxisRole`, and the built-in
  role constants `DOMAIN`, `PARAMETER`, `ENSEMBLE`.  Breaking the former
  engine→model import dependency: the engine layer now imports axis types from
  this module rather than from `model.axis`. 
- `EvaluationPlan`, `Region`, `RegionGuard` in `simulation/plan.py` — frozen
  dataclasses encoding the evaluation structure as a topologically-ordered DAG
  of computation regions.  `build_plan(strategy="monolithic" | "regional")`
  on `Evaluation` builds a plan once; `execute_plan(plan, ensemble, *,
  parameters, ...)` reuses it across multiple calls.  The `"regional"` strategy
  partitions the graph at `variant_selector` boundaries: a shared pre-selector
  region, per-branch guarded regions (evaluated only for matching scenarios via
  `np.take` slice + `np.put_along_axis` scatter-back), and a merge region
  (closing #136).
- `Evaluation.evaluate_incremental(initial_ensemble_size, ...)` →
  `EvaluationHandle` — builds a plan, runs the first batch, and returns a
  handle for checkpoint-style ensemble extension without discarding prior
  results (closing #168).
- `EvaluationHandle` in `simulation/handle.py` — incremental handle;
  `extend(n)` draws *n* additional scenarios from `DistributionEnsemble`
  using a shared `rng`, executes the plan, and merges via `_merge_results`.
  Two calls with the same seed are guaranteed to produce the same sequence as
  one call of the combined size.
- `AsyncEvaluationHandle(EvaluationHandle)` in `simulation/handle.py` —
  non-blocking variant backed by `concurrent.futures.Future`; exposes
  `poll() → (bool, EvaluationResult | None)` and `get() → EvaluationResult`;
  `extend()` delegates to the base class once the future resolves
  (closing #169). `Evaluation.submit_evaluate(initial_ensemble_size, ..., pool=)`
  — submits the initial `execute_plan` call to a `concurrent.futures.Executor`
  (defaults to a lazily-created module-level `ThreadPoolExecutor`) and returns
  an `AsyncEvaluationHandle` immediately.
- `CategoricalIndex` selector can be passed as a `parameters=` axis to
  `Evaluation.evaluate()` to sweep over variant outcomes deterministically
  (no ensemble required), including in combination with numeric PARAMETER axes
  for multi-dimensional grids.  Tests and documentation added; no implementation
  change was needed (closing #134).
- `HasNode[T]` protocol in `engine.frontend.graph` — structural protocol satisfied
  by any object with a `.node: Node[T]` property (e.g. `GenericIndex`).  Re-exported
  from `civic_digital_twins.dt_model`.  `function_call`, `piecewise`, and `where`
  now accept `HasNode` in all node positions, eliminating the need for explicit
  `.node` at call sites (closing #161).
- `NumpyBackend` class in `engine.numpybackend.executor` — user-facing binding
  point for the numpy backend.  `NumpyBackend.adapt(fn)` wraps a callable as a
  `Functor` bound to the numpy convention; `Evaluation.evaluate()` accepts
  `backend=NumpyBackend` (default, currently the only supported backend).
  `Functor`, `NumpyBackend`, and `LambdaAdapter` are re-exported from
  `civic_digital_twins.dt_model` (closing #162).
- `ConstTimeseriesIndex(name, array)` — companion to `ConstIndex`: a structural
  constant backed by a `timeseries_constant` graph node whose values are fixed at
  model-construction time and cannot be overridden in a `Scenario`.  Exported from
  `civic_digital_twins.dt_model`.
- `Scenario(model, overrides={idx: value})` in `simulation/scenario.py` — the
  canonical what-if wrapper around a `Model` or `ModelVariant`.  Accepted override
  types per index kind: `float` for `Index`; `np.ndarray` for `TimeseriesIndex`;
  `Distribution` for `DistributionIndex`; `str` (concrete pin) or `dict[str, float]`
  (new probability weights, subset of support) for `CategoricalIndex`; `str` only
  for `ConditionalCategoricalIndex`.  `ConstIndex`, `ConstTimeseriesIndex`, and
  `ConditionalDistributionIndex` do not accept overrides.
  `Scenario.abstract_indexes()` returns indexes that still require ensemble sampling.
  `Scenario.base_substitutions()` returns concrete values ready for engine injection.
  `Scenario.effective_distribution(idx)` returns the active distribution (override if
  present, else `idx.value`).
  `Scenario.effective_outcomes(idx)` returns the active outcome-probability map for
  categorical indexes (`{pin: 1.0}` for a concrete pin, override dict or `idx.outcomes`
  for `CategoricalIndex`, `None` for an unresolved `ConditionalCategoricalIndex`).

### Changed

- **Breaking: `project_using_*` nodes now accept `axis: Axis`** instead of a
  raw integer.  Pass `Axis("time", DOMAIN)` for the time axis.  The numpybackend
  raises `UnsupportedOperation` at evaluation time if any other axis is supplied.
- **Breaking: `TimeseriesIndex` is no longer a subclass of `Index`.**  Both are now
  direct subclasses of `GenericIndex`.  Code that relied on
  `isinstance(idx, Index)` matching timeseries indexes must be updated to also
  check `isinstance(idx, TimeseriesIndex)`.
- **Breaking: `Index(scalar)` and `TimeseriesIndex(array)` now produce placeholder
  graph nodes** instead of constant nodes.  The default value is stored on the index
  object (`Index.value`, `TimeseriesIndex.values`) and injected at evaluation time
  by the base `Scenario`.  Use `ConstIndex(v)` / `ConstTimeseriesIndex(array)` to
  retain constant-node semantics for values that must never be overridden.
- **Breaking: Assigning a `Distribution` directly to `Index` now raises `TypeError`.** 
  Use `DistributionIndex` instead.

### Deprecated

- `EvaluationResult.marginalize()` — use `EvaluationResult.expected_value()`
  instead.  `marginalize()` now emits `DeprecationWarning` and will be removed
  in a future release.
- `LambdaAdapter` — use `NumpyBackend.adapt()` instead.  `LambdaAdapter` now
  emits a `DeprecationWarning` on construction and will be removed in a future
  release (closing #162).
- `Evaluation(model)` and all `Ensemble(model, ...)` constructors — pass a
  `Scenario(model)` instead.  These constructors now auto-wrap the model in a
  base `Scenario` and emit `DeprecationWarning`.  The canonical chain is
  `Model → Scenario → {DistributionEnsemble | CrossProductEnsemble | PartitionedEnsemble} + Evaluation`.
- Mutable index setters `ConstIndex.v`, `ConstTimeseriesIndex.values`,
  `TimeseriesIndex.values`, and `DistributionIndex.params` — vary index values
  via `Scenario(model, overrides={idx: new_value})` instead.  All setters emit
  `DeprecationWarning` and will be removed in a future release.

### Removed

- `graph.expand_dims` and `graph.squeeze` — axis management nodes removed; no
  replacement (these were never used in production code).
- `graph.AxisOp` — abstract base class for the removed axis management nodes.
- `graph.NpAxis` — numpy axis type alias; was an implementation detail of the
  former integer-axis interface.

## [0.9.0] - 2026-05-02

### Added

- `Axis(name, role)` and `AxisRole` (`PARAMETER`, `ENSEMBLE`, `DOMAIN`) — explicit
  named dimensions for result arrays; exported from `civic_digital_twins.dt_model`.
- `AxisEnsemble` protocol — batched ensemble interface exposing `ensemble_axes`,
  `ensemble_weights`, and `assignments()`.  `DistributionEnsemble` now implements
  it natively; the legacy `Iterable[WeightedScenario]` path is still accepted via a
  deprecation adapter.
- `PartitionedEnsemble(model, axes, default_axis, rng)` — N-on-M independent ENSEMBLE
  axes; each `EnsembleAxisSpec` covers a disjoint subset of abstract indexes with its
  own sample budget.  Validates unique axis names and full index coverage.
- `Evaluation.evaluate()` — new `ensemble=` and `parameters=` keyword arguments
  (canonical names replacing the deprecated `scenarios=` / `axes=`).  Accepts any
  `AxisEnsemble`; a single batched evaluation pass replaces the old per-scenario loop.
- `EvaluationResult.parameter_values` — replaces the deprecated `result.axes`.
- Every result array is guaranteed to carry explicit ENSEMBLE singleton dims for
  nodes not downstream of ENSEMBLE substitutions, eliminating the `S == T` shape
  ambiguity (#142).
- `ConditionalCategoricalIndex(name, parents, probs_fn)` — categorical index
  whose per-outcome probabilities depend on the resolved values of parent
  indexes; exported from `civic_digital_twins.dt_model`.
- `ConditionalDistributionIndex(name, parents, dist_fn)` — distribution-backed
  index whose scipy distribution factory depends on resolved parent values;
  replaces `PresenceVariable` in the overtourism example and is available to
  any domain model.  Exported from `civic_digital_twins.dt_model`.
- `CrossProductEnsemble(model, restrictions, max_categorical_size, exclude, rng)`
  — ensemble that enumerates (or samples) the full cross-product of
  `CategoricalIndex` / `ConditionalCategoricalIndex` values discovered via
  model dependency analysis; replaces `OvertourismEnsemble`.  Indexes listed
  in `exclude` are treated as PARAMETER axes rather than ENSEMBLE axes.
- `sample_across(index, ensemble, n, rng)` — draw `n` samples from a
  `ConditionalDistributionIndex` across all scenarios of an `AxisEnsemble`,
  returning a weighted array aligned with the ensemble axis.
- `scipy-stubs` added to dev dependencies for improved type checking.
- Pyright now checks `examples/` directory (previously only `civic_digital_twins` and `tests`).
- `tests/test_doc_sync.py` — automated snippet-alignment test that compares
  every Python code block in the design docs and guides against its paired
  runnable example script in `examples/doc/`.  Run without arguments for a
  compact per-pair summary (`= OK` / `~ OK` / `~ Warn` / `✗ Fail`); pass a
  doc-name fragment for a verbose block-by-block report.  Stub and
  reference-only blocks are detected and skipped automatically.
  `= OK` now requires 100% score and all per-line ratios ≥ 0.99; any `~ OK`
  or `~ Warn` block fails the test unless listed in `_EXPECTED_NEAR_VERBATIM`.
- `examples/doc/doc_readme.py` — new script covering the two README code
  snippets (engine layer and model/simulation layer).
- `civic_digital_twins.dt_model.graph` shim — `graph` is now importable
  directly from the top-level `dt_model` package (`from
  civic_digital_twins.dt_model import graph`), closing #123.
- SPDX-License-Identifier headers added to all tracked Python and Markdown
  files; pre-release verification step added to README checklist.

### Changed

- **Python 3.11 dropped** — minimum supported version is now Python 3.12.
  The CI matrix, `pyproject.toml` classifiers, ruff `target-version`, and
  `pyrightconfig.json` are updated accordingly. (#122)
- **Breaking: `ContextVariable` hierarchy removed (closing #139).**
  `ContextVariable`, `CategoricalContextVariable`,
  `UniformCategoricalContextVariable`, and `ContinuousContextVariable` deleted.
  Context variables are now ordinary `CategoricalIndex` instances.
- **Breaking: `PresenceVariable` removed.** Presence variables are now
  `ConditionalDistributionIndex` instances from the core library.
- **Breaking: `OvertourismEnsemble` removed.** Replaced by `CrossProductEnsemble`
  (see Added), which handles enumerate-vs-sample, subset restriction, and
  weight renormalisation for any model using `CategoricalIndex` CVs.
- **Breaking: `overtourism_metamodel.py` removed.** `Constraint` (the only
  remaining class) is now defined directly in `molveno_model.py`.
- **Overtourism metamodel modernization (#152):** `OvertourismModel` removed;
  `MolvenoModel` now subclasses `Model` directly with its own
  `Inputs`/`Outputs` dataclasses; `PresenceModel` dissolved into `MolvenoModel`.
- PEP 695 generic syntax adopted throughout: `~30` generic classes in
  `graph.py` converted to `class Foo[T]`; `TypeAlias` declarations in
  `executor.py` and `IOProxy` in `model.py` converted to `type X = ...`.
  `from __future__ import annotations` removed from five modules; `Callable`
  and `Iterator` migrated from `typing` to `collections.abc` where
  applicable. (#114)
- `numpy` floor raised to `>=2.3.2`; `pandas` moved from runtime to `dev`
  dependencies (used only by example models, not the library itself). Both
  floors now guarantee pre-compiled wheels for Python 3.12, 3.13, and 3.14,
  eliminating source-compilation delays in CI. (#122)
- **Molveno example slim-down:** module-level aliases removed; modal-line
  regression replaced by orthogonal regression (SVD); miscellaneous dead code
  removed.
- Plotting in `overtourism_molveno` example now uses the same pattern as
  `mobility_bologna`: non-interactive backend (`Agg`), functions return
  `fig` objects, main execution wrapped in `if __name__ == "__main__":`.
  Plots are saved to an `output/` directory.

### Fixed

- `EvaluationResult.marginalize()` raised `IndexError` on constant nodes in
  grid+ensemble mode (two or more PARAMETER axes plus at least one ENSEMBLE axis)
  (#155).
- `Evaluation.evaluate()` raised `ValueError` (numpy broadcast failure) in
  grid+ensemble+timeseries mode when the ENSEMBLE size differed from the timeseries
  length (#156).  Also fixed for pure-PARAMETER+timeseries mode (no ensemble).
- `EvaluationResult.marginalize()` contracted the wrong axis when `S == T` (#142);
  PARAMETER dims and non-trivial DOMAIN dims are now preserved.
- The all-axes invariant (`*PARAMETER, *ENSEMBLE, *domain` shape on every result
  array) is now enforced for all axis combinations, including pure-PARAMETER mode.
  Post-normalisation assertions verify the invariant at debug time.
- **Breaking shape change (scalar ENSEMBLE results, no timeseries)** — in
  pure-ENSEMBLE mode with no PARAMETER axes and no timeseries nodes, scalar
  result arrays now have shape `(S,)` instead of `(S, 1)`.  `marginalize()`
  output is identical; only direct `result[idx].shape` comparisons are affected.
- Dependabot vulnerability alerts resolved: `fonttools` bumped to `>=4.60.2`
  (moderate) and `pillow` to `>=12.1.1` (high) via lockfile regeneration. (#132)
- Documents and `examples/doc/` scripts updated so that the scripts no longer emit
  warnings at runtime.

### Deprecated

- `evaluate(scenarios, …)` positional argument — use `ensemble=` instead.
- `evaluate(axes={…})` keyword — use `parameters=` instead.
- `result.axes` property — use `result.parameter_values` instead.
- Passing `Iterable[WeightedScenario]` to `evaluate()` — use an `AxisEnsemble`
  (e.g. `DistributionEnsemble`) instead.

## [0.8.1] - 2026-04-02

### Fixed

- `EvaluationResult.marginalize()` raised a shape mismatch when called on an
  index whose value does not depend on any scenario-varying input (e.g. a
  constant index or a timeseries whose sum collapses to shape `(1,)` regardless
  of the number of scenarios).  The fix detects arrays with no scenario
  dimension and broadcasts one in before contracting with the weights.
  A shape heuristic is used (`arr.shape[0] != S`); the known fragility when
  `S == T` is documented and tracked in #142.

### Added

**`CategoricalIndex` — probabilistic runtime model selection**

- `CategoricalIndex(name, outcomes)` — a new `Index` subclass backed by a
  finite string-keyed probability distribution.  Always abstract (placeholder
  mode).
  Raises `ValueError` at construction if `outcomes` is empty, any probability
  is non-positive, or the probabilities do not sum to 1.0.
  - `support` — ordered list of outcome keys.
  - `outcomes` — dict copy of the probability mapping.
  - `sample(rng=None)` — draw one key proportional to declared probabilities.
  - Exported from `civic_digital_twins.dt_model`.

**`ModelVariant` — runtime mode**

- `ModelVariant` now supports two additional selector types:
  - **`CategoricalIndex` selector** — all variants are preserved in the graph.
    A `variant_selector` node and one `exclusive_multi_clause_where` node per
    output field are built at construction time.  The ensemble samples the
    `CategoricalIndex` per scenario; the executor selects the correct branch
    value via `numpy.select`.
  - **`graph.Node` selector** — arbitrary boolean guard expression built with
    `ModelVariant.guards_to_selector([(key, condition), ...])`, which wraps
    `graph.piecewise`.  Guards are evaluated left-to-right; place the most
    specific guard first.
- `mv.outputs.<field>` in runtime mode returns an `Index` backed by an
  `exclusive_multi_clause_where` node.
- `mv.inputs` in runtime mode returns fields whose names appear in
  **any** variants' inputs (union by field name).
- `mv.expose` in runtime mode returns only fields whose names appear in
  **all** variants' expose proxies (intersection by field name).
- `mv._selector_index` — thin `Index` wrapping `_selector_node`; use
  `result[mv._selector_index]` from an `EvaluationResult` to retrieve a
  `(S, 1)` string array of the active variant key per scenario.

**Engine layer — `MultiClauseOp`, `variant_selector`, `exclusive_multi_clause_where`**

- `MultiClauseOp(Generic[C, T], Node[T])` — new abstract base class for
  multi-clause conditional nodes, following the `BinaryOp` / `UnaryOp` pattern.
  `multi_clause_where` is now a thin subclass.

  > **Note**: the `multi_clause_where` class hierarchy changed.  Code that
  > checks `isinstance(node, graph.multi_clause_where)` still works; code that
  > relied on `multi_clause_where` being a direct `Node` subclass may need to
  > be updated.

- `variant_selector(selector_node, branch_map, merge_nodes)` — first-class
  graph node carrying structural metadata for the runtime variant dispatch.
  Listed as a dependency of `exclusive_multi_clause_where` so it is reached by
  `linearize.forest` via normal graph traversal.  Evaluated as a no-op (empty
  sentinel array) by the executor.

- `exclusive_multi_clause_where(MultiClauseOp)` — peer of `multi_clause_where`
  under `MultiClauseOp`.  Has an additional `companion: variant_selector` field
  listed as a graph dependency.  Semantics: branches are mutually exclusive by
  construction (one per variant key); in v0.8.x evaluation is still eager
  (same as `multi_clause_where`).

**`AbstractIndexNotInInputsWarning`**

- `AbstractIndexNotInInputsWarning(ModelContractWarning)` — new soft warning
  emitted at `Model` construction (dataclass-based path only) when an abstract
  index returned by `abstract_indexes()` is not declared in `Inputs`.  Abstract
  indexes receive their values from outside the model and are semantically inputs.
  Exported from `civic_digital_twins.dt_model`.

**`DistributionEnsemble` — `CategoricalIndex` support**

- `DistributionEnsemble` now accepts models whose abstract indexes are a mix of
  `Distribution`-backed `Index` values and `CategoricalIndex` objects.
  `CategoricalIndex` entries are sampled via `CategoricalIndex.sample(rng)` and
  stored as `(S, 1)` object arrays of string keys, matching the stacking
  convention used for scalar index samples.

### Changed

**`multi_clause_where`**

- `multi_clause_where` is now a subclass of `MultiClauseOp` rather than a
  direct subclass of `Node`.  `isinstance(node, graph.multi_clause_where)` is
  unaffected; `isinstance(node, graph.MultiClauseOp)` is now the preferred
  check for code that handles both conditional node types.

**`ModelVariant` input contract**

- `ModelVariant` no longer requires variants to share identical `inputs` field
  names.  Only `outputs` field names must be identical (required to build the
  merge graph).  `inputs` may differ across variants; `mv.inputs` exposes their
  union.

**`BolognaModel` — structured inputs and direct `EvaluationResult`**

- `BolognaModel` now declares all policy (`i_p_*`) and behavioural (`i_b_*`)
  parameters in an `Inputs` dataclass, following the same constructor-argument
  pattern as its sub-models.
- `evaluate()` returns `EvaluationResult` directly instead of a normalised
  `subs` dict; `compute_kpis()` uses `result.marginalize()`.
- `__main__` demonstrates a second (stricter pricing) scenario; plots are saved
  to `examples/mobility_bologna/output/` (directory is `.gitignore`d).

## [0.8.0] - 2026-03-21

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
     optional `Expose` inner dataclass.  `Expose` is for diagnostics
     only and must not be used to wire indexes into sibling or parent models.
  3. Local variables inside `__init__` — internal to the engine graph only;
     not accessible from outside.
- **List and dict field values**: dataclass fields may hold a single
  `GenericIndex`, a `list[GenericIndex]`, or a `dict[str, GenericIndex]`.
  Iteration, `len()`, and `in` flatten these to scalar indexes only.
  Field access returns the raw value.
- **`indexes` derived automatically**: the flat `model.indexes` list is built by
  collecting and deduplicating all scalar `GenericIndex` values from `inputs`,
  `outputs`, and `expose` (first-seen order).
- **Construction-time validation**: every entry in `inputs` and `outputs` must
  appear in `indexes`; a descriptive `ValueError` is raised on violation.
- **`IOProxy`** — read-only proxy exposing declared fields via attribute access,
  iteration, `len()`, and `in` membership.  `IOProxy` is generic (`IOProxy[DC]`);
  `__getattr__` returns `Any` so typed field access flows through without `cast()`.
- **Inputs contract convention and warnings**: every `GenericIndex` received as
  a constructor parameter must be declared in `Inputs`.  Two new warning classes,
  both exported from `civic_digital_twins.dt_model`, enforce this softly at
  construction time:
  - `ModelContractWarning` — `UserWarning` base for all Model I/O contract
    warnings.  Use `warnings.filterwarnings("error", category=ModelContractWarning)`
    to turn the whole family into errors; each subclass is independently filterable.
  - `InputsContractWarning(ModelContractWarning)` — emitted when a constructor
    parameter holds a `GenericIndex` (scalar, list, or dict) that is absent from
    the declared `Inputs` dataclass.  Names the offending parameter precisely.
- **Legacy `indexes=` path deprecated**: passing `indexes` explicitly emits a
  `DeprecationWarning`.  The legacy path will be removed in a future version.

**`ModelVariant` — switching between Model implementations**

- `ModelVariant(name, variants, selector)` — selects among pre-constructed
  `Model` instances that share the same I/O contract.  The active variant is
  resolved once at construction time via a plain string key, and `ModelVariant`
  then acts as a fully transparent proxy for it, usable anywhere a `Model` is
  expected.
- `inputs`, `outputs`, `expose`, `indexes`, `abstract_indexes()`,
  `is_instantiated()`, and arbitrary attribute access all delegate to the active
  variant.  Internal indexes of inactive variants are not visible through
  `model_variant.indexes`; they remain accessible via
  `model_variant.variants["key"].*`.
- **Construction-time I/O contract validation**: `inputs` and `outputs` field
  names must be identical across all declared variants; a descriptive `ValueError`
  is raised if they differ.
- Static mode only in this release; runtime selection via `CategoricalIndex` or
  `graph.Node` selector was added in the subsequent unreleased version.
- `ModelVariant` exported from `civic_digital_twins.dt_model`.

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

**Molveno overtourism example — modular rewrite**

- `molveno_model.py` decomposed into five concern sub-models with explicit typed
  interfaces: `PresenceModel` (CVs and PVs as `Outputs`), `ParkingModel`,
  `BeachModel`, `AccommodationModel`, `FoodModel`.
- Every `i_*` parameter — including uncertain `DistributionIndex` values — is an
  `Input` to the sub-model that uses it; default values are created by
  `MolvenoModel` and passed via constructors.
- `MolvenoModel` wires the five sub-models and subclasses `OvertourismModel` so
  that `OvertourismEnsemble` and `evaluate_scenario` work without modification.
- All original module-level names (`M_Base`, `CV_*`, `PV_*`, `I_P_*`) preserved
  as aliases — `overtourism_molveno.py` requires no changes.

**Model modularity documentation**

- New concept guide `docs/design/dd-cdt-modularity.md` — three-level access
  model, constructor wiring, `Inputs` contract convention, `ModelVariant`,
  decomposition axes, annotated Bologna worked example, API reference, and
  design rationale.
- `docs/design/dd-cdt-model.md` updated with dataclass-based `Model` API,
  `ModelVariant`, and `InputsContractWarning` sections.
- `docs/getting-started.md` and `README.md` updated.
- New `examples/doc/doc_modularity.py` — runnable validation for the
  modularity guide.

**Python versions**

- Python 3.14 added to the CI test matrix and PyPI classifiers.

### Deprecated

- **`indexes=` argument to `Model.__init__`**: passing a flat index list
  explicitly emits a `DeprecationWarning`.  Use the dataclass-based
  `inputs=` / `outputs=` / `expose=` API instead.  The legacy path will
  be removed in a future version.

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

[Unreleased]: https://github.com/fbk-most/civic-digital-twins/compare/v0.9.0...HEAD
[0.9.0]: https://github.com/fbk-most/civic-digital-twins/compare/v0.8.1...v0.9.0
[0.8.1]: https://github.com/fbk-most/civic-digital-twins/compare/v0.8.0...v0.8.1
[0.8.0]: https://github.com/fbk-most/civic-digital-twins/compare/v0.7.0...v0.8.0
[0.7.0]: https://github.com/fbk-most/civic-digital-twins/compare/v0.6.0...v0.7.0
[0.6.0]: https://github.com/fbk-most/civic-digital-twins/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/fbk-most/civic-digital-twins/releases/tag/v0.5.0
