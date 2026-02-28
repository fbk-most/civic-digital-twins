# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.6.0] - unreleased

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

## [0.5.1] - 2026-02-23

Internal release. See git log for details.

## [0.5.0] - 2025-07-14

Initial public release.
