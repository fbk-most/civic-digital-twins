# Inspection Decisions & Action Plan

> Started: 2026-02-24
> Branch: `feat/engine-redesign`

This file records decisions made during code inspection sessions and the
resulting action items.

## Intended implementation order

1. **Section 1** — Engine cleanup (Steps 1–5), self-contained.
2. **Sections 2+3** — Model/simulation/symbols redesign (Steps 1–6); depends on Section 1 being complete.
3. **Section 4** — Examples + docs: `dd-000-engine.md` revision after Section 1 Step 5; examples restructuring is independent; example migration after Sections 2+3.
4. **Cross-cutting** — `sympyke` removal and API surface after Sections 2+3; documentation after all of the above.

Note: the `piecewise()` addition to `graph.py` is part of Section 1 Step 2. `sympyke.Piecewise` is kept as an alias pointing to `graph.piecewise` until Sections 2+3 are complete (backwards compatibility and easier incremental testing). `Eq` removal and full `sympyke` deletion happen in Sections 2+3 and Cross-cutting respectively.

---

## Section 1 — Engine layer

### Findings reviewed
_Complete — 2026-02-24._

### Decisions

1. Remove `times` from `timeseries_constant` — it is never read by the executor, linearizer, or JIT. `TimeseriesIndex` keeps its `_times` / `times` attribute as user-facing metadata. The `times` setter on `TimeseriesIndex` will update `self._times` only, without rebuilding the node.
2. `forest.py` / `ir.py`: move to the experimental JIT branch along with the rest of the JIT-related code. No impact on the main evaluation path (`evaluate_nodes` only depends on `graph`).

### Action items

Items are ordered by dependency — earlier items should be implemented first.

**Step 1 — Documentation only (no behavioural changes)**
- [ ] `engine/atomic/__init__.py`: document that the ID counter is global to the Python process — all graph constructions in a session share it, so node IDs are unique across all graphs, not just within one.
- [ ] `engine/compileflags/__init__.py`: (1) extend the module docstring with an end-user usage example (`export DTMODEL_ENGINE_FLAGS=trace`); (2) extend the `defaults` docstring to note that it is read once at import time — the env var must be set before importing any engine module.
- [ ] `executor.State`: add a comment clarifying that `frozen=True` prevents attribute reassignment (identity stability) but does not prevent mutation of the `values` dict — this is intentional.

**Step 2 — `graph.py` structural changes**
- [x] `timeseries_constant`: store `values` as `ArrayLike` (no `np.asarray` in `__init__`); move conversion to `_eval_timeseries_constant` in the executor — consistent with how `constant` works; remove `numpy` import from `graph.py`.
- [ ] Remove `times` parameter and attribute from `graph.timeseries_constant`; update `TimeseriesIndex.__init__` (stop passing `self._times`); update `TimeseriesIndex.times` setter (update `self._times` only, no node rebuild); delete tests asserting `node.times` on `timeseries_constant` instances; keep tests for `TimeseriesIndex.times`.
- [ ] Add `negate` node (UnaryOp) to `graph.py`; add `__neg__` to `GenericIndex` in `index.py`; update module docstring (arithmetic operations list); add tests.
- [ ] Move `Piecewise` logic from `sympyke` to `graph.py` as `piecewise(*clauses)` (lowercase); update module docstring; keep `sympyke.Piecewise = graph.piecewise` as a backward-compatibility alias until Sections 2+3 are complete; add tests.

**Step 3 — `keepdims` removal**
- [x] Remove `keepdims` parameter from `project_using_sum` / `project_using_mean`: axis preservation is always implied by their semantics. Update executor: drop the `isinstance` special-case in `_eval_axis_op` and update the `_AXIS_OP_DISPATCH` table keys from `graph.reduce_sum`/`graph.reduce_mean` to `graph.project_using_sum`/`graph.project_using_mean`. Update `GenericIndex.sum()` / `.mean()` to no longer pass `keepdims=True`. Update tests accordingly.

  **Breaking change vs 0.5.0:** In 0.5.0 `reduce_sum`/`reduce_mean` (and `project_using_sum`/`project_using_mean`) defaulted to `keepdims=False`. Callers that relied on the default now get a result with a trailing size-1 axis instead of a collapsed axis (e.g. shape `(1, 3)` instead of `(3,)` for axis=0 on a `(3, 3)` input).

**Step 4 — Alias removal**
- [ ] Remove `reduce_sum` / `reduce_mean` aliases from `graph.py`; update all callers (includes `numpy_ast.py`'s `_operation_names` dict).
- [ ] Remove `function` alias from `graph.py`; update all callers.

**Step 5 — JIT / experimental branch reorganisation**
- [ ] Rename `numpybackend/jit.py` → `numpybackend/numpy_ast.py`; update docstring (remove Numba/JIT references, describe as graph-to-numpy AST translation for debugging); update `executor.py` import and three call sites; rename `tests/.../test_jit.py` → `test_numpy_ast.py`; update `numpybackend/__init__.py` docstring.
- [ ] Add timeseries support to `numpy_ast.py`: add `timeseries_constant` and `timeseries_placeholder` to `_operation_names`; handle them in `_simple_graph_node_to_ast_expr`. Also add `negate` (added in Section 1 Step 2).
- [ ] Move `engine/frontend/forest.py` and `engine/frontend/ir.py` to the experimental JIT branch; remove `evaluate_dag` / `evaluate_trees` / `evaluate_single_tree` from `executor.py`; remove `forest, ir` from `executor.py` import; move `test_forest.py` / `test_ir.py` to experimental; update `frontend/__init__.py` docstring (remove `forest`, keep `graph` and `linearize`).

---

## Sections 2 + 3 — Model / Simulation / Symbols redesign

> Sections 2 (Simulation), 3 (Symbols + model), and 5 (Examples) from the inspection plan
> are addressed together here, as the inspection revealed they require a unified architectural
> redesign rather than isolated fixes.

### Findings reviewed
_Complete — 2026-02-25._

### Architecture decision

The model/simulation/symbols layers are to be redesigned around the following semantic model:

**Core invariant — abstract vs instantiated:**
- A `Model` is a collection of `GenericIndex` objects.
- A model is **abstract** if it contains unbound placeholder indexes or distribution indexes (cannot be evaluated yet).
- A model is **instantiated** when all indexes have concrete values bound (can be evaluated).
- `Model.abstract_indexes()` returns the subset needing external values.
- `Model.is_instantiated()` returns True when fully concrete.

**Ensemble — the abstract→instantiated bridge:**
- An ensemble produces a sequence of weighted scenarios: `list[tuple[float, dict[GenericIndex, Any]]]`.
- Each scenario maps every abstract index to a concrete value.
- This replaces the current `Ensemble` class as a core concept; the overtourism vertical provides a concrete generator.

**Generic `Evaluation`:**
- Takes a `Model` + ensemble scenarios, builds the engine substitution dict, runs `executor.evaluate_nodes`, returns `executor.State`.
- Knows nothing about grids, presence variables, sustainability, or constraints.
- Verifies at runtime that all abstract indexes are resolved before calling the engine.

**Overtourism vertical (extension layer):**
- `ContextVariable(Index)`: placeholder index with sampling methods.
- `PresenceVariable(Index)`: placeholder index with distribution callable.
- `Constraint`: named pair of (usage: `GenericIndex`, capacity: `Index`).
- `OvertourismModel(Model)`: CVs/PVs/constraints as labeled subsets of the model's indexes.
- `Ensemble` generator: samples CVs, derives PV distributions, yields `WeightedScenario`.
- `SustainabilityEvaluation(Evaluation)`: grid evaluation, sustainability field, modal lines, CI.

**Bologna usage pattern:**
- No subclass needed — `mobility_bologna.py` uses `Model` directly with a flat collection of `Index` + `TimeseriesIndex`.
- Uses generic `Evaluation` directly; provides its own post-processing.

### Decisions

1. Replace `AbstractModel` / `InstantiatedModel` with a single `Model` class (name + `list[GenericIndex]`). The abstract/instantiated distinction is expressed through `Model.abstract_indexes()` and `Model.is_instantiated()`, not through separate classes.
2. `ContextVariable`, `PresenceVariable`, `Constraint` are domain-specific — move to the overtourism vertical. They are not part of the core library.
3. Generic `Evaluation` stays in the core library; `SustainabilityEvaluation` (grid, field, modal lines) moves to the overtourism vertical.
4. `Ensemble` as a generator class moves to the overtourism vertical. The core defines only the `WeightedScenario` type alias.
5. The `cvs` attribute on `Index` (runtime-dead metadata, see Section 1 notes) is to be removed.
6. `SymIndex` and formula-mode `TimeseriesIndex` are doing the same thing for different data shapes — unify or document the distinction clearly.

### Action items

**Step 1 — Core `Model` class**
- [ ] Create `dt_model/model/model.py` with `Model(name, indexes: list[GenericIndex])`; add `abstract_indexes()` and `is_instantiated()` methods.
- [ ] Remove `AbstractModel` and `InstantiatedModel`; update all callers.

**Step 2 — Generic `Evaluation`**
- [ ] Refactor `simulation/evaluation.py`: extract the engine-bridge logic (build subs dict → run engine → collect state) into a clean generic `Evaluation.evaluate(scenarios, nodes_of_interest)`.
- [ ] Define `WeightedScenario = tuple[float, dict[GenericIndex, Any]]` type alias in core.
- [ ] Add runtime check: raise if any abstract index is unresolved when `evaluate()` is called.

**Step 3 — Overtourism vertical**
- [ ] Move `symbols/context_variable.py`, `symbols/presence_variable.py`, `symbols/constraint.py` to `dt_model/verticals/overtourism/` (new library module, not inside `examples/`).
- [ ] Create `OvertourismModel(Model)` with CVs/PVs/constraints as labeled subsets.
- [ ] Move `simulation/ensemble.py` `Ensemble` class to overtourism vertical.
- [ ] Create `SustainabilityEvaluation(Evaluation)` in overtourism vertical with grid evaluation, field computation, modal lines, CI; remove 2-PV hardcoding (`assert len(pvs) == 2`).

**Step 4 — Bologna usage pattern**
- [ ] Refactor `examples/mobility_bologna/mobility_bologna.py` to use generic `Model` + `Evaluation` directly (no subclass needed).

**Step 5 — Index cleanup**
- [ ] Remove `cvs` attribute from `Index` (runtime-dead metadata).
- [ ] Clarify or unify `SymIndex` vs formula-mode `TimeseriesIndex`.

**Step 6 — Tests**
- [ ] Add test coverage for generic `Model` and `Evaluation`.
- [ ] Add test coverage for `SustainabilityEvaluation` (grid, field, modal lines).

---

## Section 4 — Examples + docs

### Findings reviewed
_Complete — 2026-02-25._

### Decisions

1. `dd-000-engine.md` requires a comprehensive revision (not just minor edits):
   - Remove `forest.py` / `ir.py` from the public surface description (moving to experimental JIT branch).
   - Remove or demote the JIT/Numba compilation section (experimental; `jit.py` → `numpy_ast.py`).
   - Add a section on timeseries nodes (`timeseries_constant`, `timeseries_placeholder`).
   - Add a section on axis management (`project_using_sum`, `project_using_mean`, `expand_dims`, `squeeze`).
2. `overtourism_molveno.py` — migrate to the new overtourism vertical after the vertical is created. Until then it is intentionally non-functional during transition.
3. `mobility_bologna.py` — update after `piecewise()` move and `Model` introduction (minor).
4. User-facing tutorial / conceptual guide — post-redesign task.

### Action items

- [ ] Revise `docs/design/dd-000-engine.md`: remove `forest`/`ir` from public surface; demote JIT section; add timeseries node documentation; add axis management documentation.
- [ ] Restructure `examples/` into two subfolders: `examples/overtourism_molveno/` and `examples/mobility_bologna/`.
- [ ] Move `reference_models/molveno/` content into `examples/overtourism_molveno/`; remove `reference_models/` directory.
- [ ] Migrate `examples/overtourism_molveno/overtourism_molveno.py` to new overtourism vertical (blocked until vertical is created).
- [ ] Update `examples/mobility_bologna/mobility_bologna.py`: `Piecewise` → `piecewise()` (lowercase); adopt generic `Model` class.
- [ ] Write user-facing tutorial/conceptual guide (post-redesign).

---

## Cross-cutting

### Findings reviewed
_Complete — 2026-02-25._

### Decisions

1. `sympyke.Eq` — remove; use `==` operator on `GenericIndex` directly.
2. `sympyke.Piecewise` — move to `graph.py` as `piecewise()` (lowercase, module-level convenience function constructing a `multi_clause_where` node from `(expr, cond)` clauses).
3. `sympyke.Symbol` / `_SymbolTable` — dissolve as part of the model redesign; symbols become proper `Index` (placeholder) objects belonging to a `Model`.
4. `sympyke` module — remove entirely once the above three are resolved.
5. API surface (`__init__.py`) — update after the model/simulation redesign: remove `AbstractModel`, `InstantiatedModel`, `ContextVariable`, `PresenceVariable`, `Constraint`, `Ensemble` from core exports; add `Model`; add `piecewise`; overtourism vertical exports its own symbols.
6. Documentation — add user-facing docs for `Model`, `Evaluation`, and the vertical pattern after the redesign is complete.

### Action items

**Step 1 — `sympyke` cleanup (during Sections 2+3 caller migration)**
- [ ] Remove `sympyke.Eq`; update all callers to use `==` on graph nodes / `GenericIndex` directly.
- [ ] Remove `sympyke.Piecewise` alias (callers already updated to `graph.piecewise()` during migration).

**Step 2 — `sympyke` removal (after model redesign)**
- [ ] Remove `sympyke.Symbol` / `_SymbolTable` once `Symbol` usages are replaced by proper `Index` objects.
- [ ] Delete `internal/sympyke/` module entirely.

**Step 3 — API surface (after model redesign)**
- [ ] Update `dt_model/__init__.py`: remove symbols that move to overtourism vertical (`AbstractModel`, `InstantiatedModel`, `ContextVariable`, `CategoricalContextVariable`, `UniformCategoricalContextVariable`, `ContinuousContextVariable`, `PresenceVariable`, `Constraint`, `Ensemble`); add `Model`, `piecewise`; ensure exported surface matches the new architecture.
- [ ] Review and update all other `__init__.py` files in the package (engine, symbols, model, simulation submodules) to ensure their docstrings and exports are accurate and consistent after the redesign.

**Step 4 — Documentation (after model redesign)**
- [ ] Write a design document for the model/simulation layer (analogous to `dd-000-engine.md`): cover `Model`, abstract/instantiated distinction, `Evaluation`, `WeightedScenario`, and the vertical extension pattern.
- [ ] Write a user-facing tutorial / getting-started guide covering both the Bologna paradigm (direct engine use) and the overtourism paradigm (vertical extension).
- [ ] Update `README.md` to include a conceptual overview of the framework and link to the new documentation.
