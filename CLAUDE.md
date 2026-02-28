# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

All commands assume `uv` is available and the repo is cloned.

```bash
# Run all tests (with coverage)
uv run pytest

# Run a single test file
uv run pytest tests/dt_model/simulation/test_evaluation.py

# Run a single test
uv run pytest tests/dt_model/simulation/test_evaluation.py::test_1d_single_scenario_placeholder

# Lint and format
uv run ruff check .
uv run ruff format .

# Type-check
uv run pyright

# Run an example script directly
uv run python examples/overtourism_molveno/overtourism_molveno.py
```

## Architecture

The library is in `civic_digital_twins/dt_model/` with three layers:

### Engine layer (`engine/`)

A numpy-backed computation graph (DAG). The public API lives in `engine/frontend/graph.py`:
- `placeholder(name, default)` / `constant(value)` — leaf nodes
- Arithmetic, comparison, logical, and conditional operators (`piecewise`, `where`, etc.)
- Shape ops: `expand_dims`, `project_using_sum`, `project_using_mean`
- `graph.Node` objects support infix operators (`+`, `*`, `==`, etc.) which return new nodes

`engine/frontend/linearize.py` topologically sorts a subgraph (`linearize.forest(*nodes)`).

`engine/numpybackend/executor.py` evaluates a sorted node list. The entry point is:
```python
state = executor.State(substitutions: dict[graph.Node, np.ndarray])
executor.evaluate_nodes(state, *sorted_nodes)
result = state.values[some_node]
```

### Model layer (`model/`)

- `model/index.py` — index types: `GenericIndex` (ABC), `Index` (placeholder or formula), `ConstIndex`, `TimeseriesIndex`, `UniformDistIndex`, `LognormDistIndex`, `TriangDistIndex`, and the `Distribution` Protocol.
- `model/model.py` — `Model(name, indexes)`: a named collection of `GenericIndex` objects. `model.abstract_indexes()` returns indexes whose value is `None` or a `Distribution` (i.e., requires external assignment before evaluation). `model.is_instantiated()` returns `True` when all indexes have concrete values.

**Critical**: `GenericIndex.__eq__` returns a `graph.Node` (not `bool`) to support formula composition. `__hash__` is identity-based (`id(self)`). Never use `x in list` for index membership — use `{id(x) for x in collection}` instead.

### Simulation layer (`simulation/`)

- `evaluation.py` — `WeightedScenario = tuple[float, dict[GenericIndex, Any]]` and `Evaluation(model)`.
  - `Evaluation.evaluate(scenarios, nodes_of_interest, *, axes=None, functions=None)` has two modes:
    - **1-D mode** (`axes=None`): each abstract index is stacked across S scenarios → result shape `(S,)`.
    - **Grid mode** (`axes={idx: arr, ...}`): axis indexes get shape `(1,…,N_i,…,1,1)`, non-axis abstract indexes get shape `(1,…,1,S)` → result broadcasts to `(N_0,…,N_k,S)`; caller marginalises over the last dim with `np.tensordot`.
- `ensemble.py` — `Ensemble` Protocol: any iterable of `WeightedScenario`.

### Examples (`examples/`)

Examples are on `sys.path` for tests (see `pythonpath` in `pyproject.toml`). Each example is a package that imports from `civic_digital_twins.dt_model` and demonstrates the vertical extension pattern:

- `examples/overtourism_molveno/` — full vertical: `ContextVariable`, `PresenceVariable`, `Constraint` (`@dataclass(eq=False)` to preserve identity hash), `OvertourismModel`, `OvertourismEnsemble` (samples distribution-backed indexes at construction; uses `pv_ids = {id(pv) for pv in model.pvs}` for presence-variable detection).
- `examples/mobility_bologna/` — direct engine use without the model/simulation layer.

Example scripts (e.g. `overtourism_molveno.py`) add their parent `examples/` directory to `sys.path` at the top so they can be run directly with `uv run python`.

## Conventions

- Commit style: `type(scope): message` (e.g. `feat(model): …`, `fix: …`). No "Co-Authored-By" trailers.
- Docstring style: NumPy convention (enforced by ruff `D` + `pydocstyle`).
- Line length: 120. Quotes: double. Formatter: ruff.
- `@dataclass` suppresses `__hash__`; use `@dataclass(eq=False)` for objects that must be usable as dict keys.