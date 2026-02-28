# Model / Simulation Layer

|                  |                                                |
| ---------------- | ---------------------------------------------- |
| Author           | [@bassosimone](https://github.com/bassosimone) |
| Last-Updated     | 2026-02-28                                     |
| Status           | Draft                                          |
| Approved-By      | N/A                                            |

The [dt_model](../../civic_digital_twins/dt_model) package provides
a model/simulation layer built on top of the
[engine](dd-cdt-engine.md).  Where the engine deals with raw DAG nodes
and NumPy arrays, this layer offers named, typed index variables, a
uniform model abstraction, and a generic evaluation pipeline that wires
ensembles of weighted scenarios to the engine.

(See the [Appendix](#appendix) for a glossary.)

## TL;DR

**Index.** An *index* is a named wrapper around a computation graph node.
It can be a constant, a probability distribution (sampled at evaluation
time), a fixed formula referencing other indexes, or an unbound
placeholder.  Indexes can be used directly in formulas with Python
arithmetic operators (`+`, `*`, `/`, …) without unwrapping the underlying
node.

**Model.** A *model* is a named collection of indexes.  A model is
*abstract* when it contains at least one distribution-backed or
placeholder index whose value must be supplied externally before the
model can be evaluated.  It is *instantiated* when all indexes are
fully concrete.

**WeightedScenario / Ensemble.** The bridge from abstract to instantiated
is an *ensemble*: an iterable of *weighted scenarios*, each pairing a
probability weight with a dictionary that maps every abstract index to a
concrete value.  The `Ensemble` type is a structural Protocol — any
iterable of `(float, dict)` pairs satisfies it.

**Evaluation.** `Evaluation(model).evaluate(scenarios)` consumes an
ensemble, builds the engine substitution dictionary from the scenario
assignments, runs `executor.evaluate_nodes`, and returns an
`EvaluationResult`.  The result provides typed access to node arrays and
weighted marginalisation over the scenario dimension.

**Grid mode.** `evaluate(scenarios, axes={idx: array, …})` extends the
1-D scenario mode to multi-dimensional grids.  Axis indexes are swept
over a dense grid; non-axis abstract indexes are drawn from the ensemble.
Result arrays broadcast to shape `(N₀, N₁, …, S)` where each `Nᵢ`
corresponds to one grid axis and `S` is the scenario count.

## Index Types

The module [`model/index.py`](../../civic_digital_twins/dt_model/model/index.py)
defines all index types.  The class hierarchy is:

```
GenericIndex  (ABC)
├── Index
│   ├── UniformDistIndex
│   ├── LognormDistIndex
│   ├── TriangDistIndex
│   └── ConstIndex
└── TimeseriesIndex
```

### GenericIndex

`GenericIndex` is the abstract base class.  It exposes:

- **`.node`** — the underlying `graph.Node`; all arithmetic and
  comparison operators on a `GenericIndex` delegate here, returning a
  new `graph.Node`.
- **`.sum(axis=-1)` / `.mean(axis=-1)`** — convenience wrappers for
  `graph.project_using_sum` / `graph.project_using_mean`.
- **Identity-based `__hash__`** — because `__eq__` is overridden to
  return a graph node (lazy evaluation), `__hash__` must be kept
  identity-based so that `GenericIndex` objects can be used as
  dictionary keys.  *Never use `in` to test membership in a list of
  `GenericIndex` objects* — use `{id(x) for x in collection}` instead.

### Index

`Index(name, value)` is the workhorse class.  The `value` argument
determines the mode:

| `value` type | Mode | graph node created |
| ------------ | ---- | ------------------ |
| `Distribution` | distribution-backed (abstract) | `graph.placeholder` |
| `float` / `int` / `str` / `bool` | constant | `graph.constant` |
| `graph.Node` | formula | the node itself |
| `None` | explicit placeholder (abstract) | `graph.placeholder` |

```python
from civic_digital_twins.dt_model.model.index import (
    Index, UniformDistIndex, ConstIndex
)
from scipy import stats

# Distribution-backed (abstract — must be resolved in each scenario)
mu = Index("mu", stats.norm(loc=0.5, scale=0.1))

# Constant
cap = ConstIndex("capacity", 500.0)

# Formula referencing other indexes
load = Index("load", mu * cap)

# Explicit placeholder (resolved by the caller)
demand = Index("demand", None)
```

The concrete subclasses `UniformDistIndex`, `LognormDistIndex`,
`TriangDistIndex`, and `ConstIndex` are convenience wrappers that
construct the appropriate `scipy.stats` frozen distribution or scalar
constant and pass it to `Index.__init__`.

### TimeseriesIndex

`TimeseriesIndex(name, values)` wraps a time-indexed quantity.

| `values` type | Mode | graph node created |
| ------------- | ---- | ------------------ |
| `np.ndarray` | fixed array | `graph.timeseries_constant` |
| `graph.Node` | formula | the node itself |
| `None` (default) | placeholder | `graph.timeseries_placeholder` |

```python
import numpy as np
from civic_digital_twins.dt_model.model.index import TimeseriesIndex

# Fixed time series
flow = TimeseriesIndex("flow", np.array([10.0, 20.0, 30.0]))

# Placeholder (externally supplied)
demand_ts = TimeseriesIndex("demand_ts")
```

## Model

[`model/model.py`](../../civic_digital_twins/dt_model/model/model.py)

```python
class Model:
    def __init__(self, name: str, indexes: list[GenericIndex]) -> None: ...
    def abstract_indexes(self) -> list[GenericIndex]: ...
    def is_instantiated(self) -> bool: ...
```

`Model` is a plain container.  It does not build the graph — that is
done by constructing `Index` and `TimeseriesIndex` objects beforehand.
The model merely collects them so that `Evaluation` and ensemble classes
can inspect which indexes are abstract.

`abstract_indexes()` returns indexes whose `value` is `None` or a
`Distribution`.  All other indexes (constants and formulas) are concrete
and are not returned.

```python
from civic_digital_twins.dt_model.model.model import Model
from civic_digital_twins.dt_model.model.index import UniformDistIndex, Index

x = UniformDistIndex("x", loc=0.0, scale=10.0)
y = UniformDistIndex("y", loc=0.0, scale=10.0)
z = Index("z", x + y)

model = Model("demo", [x, y, z])
print(model.abstract_indexes())   # [x, y]
print(model.is_instantiated())    # False
```

Models can be subclassed to add domain-specific structure (labeled
subsets of indexes, constraint lists, etc.) while preserving the
core contract.  See [Vertical Extension](#vertical-extension) below.

## Ensemble and WeightedScenario

[`simulation/ensemble.py`](../../civic_digital_twins/dt_model/simulation/ensemble.py)

```python
WeightedScenario = tuple[float, dict[GenericIndex, Any]]

class Ensemble(Protocol):
    def __iter__(self) -> Iterator[WeightedScenario]: ...
```

`WeightedScenario` is a `(weight, assignments)` pair where:
- `weight` is a probability (all weights in an ensemble should sum to 1).
- `assignments` maps every abstract index that is not an axis to a
  concrete `np.ndarray` value.

`Ensemble` is a `runtime_checkable` Protocol: any iterable of
`WeightedScenario` tuples satisfies it without inheriting from a base
class.

### DistributionEnsemble

`DistributionEnsemble(model, size, rng=None)` is the standard ensemble
for models whose abstract indexes are all distribution-backed.  It draws
`size` independent samples from each distribution and yields
`size` equally-weighted scenarios (weight `1/size` each).

```python
from civic_digital_twins.dt_model.simulation.ensemble import DistributionEnsemble

ensemble = DistributionEnsemble(model, size=100)
scenarios = list(ensemble)
# Each scenario: (0.01, {x: array([...]), y: array([...])})
```

`DistributionEnsemble` raises `ValueError` at construction if any
abstract index of the model is not distribution-backed.

## Evaluation

[`simulation/evaluation.py`](../../civic_digital_twins/dt_model/simulation/evaluation.py)

```python
class Evaluation:
    def __init__(self, model: Model) -> None: ...

    def evaluate(
        self,
        scenarios: Ensemble,
        nodes_of_interest: list[GenericIndex] | None = None,
        *,
        axes: dict[GenericIndex, np.ndarray] | None = None,
        functions: dict[str, executor.Functor] | None = None,
    ) -> EvaluationResult: ...
```

### 1-D mode

When `axes` is `None` (or an empty dict), `evaluate` operates in *1-D
mode*: it stacks the per-scenario values for each abstract index into a
single substitution array and evaluates the graph once.

For a model with abstract indexes `[x, y]` and `S` scenarios:

1. For each abstract index `idx`, collect its values across all
   scenarios: `values = [assignments[idx] for _, assignments in scenarios]`.
2. Stack: `stacked = np.asarray(values)`, shape `(S,)` for scalars or
   `(S, T)` for timeseries-shaped values (shape `(1,)` per scenario
   → `(S, 1)` after stacking, which broadcasts with shape `(T,)`).
3. Build the engine substitution dict: `{idx.node: stacked, …}`.
4. Run `executor.evaluate_nodes`.

Result arrays have shape `(S,)` for scalar formulas or `(S, T)` for
timeseries formulas.  Use `result.marginalize(idx)` to compute the
weighted mean over `S`.

### Grid mode

When `axes={pv₀: arr₀, pv₁: arr₁, …}` is provided, `evaluate` operates
in *grid mode*:

- Each axis index at position `i` contributes a substitution of shape
  `(1, …, Nᵢ, …, 1, 1)` where `Nᵢ = arrᵢ.size` and the non-unit
  dimension is at position `i`.
- Each non-axis abstract index contributes a substitution of shape
  `(1, …, 1, S)` — one sample per scenario, broadcast-compatible with
  all axis dimensions.
- Result arrays broadcast to shape `(N₀, N₁, …, S)`.

Use `result.marginalize(idx)` to contract the `S` dimension:

```python
# shape (N₀, N₁, …, S) → (N₀, N₁, …)
marginalised = result.marginalize(idx)
```

Grid mode is the standard way to compute sustainability fields in
overtourism models, where the two presence variables define the grid
axes.

### EvaluationResult

`EvaluationResult` wraps the executor state and provides:

| API | Description |
| --- | ----------- |
| `result[idx]` | Raw array for `idx` (not yet broadcast to `full_shape`). |
| `result.marginalize(idx)` | Broadcast to `full_shape`, then `tensordot` with `weights`. |
| `result.weights` | Scenario weights, shape `(S,)`. |
| `result.axes` | The `axes` dict passed to `evaluate`. |
| `result.full_shape` | `(N₀, …, Nₖ, S)` — shape of a fully-broadcast array. |

### End-to-End Example (1-D mode)

```python
import numpy as np
from civic_digital_twins.dt_model import Evaluation, Model
from civic_digital_twins.dt_model.model.index import UniformDistIndex, Index
from civic_digital_twins.dt_model.simulation.ensemble import DistributionEnsemble

# Define the model
x = UniformDistIndex("x", loc=0.0, scale=10.0)
y = UniformDistIndex("y", loc=0.0, scale=10.0)
z = Index("z", x + y)
model = Model("demo", [x, y, z])

# Build an ensemble of 200 scenarios
ensemble = DistributionEnsemble(model, size=200)

# Evaluate
result = Evaluation(model).evaluate(ensemble)

# Weighted mean of z across all scenarios
print(result.marginalize(z))  # ≈ 10.0
```

## Vertical Extension

The core library (`dt_model`) is domain-agnostic.  Domain-specific
models are built by subclassing `Model` and composing `Index` objects
with domain semantics, without modifying the core library.

The overtourism example
(`examples/overtourism_molveno/overtourism_metamodel.py`) illustrates the
pattern.  The key classes are:

### ContextVariable

A `ContextVariable` is an `Index` with `value=None` (explicit
placeholder) and a `sample(nr, subset, force_sample)` method that
returns a list of `(probability, value)` pairs.

Three concrete subclasses are provided:

| Class | Distribution |
| ----- | ------------ |
| `UniformCategoricalContextVariable` | Uniform over a finite set of values. |
| `CategoricalContextVariable` | Categorical with explicit probabilities. |
| `ContinuousContextVariable` | Backed by a `scipy` continuous distribution. |

```python
CV_weather = CategoricalContextVariable(
    "weather",
    {"good": 0.5, "unsettled": 0.3, "bad": 0.2},
)
```

### PresenceVariable

A `PresenceVariable` is an `Index` with `value=None` and a
`sample(cvs, nr)` method that draws `nr` presence values from a
context-dependent truncated-normal distribution.  The distribution
parameters are functions of the current CV assignments.

### Constraint

```python
@dataclass(eq=False)
class Constraint:
    name: str
    usage: Index     # formula-mode index for usage
    capacity: Index  # constant or distribution-backed capacity
```

`eq=False` preserves identity-based `__hash__` so that `Constraint`
objects can be used as dictionary keys.

### OvertourismModel

`OvertourismModel(Model)` is a `Model` subclass that organises its
indexes into labeled subsets:

```python
model.cvs          # list[ContextVariable]
model.pvs          # list[PresenceVariable]
model.domain_indexes   # list[Index]  (e.g. scaling factors)
model.capacities   # list[Index]      (capacity indexes)
model.constraints  # list[Constraint]
```

The constructor automatically adds the usage index of each constraint
to the flat `indexes` list so that `abstract_indexes()` and
`Evaluation` can find them.

### OvertourismEnsemble

`OvertourismEnsemble(model, scenario, cv_ensemble_size)` yields
`WeightedScenario` instances by:

1. Enumerating all combinations of CV values from `scenario`
   (using `itertools.product`).
2. For each combination, sampling `cv_ensemble_size` values from every
   distribution-backed non-PV non-CV abstract index.
3. Yielding `(weight, assignments)` pairs with equal weight across
   all `|CV combinations| × cv_ensemble_size` scenarios.

```python
ensemble = OvertourismEnsemble(
    model,
    {CV_weather: ["good", "unsettled", "bad"]},
    cv_ensemble_size=20,
)
```

### Grid Evaluation with OvertourismEnsemble

The standard overtourism evaluation pattern:

```python
import numpy as np
from civic_digital_twins.dt_model import Evaluation
from civic_digital_twins.dt_model.model.index import Distribution

tt = np.linspace(0, 50_000, 101)   # tourist presence axis
ee = np.linspace(0, 50_000, 101)   # excursionist presence axis

result = Evaluation(model).evaluate(
    list(ensemble),
    axes={PV_tourists: tt, PV_excursionists: ee},
)

# Compute sustainability field per constraint
field = np.ones((tt.size, ee.size))
for c in model.constraints:
    usage = np.broadcast_to(result[c.usage], result.full_shape)
    if isinstance(c.capacity.value, Distribution):
        mask = 1.0 - c.capacity.value.cdf(usage)
    else:
        cap = np.broadcast_to(result[c.capacity], result.full_shape)
        mask = (usage <= cap).astype(float)
    field *= np.tensordot(mask, result.weights, axes=([-1], [0]))
```

Result arrays have shape `(tt.size, ee.size, S)`; after `tensordot`
marginalisation over `S`, the field has shape `(tt.size, ee.size)`.

## Design Rationale

### Why a single `Model` class?

Earlier versions had separate `AbstractModel` and `InstantiatedModel`
classes.  The distinction is now expressed through `abstract_indexes()`
and `is_instantiated()`, and the concrete-value binding is done at the
`Evaluation` call site via weighted scenarios.  This eliminates
mutation-based model instantiation and makes the data flow explicit.

### Why a structural `Ensemble` Protocol?

Making `Ensemble` a structural `Protocol` (rather than a base class)
means that any iterable of `(float, dict)` pairs satisfies the contract
without inheritance.  A `list[WeightedScenario]`, a generator, or a
domain-specific class all work transparently.

### Why `GenericIndex.__hash__` is identity-based

`GenericIndex.__eq__` returns a `graph.Node` (lazy evaluation) rather
than `bool`.  This is intentional — it allows writing formulas such as
`graph.piecewise((expr, cv == "good"), …)`.  But it means `__hash__`
must not call `__eq__`; identity-based hashing is the standard Python
fallback and is exactly what `graph.Node` itself uses.

### Why `Constraint` uses `@dataclass(eq=False)`

The `@dataclass` decorator normally generates `__eq__` (and suppresses
`__hash__`).  Since `Constraint` objects are used as dict keys in the
overtourism field computation, `@dataclass(eq=False)` suppresses the
generated `__eq__` and preserves the identity-based `__hash__` inherited
from `object`.

## Appendix

### Glossary

**Abstract index**: an `Index` whose `value` is `None` or a
`Distribution`; it needs an external value before the model can be
evaluated.

**Concrete index**: an `Index` whose `value` is a scalar constant or a
`graph.Node` formula; it can be evaluated without external input.

**Ensemble**: an iterable of `WeightedScenario` tuples that defines a
discrete probability distribution over model instantiations.

**Grid mode**: the `axes=` keyword of `Evaluation.evaluate`; sweeps
axis indexes over a dense grid while the ensemble provides the
non-axis abstract index values.

**Instantiated model**: a model in which `is_instantiated()` returns
`True` — all indexes are concrete.

**Marginalize**: collapse the scenario dimension `S` by computing
`tensordot(arr, weights, axes=([-1], [0]))`, returning the weighted
expectation over scenarios.

**Vertical extension**: the pattern of subclassing `Model` (and
composing domain-specific `Index` subclasses) to add domain semantics
without modifying the core library.

**WeightedScenario**: `tuple[float, dict[GenericIndex, Any]]` — a
probability weight paired with an assignment of every non-axis abstract
index to a concrete `np.ndarray` value.
