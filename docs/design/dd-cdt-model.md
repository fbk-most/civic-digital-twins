<!-- SPDX-License-Identifier: Apache-2.0 -->

# Model / Simulation Layer

|              | Document data                                  |
|--------------| ---------------------------------------------- |
| Author       | [@pistore](https://github.com/pistore)         |
| Last-Updated | 2026-05-02                                     |
| Status       | Draft                                          |
| Approved-By  | N/A                                            |

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

**Conditional indexes.** `ConditionalCategoricalIndex` and
`ConditionalDistributionIndex` are indexes whose distribution or probability
table depends on the resolved values of *parent* indexes.  They are used
for context-dependent quantities such as visitor counts that vary by season
and weather.

**AxisEnsemble.** The bridge from abstract to instantiated is an
*ensemble*: a batched object that assigns concrete arrays to every abstract
index.  `DistributionEnsemble`, `PartitionedEnsemble`, and
`CrossProductEnsemble` implement the `AxisEnsemble` protocol, which exposes
named ENSEMBLE axes with factorized weight vectors.  `CrossProductEnsemble`
is purpose-built for models with categorical context variables: it enumerates
all category combinations and pre-samples stochastic capacities.

**Evaluation.** `Evaluation(model).evaluate(ensemble=…, parameters=…)`
consumes an ensemble, builds the engine substitution dictionary from the
batched assignments, runs `executor.evaluate_nodes`, and returns an
`EvaluationResult`.  The result provides typed access to node arrays and
weighted marginalisation over ENSEMBLE and PARAMETER axes.

**Grid mode.** `evaluate(ensemble=…, parameters={idx: array, …})` extends
ensemble evaluation to multi-dimensional parameter grids.  PARAMETER
indexes are swept over a dense grid; abstract indexes are handled by the
ensemble.  Result arrays have canonical shape `(*PARAMETER, *ENSEMBLE)`
where each PARAMETER size is `Nᵢ` and the ENSEMBLE size is `S`.

## Index Types

The module [`model/index.py`](../../civic_digital_twins/dt_model/model/index.py)
defines all index types.  The class hierarchy is:

```
GenericIndex  (ABC)
├── Index
│   ├── ConstIndex
│   ├── DistributionIndex
│   ├── ConditionalDistributionIndex
│   ├── CategoricalIndex
│   └── ConditionalCategoricalIndex
└── TimeseriesIndex
```

### GenericIndex

`GenericIndex` is the abstract base class.  It exposes:

- **`.node`** — the underlying `graph.Node`; all arithmetic and
  comparison operators on a `GenericIndex` delegate here, returning a
  new `graph.Node`.
- **Axis reduction methods** — convenience wrappers for axis reduction operators:
  `.sum(axis=-1)`, `.mean(axis=-1)`, `.min(axis=-1)`, `.max(axis=-1)`,
  `.std(axis=-1)`, `.var(axis=-1)`, `.median(axis=-1)`, `.prod(axis=-1)`,
  `.any(axis=-1)`, `.all(axis=-1)`, `.count_nonzero(axis=-1)`,
  and `.quantile(q, axis=-1)`. These delegate to the corresponding
  `graph.project_using_*` operators.
- **Identity-based `__hash__`** — because `__eq__` is overridden to
  return a graph node (lazy evaluation), `__hash__` must be kept
  identity-based so that `GenericIndex` objects can be used as
  dictionary keys.  *Never use `in` to test membership in a list of
  `GenericIndex` objects* — use `{id(x) for x in collection}` instead.

### Index

`Index(name, value)` is the base concrete index class.  In most cases you will use one of the
dedicated subclasses (`DistributionIndex`, `ConstIndex`, `CategoricalIndex`) rather than
`Index` directly.  `Index` itself is appropriate when passing a pre-frozen distribution or a
`graph.Node` formula as the value:

```python
from scipy import stats

from civic_digital_twins.dt_model import ConstIndex, DistributionIndex, Index

# Distribution-backed (abstract — must be resolved in each scenario)
# Pass any scipy-compatible distribution callable and a params dict:
cap_dist = DistributionIndex("capacity", stats.uniform, {"loc": 400.0, "scale": 200.0})
mu       = DistributionIndex("mu",       stats.norm,    {"loc": 0.5,   "scale": 0.1})

# Constant
cap = ConstIndex("capacity", 500.0)

# Formula referencing other indexes
load = Index("load", mu * cap)

# Explicit placeholder (resolved by the caller)
demand = Index("demand", None)
```

`DistributionIndex(name, distribution, params)` accepts any callable that
returns a `Distribution`-conformant object (e.g. any `scipy.stats`
distribution) plus a `params` dict forwarded verbatim.  The `params`
property supports full replacement (`idx.params = {...}`) and partial
update via the Python dict-merge operator (`idx.params |= {"loc": 200}`).

`ConstIndex` is a convenience wrapper that accepts a scalar constant and
passes it to `Index.__init__`.

### ConditionalDistributionIndex

`ConditionalDistributionIndex(name, parents, factory)` is always abstract
(placeholder mode).  The `factory` is called with the current parent values as
keyword arguments and must return a frozen scipy distribution.  The ensemble
calls the factory per-scenario to obtain the distribution; during grid evaluation
the index is supplied as a PARAMETER axis and is not resolved per-scenario.

```python
from scipy import stats

from civic_digital_twins.dt_model import CategoricalIndex, ConditionalDistributionIndex

cv_weather = CategoricalIndex("weather", {"good": 0.5, "bad": 0.5})

def load_dist(weather):
    if weather == "good":
        return stats.uniform(loc=100.0, scale=200.0)
    return stats.uniform(loc=50.0, scale=100.0)

pv_load = ConditionalDistributionIndex("load", [cv_weather], load_dist)

assert pv_load.value is None  # abstract — resolved per-scenario by the ensemble
```

The factory receives string values for `CategoricalIndex` parents and float values
for `DistributionIndex` parents, matching the types stored in `ensemble.assignments()`.

### CategoricalIndex

`CategoricalIndex(name, outcomes)` is a placeholder `Index` whose per-scenario values are strings
drawn from a finite named set.  It extends `Index` with `value=None`, so it is automatically
abstract and must be resolved in every scenario.

| Parameter | Type | Description |
|-----------|------|-------------|
| `name` | `str` | Human-readable name. |
| `outcomes` | `dict[str, float]` | Maps outcome key to probability.  Values must be positive and sum to 1.0. |

```python
from civic_digital_twins.dt_model import CategoricalIndex

mode = CategoricalIndex("mode", {"bike": 0.3, "train": 0.7})
```

Because the full `GenericIndex` algebra protocol is inherited, `mode == "bike"` produces a
`graph.equal` node usable in formulas and `graph.piecewise` guards.

For the usage pattern and integration with `ModelVariant`, see
[`dd-cdt-modularity.md`](dd-cdt-modularity.md#runtime-variant-selection).

### ConditionalCategoricalIndex

`ConditionalCategoricalIndex(name, parents, support, factory)` is a `CategoricalIndex`
whose per-outcome probabilities depend on the resolved values of *parent* indexes.
The `support` list declares all possible outcome strings.  The `factory` is called
with parent values as keyword arguments and must return a `dict[str, float]` mapping
each support value to its probability (must sum to 1.0).

```python
from civic_digital_twins.dt_model import CategoricalIndex, ConditionalCategoricalIndex

cv_season = CategoricalIndex("season", {"low": 0.6, "high": 0.4})

def weekend_probs(season):
    return {"yes": 0.4, "no": 0.6} if season == "low" else {"yes": 0.3, "no": 0.7}

cv_weekend = ConditionalCategoricalIndex("weekend", [cv_season], ["yes", "no"], weekend_probs)

assert cv_weekend.value is None  # abstract
```

Like `CategoricalIndex`, `cv_weekend == "yes"` returns a `graph.equal` node usable
in `graph.piecewise` guards.

### TimeseriesIndex

`TimeseriesIndex(name, values)` wraps a time-indexed quantity.

| `values` type | Mode | graph node created |
| ------------- | ---- | ------------------ |
| `np.ndarray` | fixed array | `graph.timeseries_constant` |
| `graph.Node` | formula | the node itself |
| `None` (default) | placeholder | `graph.timeseries_placeholder` |

```python
import numpy as np

from civic_digital_twins.dt_model import TimeseriesIndex

# Fixed time series
flow = TimeseriesIndex("flow", np.array([10.0, 20.0, 30.0]))

# Placeholder (externally supplied)
demand_ts = TimeseriesIndex("demand_ts")
```

## Model

[`model/model.py`](../../civic_digital_twins/dt_model/model/model.py)

```python
class Model:
    def __init__(
        self,
        name: str,
        indexes: list[GenericIndex] | None = None,  # deprecated
        *,
        inputs:  Any | None = None,   # dataclass instance (new API)
        outputs: Any | None = None,   # dataclass instance (new API)
        expose:  Any | None = None,   # dataclass instance (new API)
    ) -> None: ...
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

### New dataclass-based API (recommended)

Declare `Inputs`, `Outputs`, and optionally `Expose` as inner
`@dataclass` classes on the subclass.  Construct instances of these
dataclasses and pass them to `super().__init__()` via the keyword
arguments `inputs=`, `outputs=`, and `expose=`.  `model.indexes` is
derived automatically by collecting and deduplicating all scalar
`GenericIndex` values found in `inputs`, `outputs`, and `expose` — no
manual list needed.

**Three access levels** define the visibility contract:

1. `model.outputs.<field>` / `model.inputs.<field>` — **contractual,
   stable**: these are the primary wiring points between models and the
   evaluation layer.
2. `model.expose.<field>` — **inspectable, not contracted**: useful for
   debugging or visualisation, but `Expose` fields MUST NOT be used to
   wire indexes between models.
3. Local variables inside `__init__` — **internal, not accessible**
   outside the constructor.

**Inputs contract convention**: every `GenericIndex` that is passed as a
constructor parameter must be declared as a field in `Inputs`.  If a
`GenericIndex` parameter is absent from `Inputs`, an
`InputsContractWarning` is emitted naming the offending parameter (see
[Contract Warnings](#contract-warnings)).

```python
from dataclasses import dataclass

from scipy import stats

from civic_digital_twins.dt_model import DistributionIndex, Index, Model

class DemoModel(Model):

    @dataclass
    class Outputs:
        z: Index

    def __init__(self) -> None:
        x = DistributionIndex("x", stats.uniform, {"loc": 0.0, "scale": 10.0})
        y = DistributionIndex("y", stats.uniform, {"loc": 0.0, "scale": 10.0})
        z = Index("z", x + y)

        super().__init__(
            "demo",
            outputs=DemoModel.Outputs(z=z),
        )

m = DemoModel()
print(m.abstract_indexes())   # [x, y]  — derived automatically
print(m.is_instantiated())    # False
```

### Legacy `indexes=` API

Passing a flat `list[GenericIndex]` via the positional `indexes`
parameter still works but emits a `DeprecationWarning`.  New code should
use the dataclass-based API above.

```python
from scipy import stats

from civic_digital_twins.dt_model import DistributionIndex, Index, Model

x = DistributionIndex("x", stats.uniform, {"loc": 0.0, "scale": 10.0})
y = DistributionIndex("y", stats.uniform, {"loc": 0.0, "scale": 10.0})
z = Index("z", x + y)

model = Model("demo", [x, y, z])   # DeprecationWarning
print(model.abstract_indexes())    # [x, y]
print(model.is_instantiated())     # False
```

Models can be subclassed to add domain-specific structure (labeled
subsets of indexes, constraint lists, etc.) while preserving the
core contract.  See [Vertical Extension](#vertical-extension) below.

## ModelVariant

[`model/model_variant.py`](../../civic_digital_twins/dt_model/model/model_variant.py)

```python
class ModelVariant:
    def __init__(
        self,
        name: str,
        variants: Mapping[str, Model],
        selector: str | CategoricalIndex | graph.Node,
    ) -> None: ...

    @staticmethod
    def guards_to_selector(
        guards: list[tuple[str, graph.Node | bool]],
    ) -> graph.Node: ...
```

`ModelVariant` selects among pre-constructed `Model` instances that share the same `outputs` field
names.  It operates in two modes:

**Static mode** (`selector: str`) — the active variant is resolved once at construction time.
`ModelVariant` acts as a fully transparent proxy for the active variant; all attribute access
delegates to it.  `outputs` field names must be identical across all variants.

**Runtime mode** (`selector: CategoricalIndex | graph.Node`) — the active variant is determined
per scenario at evaluation time.  `ModelVariant` builds a merged computation graph at construction:
`mv.outputs.x` is a real `Index` backed by a `exclusive_multi_clause_where` node, usable in parent
model formulas.  `inputs` may differ across variants (they are surfaced as a union); `outputs`
names must be identical.  See [`dd-cdt-modularity.md`](dd-cdt-modularity.md#runtime-variant-selection)
for full usage documentation.

`ModelVariant` is exported from `civic_digital_twins.dt_model`.

```python
from civic_digital_twins.dt_model import CategoricalIndex, ModelVariant

# Static
mv = ModelVariant("T", variants={"bike": BikeModel(), "train": TrainModel()}, selector="bike")

# Runtime — probabilistic
mode = CategoricalIndex("mode", {"bike": 0.3, "train": 0.7})
mv = ModelVariant("T", variants={"bike": BikeModel(), "train": TrainModel()}, selector=mode)
```

## Contract Warnings

[`model/model.py`](../../civic_digital_twins/dt_model/model/model.py)

Both warning classes are exported from `civic_digital_twins.dt_model`.

**`ModelContractWarning(UserWarning)`** — base class for all Model I/O
contract warnings.  Use

```python
import warnings
warnings.filterwarnings("error", category=ModelContractWarning)
```

to promote the entire family of contract warnings into hard errors,
which is recommended in test suites.

**`InputsContractWarning(ModelContractWarning)`** — emitted when a
`GenericIndex` constructor parameter is absent from the declared
`Inputs` dataclass.  The warning message names the offending parameter
precisely so it can be located and added to `Inputs`.

**`AbstractIndexNotInInputsWarning(ModelContractWarning)`** — emitted when
an abstract index (one whose value is `None` or a `Distribution`) is not
reachable via `self.inputs`.  Abstract indexes receive their values from
outside the model and are therefore inputs by definition.  Currently a soft
warning for backwards compatibility; planned for promotion to an error in a
future release.

Example — the following model triggers an `InputsContractWarning`
because `x` is a `GenericIndex` constructor parameter but is not
declared in `Inputs`:

```python
from dataclasses import dataclass

from scipy import stats

from civic_digital_twins.dt_model import DistributionIndex, Index, Model

class BadModel(Model):

    @dataclass
    class Outputs:
        z: Index

    def __init__(self, x: DistributionIndex) -> None:
        # x is a GenericIndex parameter but not in Inputs — warns!
        z = Index("z", x + x)
        super().__init__("bad", outputs=BadModel.Outputs(z=z))
```

Declare an `Inputs` dataclass and pass an instance to silence the
warning:

```python
class GoodModel(Model):

    @dataclass
    class Inputs:
        x: DistributionIndex

    @dataclass
    class Outputs:
        z: Index

    def __init__(self, x: DistributionIndex) -> None:
        z = Index("z", x + x)
        super().__init__(
            "good",
            inputs=GoodModel.Inputs(x=x),
            outputs=GoodModel.Outputs(z=z),
        )
```

## Ensemble

[`simulation/ensemble.py`](../../civic_digital_twins/dt_model/simulation/ensemble.py)

The canonical ensemble type is `AxisEnsemble`:

```python
class AxisEnsemble(Protocol):
    @property
    def ensemble_axes(self) -> tuple[Axis, ...]: ...
    @property
    def ensemble_weights(self) -> tuple[np.ndarray, ...]: ...
    def assignments(self) -> Mapping[GenericIndex, np.ndarray]: ...
```

Each `Axis` in `ensemble_axes` names one ENSEMBLE dimension.
`ensemble_weights` provides the per-axis weight vector (sums to 1).
`assignments()` returns batched arrays — one array per abstract index,
with ENSEMBLE dimensions at the positions declared by the axes.

> **Legacy API (deprecated):** `WeightedScenario = tuple[float, dict[GenericIndex, Any]]`
> and the `Ensemble` iterable protocol are still accepted by `evaluate()` but emit a
> `DeprecationWarning`.  Migrate to `AxisEnsemble` (e.g. `DistributionEnsemble`).

### DistributionEnsemble

`DistributionEnsemble(model, size, rng=None)` is the standard ensemble
for models whose abstract indexes are all distribution-backed.  It draws
`size` independent samples from each distribution into a single ENSEMBLE
axis with uniform weights (`1/size` each).

```python
from civic_digital_twins.dt_model import DistributionEnsemble

ensemble = DistributionEnsemble(model, size=100)
# ensemble.ensemble_axes   → (Axis("x", ENSEMBLE), …)
# ensemble.ensemble_weights[0]  → array of 100 weights summing to 1
assignments = ensemble.assignments()
# assignments[x]  → shape (100,) array of sampled values
```

`DistributionEnsemble` also handles `CategoricalIndex` abstract indexes automatically —
each categorical index is sampled proportional to its outcome weights.

A `ValueError` is raised at construction if any abstract index is neither
distribution-backed nor a `CategoricalIndex`.

### PartitionedEnsemble

`PartitionedEnsemble(model, axes, default_axis=None, rng=None)` creates
N independent ENSEMBLE axes, each covering a disjoint subset of the
model's abstract indexes.  Each `EnsembleAxisSpec` names the axis and
lists the indexes it covers:

```python
from civic_digital_twins.dt_model import EnsembleAxisSpec, PartitionedEnsemble

ens = PartitionedEnsemble(
    model,
    axes=[
        EnsembleAxisSpec("demand", indexes=[demand_idx], size=50),
        EnsembleAxisSpec("capacity", indexes=[cap_idx], size=20),
    ],
)
# Result arrays have shape (*PARAMETER, 50, 20) — two independent ENSEMBLE dims.
```

### CrossProductEnsemble

`CrossProductEnsemble(model, restrictions, max_categorical_size, exclude, rng)` implements
`AxisEnsemble`.  It materialises a batched ENSEMBLE axis by:

1. Discovering the model's abstract indexes via `model.abstract_indexes()`.
2. Enumerating all combinations of `CategoricalIndex` /
   `ConditionalCategoricalIndex` values (filtered by `restrictions`);
   probability weights are the product of per-category outcome probabilities.
   When a categorical's support exceeds `max_categorical_size`, values are
   Monte-Carlo sampled instead.
3. Pre-sampling one value per distribution-backed abstract index
   (e.g. stochastic capacities) for each scenario.

Indexes passed via `exclude` are skipped in steps 2–3 — they are provided
as PARAMETER axes at evaluation time.  See [Domain Modeling Pattern](#domain-modeling-pattern)
for a worked example.

```python
class CrossProductEnsemble:
    def __init__(
        self,
        model: Model | ModelVariant,
        restrictions: Mapping[Any, Sequence[str]] | None = None,
        max_categorical_size: int = 20,
        exclude: Sequence[GenericIndex] | None = None,
        rng: np.random.Generator | None = None,
    ) -> None: ...
    def __len__(self) -> int: ...
```

`sample_across(ensemble, indexes, total=200, rng=None)` draws weighted samples
from `ConditionalDistributionIndex` indexes across all scenarios — useful for
scatter-plot visualisation of presence variables against the sustainability field.

## Evaluation

[`simulation/evaluation.py`](../../civic_digital_twins/dt_model/simulation/evaluation.py)

```python
class Evaluation:
    def __init__(self, model: Model) -> None: ...

    def evaluate(
        self,
        ensemble: AxisEnsemble | None = None,
        nodes_of_interest: list[GenericIndex] | None = None,
        *,
        parameters: dict[GenericIndex, np.ndarray] | None = None,
        functions: dict[str, executor.Functor] | None = None,
    ) -> EvaluationResult: ...
```

### Ensemble mode

When `parameters` is `None`, `evaluate` operates in *ensemble mode*: the
`AxisEnsemble.assignments()` arrays are used as ENSEMBLE substitutions and
evaluated in a single batched pass.

For a model with abstract indexes `[x, y]` and ENSEMBLE size `S`:

1. `ensemble.assignments()` returns `{x: arr_x, y: arr_y}` where each
   `arr` has shape `(S,)` (or `(S, T)` for timeseries-shaped values).
2. Shapes are normalised to `(*PARAMETER, *ENSEMBLE)` canonical form.
3. Run `executor.evaluate_nodes` once.

Result arrays have shape `(S,)` for scalar formulas (plus any trailing
DOMAIN dims, e.g. `(S, T)` for timeseries).

### Grid mode

When `parameters={pv₀: arr₀, pv₁: arr₁, …}` is provided alongside an
ensemble, `evaluate` operates in *grid mode*:

- Each PARAMETER index at position `i` contributes a substitution of shape
  `(1, …, Nᵢ, …, 1)` where `Nᵢ = arrᵢ.size`.
- ENSEMBLE indexes get shapes `(1, …, 1, S, 1)` — broadcast-compatible
  with all PARAMETER dimensions.
- Result arrays have canonical shape `(*PARAMETER, *ENSEMBLE)`.

Use `result.expected_value(idx)` to contract all ENSEMBLE dimensions:

```python
# shape (*PARAMETER, *ENSEMBLE) → (*PARAMETER)
marginalised = result.expected_value(idx)
```

Grid mode is the standard way to compute sustainability fields in
overtourism models, where the two presence variables define the parameter
grid.

### EvaluationResult

`EvaluationResult` wraps the executor state and provides:

| API | Description |
| --- | ----------- |
| `result[idx]` | Raw array for `idx` in canonical `(*PARAMETER, *ENSEMBLE)` shape prefix. |
| `result.expected_value(idx)` | Contract all ENSEMBLE axes using factorized weights; result shape is `(*PARAMETER, *DOMAIN)`. |
| `result.weights` | Joint weight array (outer product of per-axis weights). |
| `result.parameter_values` | The `parameters=` dict passed to `evaluate`. |
| `result.full_shape` | `(*PARAMETER, *ENSEMBLE)` sizes in axis-layout order. |

### End-to-End Example

```python
from scipy import stats

from civic_digital_twins.dt_model import DistributionEnsemble, DistributionIndex, Evaluation, Index, Model

# Define the model
x = DistributionIndex("x", stats.uniform, {"loc": 0.0, "scale": 10.0})
y = DistributionIndex("y", stats.uniform, {"loc": 0.0, "scale": 10.0})
z = Index("z", x + y)
model = Model("demo", [x, y, z])

# Build an ensemble of 200 scenarios
ensemble = DistributionEnsemble(model, size=200)

# Evaluate
result = Evaluation(model).evaluate(ensemble=ensemble)

# Weighted mean of z across all scenarios
print(result.expected_value(z))  # ≈ 10.0
```

## Domain Modeling Pattern

Many real applications share a common shape:

- the system has **categorical scenario factors** outside the modeller's
  control (season, weather, day type, …);
- one or more quantities of interest have **distributions that depend on
  those factors** (e.g. visitor counts that depend on season and weather);
- the modeller wants to know, for every value of those quantities, the
  **probability that all domain constraints are satisfied** under the
  weighted population of scenarios.

The library supports this shape directly.  The categorical factors are
`CategoricalIndex` instances (called *context variables* — CVs); the
quantities of interest are `ConditionalDistributionIndex` instances (called
*presence variables* — PVs); `CrossProductEnsemble` enumerates the CV
combinations into weighted scenarios; `Evaluation.evaluate(parameters=…)`
sweeps each PV across a numerical axis.  The remaining ingredient — pairing
each output with an admissibility limit — is domain-specific and described
below.

The sections that follow walk through the four steps of the pattern,
using overtourism as the running example.  For an end-to-end walkthrough
with concrete CV/PV definitions and plotting code, see the
[overtourism getting-started guide](../../examples/overtourism_molveno/overtourism-getting-started.md).

### Step 1 — Constraints (domain-defined)

A *constraint* pairs a usage formula with a capacity limit; the system is
considered admissible when usage stays below capacity.  Each domain defines
its own `Constraint` dataclass inline — the core library provides no base
class, since the fields a domain needs (extra metadata, units, severity)
vary:

```python
from dataclasses import dataclass

from civic_digital_twins.dt_model import Index


@dataclass(eq=False)
class Constraint:
    name: str
    usage: Index     # formula-mode index — typically PV × per-CV usage factor
    capacity: Index  # constant or distribution-backed capacity
```

`eq=False` preserves identity-based `__hash__` so that `Constraint`
objects can be used as dictionary keys, matching the convention of
`graph.Node` and `GenericIndex`.

### Step 2 — Ensemble (CV cross-product)

`CrossProductEnsemble` enumerates the joint support of the model's
`CategoricalIndex` instances into weighted scenarios.  Each scenario also
includes one Monte-Carlo sample of every distribution-backed abstract index
that is *not* listed in `exclude` (PVs are excluded because they are swept
on the grid in step 3 instead):

```python
from civic_digital_twins.dt_model import CrossProductEnsemble

ensemble = CrossProductEnsemble(
    model,
    restrictions={CV_weather: ["good", "unsettled", "bad"]},
    max_categorical_size=20,
    exclude=model.pvs,
)
```

### Step 3 — Grid evaluation (PV sweep)

PVs are not resolved per-scenario; they define the **grid axes** along
which the sustainability field is computed.  `Evaluation` sweeps each PV
across the supplied numerical axis, returning result arrays with one
parameter axis per PV plus the trailing scenario axis `S`:

```python
import numpy as np

from civic_digital_twins.dt_model import Evaluation

tt = np.linspace(0, 50_000, 101)   # tourist presence axis
ee = np.linspace(0, 50_000, 101)   # excursionist presence axis

result = Evaluation(model).evaluate(
    ensemble=ensemble,
    parameters={PV_tourists: tt, PV_excursionists: ee},
)
# result[c.usage] has shape (tt.size, ee.size, S)
```

### Step 4 — Sustainability field

For each scenario and each grid point, every constraint produces a
*satisfaction mask* — `1.0` where usage ≤ capacity, `0.0` otherwise (or
the analytic survival probability when the capacity is itself a
distribution).  Marginalising the mask over the scenario axis with the
ensemble weights yields, per grid point, the **probability that the
constraint is satisfied** under the weighted scenario population.
Multiplying the per-constraint probabilities gives the joint
*sustainability field* over the grid:

```python
from civic_digital_twins.dt_model import Distribution

field = np.ones((tt.size, ee.size))
for c in model.constraints:
    usage = np.broadcast_to(result[c.usage], result.full_shape)
    if isinstance(c.capacity.value, Distribution):
        mask = 1.0 - c.capacity.value.cdf(usage)        # P(usage ≤ capacity)
    else:
        cap = np.broadcast_to(result[c.capacity], result.full_shape)
        mask = (usage <= cap).astype(float)
    field *= np.tensordot(mask, result.weights, axes=([-1], [0]))

# field[i, j] ∈ [0, 1] — joint sustainability score at (tt[i], ee[j])
```

The product across constraints assumes per-constraint independence given
the scenario; if a domain needs joint constraints, it can build a single
combined mask before marginalising instead.

## Design Rationale

### Why a single `Model` class?

Earlier versions had separate `AbstractModel` and `InstantiatedModel`
classes.  The distinction is now expressed through `abstract_indexes()`
and `is_instantiated()`, and the concrete-value binding is done at the
`Evaluation` call site via weighted scenarios.  This eliminates
mutation-based model instantiation and makes the data flow explicit.

The three-level access model (`inputs` / `outputs` / `expose`) was added
in v0.8.0 to make the inter-model data-flow contract explicit.  The core
`Model` class is extended by the dataclass path without breaking the
existing flat-list path.

### Why `ModelVariant` rather than subclassing?

A subclass would fix the implementation at class-definition time.
`ModelVariant` lets the same parent model choose among pre-constructed
instances at construction time — the selector is a plain string.  This
keeps variant switching visible at the call site and avoids deep
inheritance hierarchies.

### Why a structural `AxisEnsemble` Protocol?

Making `AxisEnsemble` a structural `Protocol` (rather than a base class)
means that any class exposing `ensemble_axes`, `ensemble_weights`, and
`assignments()` satisfies the contract without inheritance.
`DistributionEnsemble`, `PartitionedEnsemble`, and domain-specific
ensemble classes (e.g. `CrossProductEnsemble`) all work transparently.
The legacy `Iterable[WeightedScenario]` path is still supported via a
deprecation adapter.

### Why `GenericIndex.__hash__` is identity-based

`GenericIndex.__eq__` returns a `graph.Node` (lazy evaluation) rather
than `bool`.  This is intentional — it allows writing formulas such as
`graph.piecewise((expr, cv == "good"), …)`.  But it means `__hash__`
must not call `__eq__`; identity-based hashing is the standard Python
fallback and is exactly what `graph.Node` itself uses.

### Why domain `Constraint` uses `@dataclass(eq=False)`

The `@dataclass` decorator normally generates `__eq__` (and suppresses
`__hash__`).  Since `Constraint` objects are used as dict keys in the
domain field computation, `@dataclass(eq=False)` suppresses the
generated `__eq__` and preserves the identity-based `__hash__` inherited
from `object`.

## Appendix

### Glossary

**Abstract index**: an `Index` whose `value` is `None` or a
`Distribution`; it needs an external value before the model can be
evaluated.

**Concrete index**: an `Index` whose `value` is a scalar constant or a
`graph.Node` formula; it can be evaluated without external input.

**AxisEnsemble**: the canonical batched ensemble protocol.  Exposes named
ENSEMBLE `Axis` objects, per-axis weight vectors, and a batched
`assignments()` mapping.  `DistributionEnsemble` and `PartitionedEnsemble`
implement this protocol.

**Ensemble (legacy)**: an iterable of `WeightedScenario` tuples.  Still
accepted by `evaluate()` via a deprecation adapter; migrate to
`AxisEnsemble`.

**Expose**: optional inner `@dataclass` on a `Model` subclass that
holds inspectable but non-contractual intermediate indexes.  `Expose`
fields are intended for debugging and visualisation only; they MUST NOT
be used to wire indexes between models.

**Grid mode**: the `parameters=` keyword of `Evaluation.evaluate`; sweeps
PARAMETER indexes over a dense grid while the ensemble provides the
ENSEMBLE abstract index values.

**InputsContractWarning**: a `ModelContractWarning` emitted when a
`GenericIndex` constructor parameter is absent from the declared
`Inputs` dataclass; the warning message names the offending parameter.

**Inputs**: inner `@dataclass` on a `Model` subclass that declares the
model's contractual constructor inputs (i.e., the `GenericIndex`
instances passed in from outside).

**Instantiated model**: a model in which `is_instantiated()` returns
`True` — all indexes are concrete.

**Marginalize**: contract all ENSEMBLE axes by computing the weighted
average over each ENSEMBLE dimension using the factorized per-axis weights.
Result shape is `(*PARAMETER, *DOMAIN)`.

**ModelVariant**: a transparent proxy that selects among pre-constructed
`Model` instances sharing the same I/O contract (`Inputs` / `Outputs`
field names).  The active instance is chosen by a plain string
`selector` at construction time.

**Outputs**: inner `@dataclass` on a `Model` subclass that declares the
model's contractual outputs — the indexes that downstream models or the
evaluation layer may read.

**Domain modeling pattern**: the pattern of subclassing `Model` with
`Inputs` / `Outputs` dataclasses and composing core index types
(`CategoricalIndex`, `ConditionalDistributionIndex`, etc.) to add domain
semantics without modifying the core library.  See
[Domain Modeling Pattern](#domain-modeling-pattern).

**WeightedScenario** (deprecated): `tuple[float, dict[GenericIndex, Any]]` — a
probability weight paired with an assignment dict.  The legacy iterable
protocol is still accepted but emits `DeprecationWarning`; use `AxisEnsemble`
instead.
