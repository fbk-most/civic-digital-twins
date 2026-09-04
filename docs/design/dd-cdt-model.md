<!-- SPDX-License-Identifier: Apache-2.0 -->

# Model / Simulation Layer

|              | Document data                                  |
|--------------| ---------------------------------------------- |
| Author       | [@pistore](https://github.com/pistore)         |
| Last-Updated | 2026-07-24                                     |
| Status       | Draft                                          |
| Approved-By  | N/A                                            |

The [dt_model](../../civic_digital_twins/dt_model) package provides
a model/simulation layer built on top of the
[engine](dd-cdt-engine.md).  Where the engine deals with raw DAG nodes
and NumPy arrays, this layer offers named, typed index variables, a
uniform model abstraction, a what-if `Scenario` wrapper, and a generic
evaluation pipeline that wires batched ensembles to the engine.

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

**Scenario.** A *scenario* wraps a model (or model variant) with optional
*value overrides* that shadow the model's own index values — a concrete
scalar, array, distribution, or restricted categorical support, depending
on the index kind.  It is the canonical first argument to every ensemble
class and to `Evaluation`: `model → Scenario(model, overrides={…}) → ensemble + Evaluation(scenario)`.  Ensembles 
sample `scenario.abstract_indexes()`, not the model's own, so an override can turn
an abstract index concrete (removing it from sampling) or replace its
distribution/support.

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

**Evaluation.** `Evaluation(scenario).evaluate(ensemble=…, parameters=…)`
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
    └── ConstTimeseriesIndex
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
  `GenericIndex` objects* — use `any(idx is item for item in collection)`
  instead.

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

Passing another `Index` (or one of its subclasses, e.g. `ConstIndex`) as
`value` reuses its underlying `graph.Node` as a formula, exactly like
`Index("load", mu * cap)` above — `Index("y", x)` and `Index("y", x.node)`
are equivalent. Passing a differently-shaped `GenericIndex` sibling
instead — e.g. a `TimeseriesIndex`, whose values don't share `Index`'s
scalar shape — raises `TypeError`.

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

assert pv_load.is_abstract  # abstract — resolved per-scenario by the ensemble
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

assert cv_weekend.is_abstract  # abstract
```

Like `CategoricalIndex`, `cv_weekend == "yes"` returns a `graph.equal` node usable
in `graph.piecewise` guards.

### TimeseriesIndex

`TimeseriesIndex(name, values)` wraps a time-indexed quantity.

| `values` type | Mode | graph node created |
| ------------- | ---- | ------------------ |
| `np.ndarray` | fixed array | `graph.array_placeholder` over `axes=(TIME_AXIS,)`; the array is the default, seeded via `Index.concrete_default` and overridable by Scenario |
| `graph.Node` | formula | the node itself |
| `None` (default) | placeholder | `graph.array_placeholder` over `axes=(TIME_AXIS,)` |

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
        *,
        inputs:    Any | None = None,   # dataclass instance
        outputs:   Any | None = None,   # dataclass instance
        expose:    Any | None = None,   # dataclass instance
        functions: Any | None = None,   # @functions-decorated class instance
    ) -> None: ...
    def abstract_indexes(self) -> list[GenericIndex]: ...
    def is_instantiated(self) -> bool: ...
```

`Model` is a plain container.  It does not build the graph — that is
done by constructing `Index` and `TimeseriesIndex` objects beforehand.
The model merely collects them so that `Evaluation` and ensemble classes
can inspect which indexes are abstract.

`abstract_indexes()` returns indexes whose `is_abstract` property is `True`
(a bare placeholder, or a `DistributionIndex`).  All other indexes (constants
and formulas) are concrete and are not returned.

### Recommended API: `@define` + `compute()`

The `@define` decorator generates `__init__` from a `compute()` method — the
recommended way to build all leaf models.  Declare `Inputs`, `Outputs`, and
optionally `Expose` as inner classes decorated with `@inputs`, `@outputs`, and
`@expose`.  `model.indexes` is derived automatically from those inner classes — no
manual list needed.

**Three access levels** define the visibility contract:

1. `model.outputs.<field>` / `model.inputs.<field>` — **contractual,
   stable**: these are the primary wiring points between models and the
   evaluation layer.
2. `model.expose.<field>` — **inspectable, not contracted**: useful for
   debugging or visualisation, but `Expose` fields MUST NOT be used to
   wire indexes between models.
3. Local variables inside `compute()` — **internal, not accessible**
   outside the method.

**Inputs contract convention**: every `GenericIndex` that is an external input must
be declared as a field in `Inputs`.  If a `GenericIndex` value is used but absent
from `Inputs`, `InputsContractError` is raised naming the offending field (see
[Contract Violations](#contract-violations)).

```python
from scipy import stats

from civic_digital_twins.dt_model import DistributionIndex, Index, Model, define, inputs, outputs

@define("Demo")
class DemoModel(Model):

    @inputs
    class Inputs:
        x: DistributionIndex
        y: DistributionIndex

    @outputs
    class Outputs:
        z: Index

    def compute(self, inputs: Inputs) -> Outputs:
        z = Index("z", inputs.x + inputs.y)
        return DemoModel.Outputs(z=z)

x = DistributionIndex("x", stats.uniform, {"loc": 0.0, "scale": 10.0})
y = DistributionIndex("y", stats.uniform, {"loc": 0.0, "scale": 10.0})
m = DemoModel(inputs=DemoModel.Inputs(x=x, y=y))
print(m.abstract_indexes())   # [x, y]  — derived automatically
print(m.is_instantiated())    # False
```

This is enough to define a single leaf model. For advanced topics not
covered here — `@expose`, `@functions`, `default_inputs()`, and composite
("root") models that wire several sub-models together — see the full
`@define`/`compute()` guide in [dd-cdt-modularity.md](dd-cdt-modularity.md).

### Direct subclassing with `legacy=True`

For composite models that wire sub-models together, or any model that cannot be expressed
with `compute()`, define `__init__` directly and pass `legacy=True` to opt in.  A subclass
that defines `__init__` directly without `legacy=True` raises `TypeError` at class-definition
time; `legacy=True` itself is deprecated (emits a `DeprecationWarning`) and staged for
removal in a future milestone:

```python
class CompositeModel(Model, legacy=True):
    ...
```

See [dd-cdt-modularity.md § Worked Example: Bologna Mobility Model](dd-cdt-modularity.md#worked-example-bologna-mobility-model)
for a worked example.

Models can be subclassed to add domain-specific structure (labeled
subsets of indexes, constraint lists, etc.) while preserving the
core contract.

## ModelVariant

[`model/model_variant.py`](../../civic_digital_twins/dt_model/model/model_variant.py)

```python
class ModelVariant:
    def __init__(
        self,
        name: str,
        variants: Mapping[str, Model | ModelVariant],
        selector: str | CategoricalIndex | graph.Node,
    ) -> None: ...

    @staticmethod
    def guards_to_selector(
        guards: list[tuple[str, graph.Node | bool]],
    ) -> graph.Node: ...
```

`ModelVariant` selects among pre-constructed `Model` instances that share the same `outputs` field
names.  A variant may itself be a `ModelVariant`, enabling nested/recursive composition — see
[`dd-cdt-modularity.md`](dd-cdt-modularity.md) for that case.  It operates in two modes:

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

## Scenario

[`simulation/scenario.py`](../../civic_digital_twins/dt_model/simulation/scenario.py)

```python
class Scenario:
    def __init__(
        self,
        model: Model | ModelVariant,
        overrides: dict[GenericIndex, DomainValue] | None = None,
        parameter_axes: Iterable[GenericIndex] | None = None,
    ) -> None: ...
    def abstract_indexes(self) -> list[GenericIndex]: ...
    def effective_distribution(self, idx: GenericIndex) -> Distribution | None: ...
    def effective_outcomes(self, idx: CategoricalIndex | ConditionalCategoricalIndex) -> dict[str, float] | None: ...
```

A `Scenario` wraps a `Model` (or `ModelVariant`) and optionally carries
*value overrides* that shadow the model's own index values at evaluation
time.  It is the canonical first step for both ensembles and `Evaluation` —
the full chain is:

```text
model → Scenario(model, overrides={…}) → {DistributionEnsemble | CrossProductEnsemble
                                           | PartitionedEnsemble} + Evaluation(scenario)
```

`Scenario.abstract_indexes()` — not the model's own `abstract_indexes()` —
is what an ensemble actually samples: a model-abstract index (`None`-valued
or distribution-backed) that is *not* concretely overridden.  Indexes
declared in `parameter_axes=` are always excluded, since they are instead
swept via `parameters=` at evaluation time (see [Grid mode](#grid-mode));
`CrossProductEnsemble` skips them automatically and `evaluate()` raises
`ValueError` if a declared parameter axis is missing from `parameters=`.

**Override compatibility** — the accepted override type depends on the
index kind:

| Index kind | Accepted override |
| --- | --- |
| `Index` | `float` |
| `TimeseriesIndex` | 1-D `np.ndarray` |
| `DistributionIndex` | `Distribution` |
| `CategoricalIndex` | `str` (concrete pin), `dict[str, float]` (new weights, subset of support), or `list[str]` (subset of support — original model probabilities renormalised over it) |
| `ConditionalCategoricalIndex` | `str` (concrete pin) |
| `ConstIndex`, `ConstTimeseriesIndex`, `ConditionalDistributionIndex` | not overridable |

Overrides are matched by object identity, not name — always pass the same
index objects the model was built with.  See the `Scenario` class
docstring for the full per-method override-handling table.

```python
from civic_digital_twins.dt_model import Index, Model, Scenario, define, inputs, outputs

@define("Parking")
class ParkingModel(Model):

    @inputs
    class Inputs:
        cost: Index

    @outputs
    class Outputs:
        cost: Index

    def compute(self, inputs: Inputs) -> Outputs:
        return ParkingModel.Outputs(cost=inputs.cost)

cost = Index("cost", 8.0)
model = ParkingModel(inputs=ParkingModel.Inputs(cost=cost))
base = Scenario(model)                               # uses the model's own value
expensive = Scenario(model, overrides={cost: 12.0})   # what-if: cost = 12.0
```

## Contract Violations

[`model/model.py`](../../civic_digital_twins/dt_model/model/model.py)

All contract-violation classes are exported from `civic_digital_twins.dt_model`.

**`ModelContractViolation(Exception)`** — common base for any contract
violation, soft or hard.  It inherits from `Exception` (rather than being a
bare marker) solely so it is itself a valid `except` target; it is never
raised or emitted directly.  Catch it to handle any contract violation
regardless of severity:

```python
try:
    SomeModel(...)
except ModelContractViolation:
    ...
```

**`ModelContractWarning(ModelContractViolation, UserWarning)`** — base class
for *soft* Model I/O contract warnings, filterable via
`warnings.filterwarnings`.  It currently has no concrete members — both
`InputsContractError` and `AbstractIndexNotInInputsError` (below) are hard
errors — but remains the extension point for any future contract violation
that should stay soft rather than fatal.

**`ModelContractError(ModelContractViolation)`** — base class for all *hard*
Model I/O contract errors.  `ModelContractWarning` and `ModelContractError`
are siblings, not parent/child: a hard error is not a stricter kind of soft
warning, it is a different thing that happens to share a family lineage.

**`InputsContractError(ModelContractError)`** — raised when a
`GenericIndex` constructor parameter is absent from the declared
`Inputs` dataclass.  The message names the offending
parameter precisely so it can be located and added to `Inputs`.

**`AbstractIndexNotInInputsError(ModelContractError)`** — raised when an
abstract index (one whose value is `None` or a `Distribution`) is not
reachable via `self.inputs`.  Abstract indexes receive their values from
outside the model and are therefore inputs by definition.  

**`InputsTypeMismatchError(ModelContractError)`** — raised when the
`inputs` passed to a model's constructor is a valid `Inputs`-shaped
dataclass instance, but belongs to a *different* model — including a
same-shaped sibling that Python's structural typing can't tell apart.
Unlike the two errors above, this doesn't reflect a missing declaration;
it catches wiring mistakes such as passing `OtherModel.Inputs(...)` where
`ThisModel.Inputs(...)` was meant.

**`FunctionsTypeMismatchError(ModelContractError)`** — the same check for
`@functions`: raised when the `fns` value passed at construction is not
an instance of the model's own declared `Functions` class.

Both mismatch checks are always on and need no annotation — see
[Static checking with Pyright](#static-checking-with-pyright) below for
how they relate to what a type checker can catch ahead of time.

Example — the following model raises `InputsContractError`
because `x` is a `GenericIndex` constructor parameter but is not
declared in `Inputs`:

```python
from civic_digital_twins.dt_model import DistributionIndex, Index, Model, outputs

class BadModel(Model, legacy=True):

    @outputs
    class Outputs:
        z: Index

    def __init__(self, x: DistributionIndex) -> None:
        # x is a GenericIndex parameter but not in Inputs — raises!
        z = Index("z", x + x)
        super().__init__("bad", outputs=BadModel.Outputs(z=z))
```

Declare an `Inputs` dataclass and pass an instance to avoid the error:

```python
from civic_digital_twins.dt_model import DistributionIndex, Index, Model, inputs, outputs

class GoodModel(Model, legacy=True):

    @inputs
    class Inputs:
        x: DistributionIndex

    @outputs
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

### Static checking with Pyright

Everything above happens at *runtime* — the moment a model is constructed,
with no setup required. Pyright can catch some of the same mistakes
*before* the code even runs, but with a caveat: `@define` generates
`__init__` at runtime from `compute()`'s signature, so Pyright cannot see
it from the class's own annotations. To keep `@define` + `compute()`
free of extra typing boilerplate, `Model` instead exposes a permissive
constructor "floor" to the type checker: `DemoModel(inputs=...)`
type-checks by default, with no per-model annotation needed.

This floor already catches most mistakes statically — an unknown keyword
argument, or passing something that isn't a dataclass at all. The one
thing it *can't* tell apart is one model's `Inputs` from another's, since
both are equally valid dataclasses to the type checker. That's exactly
the gap `InputsTypeMismatchError`/`FunctionsTypeMismatchError` close at
runtime (above), so nothing is silently missed — the floor just trades
that one static check for zero authoring burden, in the spirit of
Python's gradual typing.

If a particular model's constructor is important enough to warrant a
compile-time guarantee too, add a `TYPE_CHECKING`-guarded `__init__`
matching `compute()`'s signature:

```python
from typing import TYPE_CHECKING

@define("Demo")
class DemoModel(Model):
    # ... Inputs / Outputs as before ...

    # Optional: opt in to full static constructor checking.
    if TYPE_CHECKING:
        def __init__(self, inputs: Inputs) -> None: ...

    # ... compute() as before ...
```

This block is invisible at runtime — `@define`'s generated `__init__`
still runs — and only informs Pyright, overriding the permissive floor
with the model's exact `Inputs` type. Add it where a compile-time
guarantee earns its keep; leave it off everywhere else and rely on the
runtime check.

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

### DistributionEnsemble

`DistributionEnsemble(scenario, size, rng=None)` is the standard ensemble
for models whose abstract indexes are all distribution-backed.  It draws
`size` independent samples from each distribution into a single ENSEMBLE
axis with uniform weights (`1/size` each).

```python
from civic_digital_twins.dt_model import DistributionEnsemble, Scenario

scenario = Scenario(model)
ensemble = DistributionEnsemble(scenario, size=100)
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

`PartitionedEnsemble(scenario, axes, default_axis=None, rng=None)` creates
N independent ENSEMBLE axes, each covering a disjoint subset of the
scenario's abstract indexes.  Each `EnsembleAxisSpec` names the axis and
lists the indexes it covers:

```python
from civic_digital_twins.dt_model import EnsembleAxisSpec, PartitionedEnsemble

ens = PartitionedEnsemble(
    scenario,
    axes=[
        EnsembleAxisSpec("demand", indexes=[demand_idx], size=50),
        EnsembleAxisSpec("capacity", indexes=[cap_idx], size=20),
    ],
)
# Result arrays have shape (*PARAMETER, 50, 20) — two independent ENSEMBLE dims.
```

### CrossProductEnsemble

`CrossProductEnsemble(scenario, max_categorical_size, n_samples_per_combo, rng)` implements
`AxisEnsemble`.  It materialises a batched ENSEMBLE axis by:

1. Discovering the scenario's abstract indexes via `scenario.abstract_indexes()`.
2. Enumerating all combinations of `CategoricalIndex` /
   `ConditionalCategoricalIndex` values; probability weights are the product
   of per-category outcome probabilities.  When a categorical's support
   exceeds `max_categorical_size`, values are Monte-Carlo sampled instead.
3. Pre-sampling `n_samples_per_combo` independent values per distribution-backed
   abstract index (e.g. stochastic capacities) for each categorical combination.
   Total ensemble size is `|categorical cross-product| × n_samples_per_combo`.
   The default `n_samples_per_combo=1` draws one sample per combination;
   increase it to reduce Monte Carlo variance when distribution-backed indexes
   are retained in the ensemble.

Indexes declared as `parameter_axes` on the `Scenario` are skipped in steps
2–3 — they are provided as PARAMETER axes at evaluation time instead.  See
[Domain Modeling Pattern](#domain-modeling-pattern) for a worked example.

```python
class CrossProductEnsemble:
    def __init__(
        self,
        scenario: Scenario,
        max_categorical_size: int = 20,
        n_samples_per_combo: int = 1,
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
    def __init__(self, scenario: Scenario) -> None: ...

    def evaluate(
        self,
        nodes_of_interest: list[GenericIndex] | None = None,
        *,
        parameters: dict[GenericIndex, Any] | None = None,
        parameter_axes: dict[str, np.ndarray] | None = None,
        ensemble: AxisEnsemble | None = None,
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
| `result.factorized_weights` | Per-ENSEMBLE-axis weight vectors, keyed by `Axis` (`result.weights` is their outer product). |
| `result.parameter_values` | The `parameters=` dict passed to `evaluate`. |
| `result.full_shape` | `(*PARAMETER, *ENSEMBLE)` sizes in axis-layout order. |
| `result.layout` | The `AxisLayout` mapping each `Axis` to its numpy dimension position and size. |

### End-to-End Example

```python
from scipy import stats

from civic_digital_twins.dt_model import (
    DistributionEnsemble, DistributionIndex, Evaluation, Index, Model, Scenario, define, inputs, outputs,
)

@define("Demo")
class DemoModel(Model):

    @inputs
    class Inputs:
        x: DistributionIndex
        y: DistributionIndex

    @outputs
    class Outputs:
        z: Index

    def compute(self, inputs: Inputs) -> Outputs:
        z = Index("z", inputs.x + inputs.y)
        return DemoModel.Outputs(z=z)

# Define the model
x = DistributionIndex("x", stats.uniform, {"loc": 0.0, "scale": 10.0})
y = DistributionIndex("y", stats.uniform, {"loc": 0.0, "scale": 10.0})
model = DemoModel(inputs=DemoModel.Inputs(x=x, y=y))
scenario = Scenario(model)

# Build an ensemble of 200 scenarios
ensemble = DistributionEnsemble(scenario, size=200)

# Evaluate
result = Evaluation(scenario).evaluate(ensemble=ensemble)

# Weighted mean of z across all scenarios
print(result.expected_value(model.outputs.z))  # ≈ 10.0
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

`CrossProductEnsemble` enumerates the joint support of a `Scenario`'s
`CategoricalIndex` instances into weighted scenarios.  Each categorical
combination also draws `n_samples_per_combo` independent Monte-Carlo samples
of every distribution-backed abstract index that is not a *parameter* index
(indexes declared as `parameter_axes` on the `Scenario` are swept on the grid
in step 3 instead).
The default `n_samples_per_combo=1` gives one sample per combination; increase
it to reduce variance when stochastic capacities are retained in the ensemble:

```python
from civic_digital_twins.dt_model import CrossProductEnsemble, Scenario

scenario = Scenario(
    model,
    overrides={CV_weather: ["good", "unsettled", "bad"]},
    parameter_axes=[PV_tourists, PV_excursionists],
)
ensemble = CrossProductEnsemble(
    scenario,
    max_categorical_size=20,
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

result = Evaluation(scenario).evaluate(
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
from civic_digital_twins.dt_model import DistributionIndex

field = np.ones((tt.size, ee.size))
for c in model.constraints:
    usage = np.broadcast_to(result[c.usage], result.full_shape)
    if isinstance(c.capacity, DistributionIndex):
        mask = 1.0 - c.capacity.frozen_distribution.cdf(usage)  # P(usage ≤ capacity)
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
in v0.8.0 to make the inter-model data-flow contract explicit.  The
original flat-list `indexes=` constructor API was removed in v0.11.0;
`inputs` / `outputs` / `expose` are now the sole way to declare a
model's indexes.

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

**AxisLayout**: maps each semantic `Axis` to its numpy dimension position
and size in a result array, enforcing the canonical
`(*PARAMETER, *ENSEMBLE, *DOMAIN)` dimension ordering.  Exposed as
`EvaluationResult.layout`; the single source of truth for axis layout
consumed by `Evaluation`, `EvaluationHandle`'s merge logic, and
`ModelEvaluator` resume serialization.

**Ensemble**: a structural `Protocol` for iterables that yield
`WeightedScenario` tuples.  Used as a common type for ensemble
generators; not directly accepted by `evaluate()`, which requires an
`AxisEnsemble`.

**Expose**: optional inner `@dataclass` on a `Model` subclass that
holds inspectable but non-contractual intermediate indexes.  `Expose`
fields are intended for debugging and visualisation only; they MUST NOT
be used to wire indexes between models.

**Grid mode**: the `parameters=` keyword of `Evaluation.evaluate`; sweeps
PARAMETER indexes over a dense grid while the ensemble provides the
ENSEMBLE abstract index values.

**InputsContractError**: a `ModelContractError` raised when a
`GenericIndex` constructor parameter is absent from the declared
`Inputs` dataclass; a hard error naming the offending parameter.

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

**Scenario**: a what-if wrapper around a `Model` (or `ModelVariant`)
carrying optional value overrides.  The canonical first argument to every
ensemble class and to `Evaluation`.  See [Scenario](#scenario).

**Domain modeling pattern**: the pattern of subclassing `Model` with
`Inputs` / `Outputs` dataclasses and composing core index types
(`CategoricalIndex`, `ConditionalDistributionIndex`, etc.) to add domain
semantics without modifying the core library.  See
[Domain Modeling Pattern](#domain-modeling-pattern).

**WeightedScenario**: `tuple[float, dict[GenericIndex, Any]]` — a
probability weight paired with an assignment dict.  Used by the `Ensemble`
protocol; not directly accepted by `evaluate()`, which requires an
`AxisEnsemble`.
