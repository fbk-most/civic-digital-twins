<!-- SPDX-License-Identifier: Apache-2.0 -->

# Model Modularity

|              | Document data                                  |
|--------------| ---------------------------------------------- |
| Author       | [@pistore](https://github.com/pistore)         |
| Last-Updated | 2026-06-07                                     |
| Status       | Draft                                          |
| Approved-By  | N/A                                            |

This guide explains how to decompose a `Model` into cooperating sub-models in the civic-digital-twins
framework.  It covers the `@define` pattern and contract decorators, the three-level access contract,
constructor wiring via `compute()`, `ModelVariant`, the two decomposition axes (pipeline stages vs.
independent concerns), and a full annotated walkthrough of the Bologna mobility example.

See [dd-cdt-model.md](dd-cdt-model.md) for the index and evaluation layer reference, including the
full `Model` API, `ModelVariant`, and the I/O contract.

---

## TL;DR

A `Model` subclass declares its public interface through three inner dataclasses — `Inputs`, `Outputs`,
and optionally `Expose` — annotated with `@inputs`, `@outputs`, and `@expose`.  The `@define`
decorator generates the constructor and calls the model's `compute()` method, which creates all
indexes and returns the outputs.  A **root model** wires sub-models together by constructing them
inside its own `compute()`, threading output indexes from one sub-model into the next.
`ModelVariant` lets the root choose among alternative implementations at construction time without
changing the downstream wiring.

```python
@define("Traffic")
class TrafficModel(Model):

    @inputs
    class Inputs:
        ts_inflow:         TimeseriesIndex
        ts_starting:       TimeseriesIndex
        modified_inflow:   Index
        modified_starting: Index

    @outputs
    class Outputs:
        traffic:                TimeseriesIndex
        modified_traffic:       TimeseriesIndex
        total_modified_traffic: Index
        inflow_ratio:           Index
        starting_ratio:         Index
        traffic_ratio:          Index

    def compute(self, inputs: Inputs) -> Outputs:
        traffic                = TimeseriesIndex("reference traffic", inputs.ts_inflow + inputs.ts_starting)
        modified_traffic       = TimeseriesIndex("modified traffic", inputs.modified_inflow + inputs.modified_starting)
        total_modified_traffic = Index("total modified traffic", modified_traffic.sum())
        inflow_ratio           = Index("inflow ratio", inputs.ts_inflow / inputs.modified_inflow)
        starting_ratio         = Index("starting ratio", inputs.ts_starting / inputs.modified_starting)
        traffic_ratio          = Index("traffic ratio", traffic / modified_traffic)
        return TrafficModel.Outputs(
            traffic=traffic,
            modified_traffic=modified_traffic,
            total_modified_traffic=total_modified_traffic,
            inflow_ratio=inflow_ratio,
            starting_ratio=starting_ratio,
            traffic_ratio=traffic_ratio,
        )
```

Construct the model by passing an `Inputs` instance:

```python
mod_in = Index("modified_inflow", 0.9)
mod_st = Index("modified_starting", 0.95)

m = TrafficModel(inputs=TrafficModel.Inputs(
    ts_inflow=ts_in,
    ts_starting=ts_st,
    modified_inflow=mod_in,
    modified_starting=mod_st,
))
```

The root model wires sub-models by constructing them inside `compute()` and threading Level-1 outputs
from one to the next:

```python
def compute(self, inputs: Inputs) -> tuple[Outputs, Expose]:
    _inflow = InflowModel(inputs=InflowModel.Inputs(
        ts_inflow=inputs.ts_inflow,
        ts_starting=inputs.ts_starting,
        ...
    ))
    _traffic = TrafficModel(inputs=TrafficModel.Inputs(
        ts_inflow=inputs.ts_inflow,
        ts_starting=inputs.ts_starting,
        modified_inflow=_inflow.outputs.modified_inflow,     # Level-1 wiring
        modified_starting=_inflow.outputs.modified_starting,
    ))
    _emissions = EmissionsModel(inputs=EmissionsModel.Inputs(
        traffic=_traffic.outputs.traffic,                    # Level-1 wiring
        modified_traffic=_traffic.outputs.modified_traffic,
        modified_euro_class_split=_inflow.outputs.modified_euro_class_split,
        ...
    ))
    ...
```

---

## Background

### Why decompose a model?

A monolithic `Model.compute()` that constructs every index in a single flat function becomes hard to
read, impossible to test in isolation, and brittle to extend — adding a new policy dimension means
touching hundreds of lines instead of a single sub-model boundary.

Decomposition solves three concrete problems:

1. **Readability.**  Each sub-model fits on one screen.  Its `Inputs` and `Outputs` dataclasses state
   the interface at the top, before any implementation.  A reader understands the contract without
   tracing through formula definitions.

2. **Testability.**  A sub-model is a plain Python object.  It can be constructed in isolation with
   stub indexes, and its outputs can be inspected directly —
   `assert traffic_model.outputs.traffic_ratio is not None`.

3. **Replaceability.**  `ModelVariant` lets a root model swap one sub-model implementation for another
   (e.g. `BikeModel` vs. `TrainModel`) without changing how upstream or downstream models are wired.

### What this guide assumes

You are already familiar with:

- `Index`, `TimeseriesIndex`, `DistributionIndex` — see [dd-cdt-model.md](dd-cdt-model.md)
- The `Evaluation` pipeline and `Ensemble` — see [dd-cdt-model.md](dd-cdt-model.md)
- The `Scenario` wrapper — see [dd-cdt-model.md](dd-cdt-model.md)

---

## `@define` and Contract Decorators

The framework provides five decorators that replace hand-written `__init__` boilerplate: `@inputs`,
`@outputs`, `@expose`, and `@functions` for inner dataclasses, and `@define` for the model class
itself.

### Inner dataclasses: `@inputs`, `@outputs`, and `@expose`

Annotate each inner class with the appropriate decorator instead of `@dataclass`:

| Decorator | Class name | Purpose |
|-----------|-----------|---------|
| `@inputs` | `Inputs` | Parameters received from outside; defines the public input interface |
| `@outputs` | `Outputs` | Computed results; the stable public output interface |
| `@expose` | `Expose` | Diagnostic intermediates — readable but not wireable |
| `@functions` | `Functions` | Typed functor injection (see below) |

Each decorator wraps `@dataclass` and additionally validates that every declared field holds a
`GenericIndex` instance (or a `list` / `dict` thereof).  An `InputsContractWarning` is emitted if
a `GenericIndex` is passed to the constructor but absent from the `Inputs` declaration.

### `@define` — generating `__init__` from `compute()`

`@define("Name")` generates an `__init__` that:

1. Accepts keyword argument `inputs: Inputs` (and optionally `fns: Functions`).
2. Calls `compute(inputs=inputs)` (forwarding `fns` if declared).
3. Passes the result of `compute()` to `super().__init__()`, wiring outputs and expose
   into the model automatically.

The `compute()` method is the factory: it creates index nodes from the inputs and returns an
`Outputs` instance (or a `(Outputs, Expose)` tuple when `@expose` is used):

```python
@define("ThreeLevel")
class ThreeLevelModel(Model):

    @inputs
    class Inputs:
        base: Index

    @outputs
    class Outputs:
        result: Index

    @expose
    class Expose:
        intermediate: Index

    def compute(self, inputs: Inputs) -> tuple[Outputs, Expose]:
        intermediate = Index("intermediate", inputs.base * 2)
        result = Index("result", intermediate + 1)
        return (
            ThreeLevelModel.Outputs(result=result),
            ThreeLevelModel.Expose(intermediate=intermediate),
        )
```

Construct the model by passing an `Inputs` instance — the `@define`-generated `__init__` handles
everything else:

```python
b = Index("base", 5.0)
m3 = ThreeLevelModel(inputs=ThreeLevelModel.Inputs(base=b))
```

Sub-models are wired inside `compute()` by constructing them and threading their Level-1 outputs:

```python
@define("Pipeline")
class PipelineModel(Model):

    @inputs
    class Inputs:
        raw_data: DistributionIndex

    @outputs
    class Outputs:
        result: Index

    def compute(self, inputs: Inputs) -> Outputs:
        stage_a = StageAModel(inputs=StageAModel.Inputs(raw_data=inputs.raw_data))
        stage_b = StageBModel(inputs=StageBModel.Inputs(
            processed=stage_a.outputs.processed,
            ratio=stage_a.outputs.ratio,
        ))
        return PipelineModel.Outputs(result=stage_b.outputs.result)
```

### `default_inputs()` class method

Models with many inputs often provide a `default_inputs()` class method that returns a pre-populated
`Inputs` instance for the reference scenario.  Callers can construct the model directly from defaults
or override individual fields:

```python
@define("Bike")
class BikeModel(Model):

    @inputs
    class Inputs:
        capacity: Index

    @outputs
    class Outputs:
        emissions: Index

    @classmethod
    def default_inputs(cls, capacity: float = 100.0) -> Inputs:
        return cls.Inputs(capacity=ConstIndex("bike_capacity", capacity))

    def compute(self, inputs: Inputs) -> Outputs:
        emissions = Index("bike_emissions", inputs.capacity * 3.0)
        return BikeModel.Outputs(emissions=emissions)
```

Models that also declare a `@functions` inner class may similarly provide a `default_fns()` class
method returning a pre-populated `Functions` instance — see the next section.

### `@functions` — typed functor injection

When a model depends on a pluggable algorithm (e.g. a timeseries solver), declare it in a
`@functions` inner class.  `@define` passes the `fns` object to `compute()` as a keyword argument:

```python
@define("Smoother")
class SmootherModel(Model):

    @inputs
    class Inputs:
        signal: TimeseriesIndex

    @functions
    class Functions:
        smooth: Functor

    @outputs
    class Outputs:
        smoothed: TimeseriesIndex

    def compute(self, inputs: Inputs, *, fns: Functions) -> Outputs:
        smoothed = TimeseriesIndex("smoothed", graph.function_call("smooth", inputs.signal))
        return SmootherModel.Outputs(smoothed=smoothed)

signal = TimeseriesIndex("signal", np.array([1.0, 2.0, 3.0, 2.0, 1.0]))
m = SmootherModel(
    inputs=SmootherModel.Inputs(signal=signal),
    fns=SmootherModel.Functions(smooth=NumpyBackend.adapt(lambda x: x)),
)
```

`graph.function_call("smooth", ...)` creates a graph node that the engine evaluates via the
registered functor.  `NumpyBackend.adapt(fn)` wraps a plain NumPy function as a `Functor`.

### `legacy=True` — opting out of `@define`

Models with complex construction logic that cannot be expressed in `compute()` can opt out by passing
`legacy=True` to the base class.  This suppresses the `DeprecationWarning` that `Model` emits when a
hand-written `__init__` is detected and no `@define` is present.

A `legacy=True` model **must still** declare `@inputs`, `@outputs`, and `@expose` inner classes and
construct them correctly; the `InputsContractWarning` mechanism still applies.  See "Inputs Contract
Convention" below:

```python
class GoodModel(Model, legacy=True):

    @inputs
    class Inputs:
        inflow: TimeseriesIndex      # declared here ...

    @outputs
    class Outputs:
        total: Index

    def __init__(self, inflow: TimeseriesIndex) -> None:
        Inputs = GoodModel.Inputs
        inputs_ = Inputs(inflow=inflow)  # ... and forwarded here
        total_idx = Index("total_good", inputs_.inflow.sum())
        super().__init__("Good", inputs=inputs_, outputs=GoodModel.Outputs(total=total_idx))
```

---

## Three-Level Access Model

Every `Model` instance exposes indexes at exactly three levels of visibility.  The levels are enforced
by convention and by the `InputsContractWarning` mechanism; they are not enforced by Python's access
control.

### Level 1 — Contractual (`inputs` and `outputs`)

`model.inputs.<field>` and `model.outputs.<field>` are the **stable public interface** of the model.
Parent models and callers may depend on these field names across library versions.  Renaming or removing
a field is a breaking change and must be flagged in `CHANGELOG.md`.

```python
traffic = TrafficModel(...)
ts      = traffic.outputs.traffic           # contractual output — stable
mod     = traffic.outputs.modified_traffic  # contractual output — stable
inp     = traffic.inputs.ts_inflow          # contractual input  — stable
```

`inputs` is equally contractual: a parent model that passes `ts_inflow` into a sub-model can verify
after construction that `sub.inputs.ts_inflow is ts_inflow`.

### Level 2 — Inspectable (`expose`)

`model.expose.<field>` surfaces intermediate indexes that are **useful for diagnostics, plotting, or
debugging but are not part of the stable interface**.  Callers *may* read them, but must **not** wire
them into sibling or parent models.  Field names and the set of exposed indexes may change between
versions without a breaking-change notice.

```python
inflow = InflowModel(...)

# Acceptable — diagnostic read
frac = inflow.expose.i_fraction_anticipating

# WRONG — do not wire expose fields into another model's constructor
bad = SomeModel(anticipating=inflow.expose.i_fraction_anticipating)  # forbidden
```

The rule is simple: `Expose` is for *reading*, never for *wiring*.

### Level 3 — Internal (local variables inside `compute()`)

Indexes bound only to local variables inside `compute()` are engine-internal.  They participate in the
computation graph (because other indexes reference their nodes) but are not accessible from outside
the method.  No naming convention is required — use whatever makes the implementation readable.

```python
def compute(self, inputs: Inputs) -> Outputs:
    ...
    # i_modified_average_emissions is a local — never promoted to Outputs or Expose.
    # The engine reaches it transitively via modified_emissions, which references it.
    i_modified_average_emissions = Index(
        "modified average emissions (per vehicle, per km)",
        euro_class_emission["euro_0"] * inputs.modified_euro_class_split[0] + ...,
    )
    modified_emissions = Index(
        "modified emissions",
        graph.piecewise(
            (2.5 * i_modified_average_emissions * inputs.modified_traffic, ...),
            (2.5 * average_emissions * inputs.modified_traffic, True),
        ),
    )
    ...
```

### Summary table

| Level | How to access | Stability | May wire into another model? |
|-------|---------------|-----------|-----------------------------|
| 1 | `model.outputs.<field>` / `model.inputs.<field>` | Contractual — stable across versions | Yes |
| 2 | `model.expose.<field>` | Inspectable — may change between versions | No |
| 3 | Local variables inside `compute()` | Internal — not accessible | — |

---

## Wiring Sub-Models via `compute()`

Sub-models receive their upstream dependencies as **typed constructor arguments** that are declared in
the `Inputs` dataclass.  The root model constructs sub-models inside its own `compute()`, threading
indexes from one to the next.

### Pattern

```python
@define("Pipeline")
class PipelineModel(Model):

    @inputs
    class Inputs:
        raw_data: DistributionIndex

    @outputs
    class Outputs:
        result: Index

    def compute(self, inputs: Inputs) -> Outputs:
        stage_a = StageAModel(inputs=StageAModel.Inputs(raw_data=inputs.raw_data))
        stage_b = StageBModel(inputs=StageBModel.Inputs(
            processed=stage_a.outputs.processed,
            ratio=stage_a.outputs.ratio,
        ))
        return PipelineModel.Outputs(result=stage_b.outputs.result)
```

### Key rules

1. **Construct sub-models as local variables inside `compute()`.**  Sub-model instances
   (`_inflow`, `_traffic`, …) live only inside the root's `compute()`.  They are not assigned to
   `self.*` and are not exposed directly — only their index *objects* are returned via `Outputs` or
   `Expose`.

2. **Wire outputs by name, not by position.**  Always use
   `stage_a.outputs.modified_inflow` rather than indexing into a flat list.  Named access is
   self-documenting and type-safe.

3. **The root's `outputs` hold references, not copies.**  `total_base_inflow=_inflow.outputs.total_base_inflow`
   stores a reference to the same `Index` object that lives inside `InflowModel`.  The evaluation
   engine operates on object identity, so no duplication or aliasing occurs.

4. **Declare all constructor-received indexes in `Inputs` for engine visibility.**  The engine
   traverses `model.indexes`, which is derived from `inputs`, `outputs`, and `expose`.  Every index
   received as a constructor argument — including abstract parameters such as `DistributionIndex` —
   must be declared in `Inputs` so the engine can reach it.  Root models follow the same rule as
   sub-models: all policy and behavioural parameters belong in `Inputs`.  See
   "Why declare all parameters in root `Inputs`?" in the [Design Rationale](#design-rationale) section.

---

## Inputs Contract Convention and `InputsContractWarning`

### The convention

For models using `@define`, the contract is enforced automatically: `@define` generates an `__init__`
that receives only an `Inputs` dataclass instance, guaranteeing that every index in scope has been
declared.

For `legacy=True` models with hand-written `__init__`, the convention must be followed manually:
every `GenericIndex` (or `list[GenericIndex]` / `dict[str, GenericIndex]`) passed into the
constructor as a parameter **must** be declared as a field of the `Inputs` dataclass and forwarded to
`super().__init__(inputs=Inputs(...))`.

```python
# CORRECT — every GenericIndex parameter is declared in Inputs and forwarded
class GoodModel(Model, legacy=True):

    @inputs
    class Inputs:
        inflow: TimeseriesIndex      # declared here ...

    @outputs
    class Outputs:
        total: Index

    def __init__(self, inflow: TimeseriesIndex) -> None:
        Inputs = GoodModel.Inputs
        inputs_ = Inputs(inflow=inflow)  # ... and forwarded here
        total_idx = Index("total_good", inputs_.inflow.sum())
        super().__init__("Good", inputs=inputs_, outputs=GoodModel.Outputs(total=total_idx))
```

```python
# INCORRECT — 'inflow' is received but absent from Inputs; InputsContractWarning fires
class BadModel(Model, legacy=True):

    @inputs
    class Inputs:
        pass   # inflow is missing

    def __init__(self, inflow: TimeseriesIndex) -> None:
        # InputsContractWarning fires here: 'inflow' holds a GenericIndex
        # that is not declared in Inputs.
        total = Index("total_bad", inflow.sum())
        super().__init__("Bad", inputs=BadModel.Inputs())
```

This rule exists because `Inputs` is the only place where the inter-model wiring contract is expressed
as inspectable metadata.  `ModelVariant`'s cross-variant consistency check reads `model.inputs` field
names — if an index is received but not declared in `Inputs`, the check is blind to it.

### `InputsContractWarning`

At construction time, `Model.__init__` inspects the calling frame and compares the constructor's
`GenericIndex` parameters against the declared `Inputs` fields.  Any undeclared index parameter
triggers an `InputsContractWarning`.

The warning is **soft** — it does not abort execution — so that existing models can be migrated
incrementally.  During development, escalate it to an error:

```python
import warnings

from civic_digital_twins.dt_model import InputsContractWarning, ModelContractWarning

with warnings.catch_warnings():
    # Escalate all contract warnings to errors (recommended for CI)
    warnings.filterwarnings("error", category=ModelContractWarning)

    # Or target only the inputs-specific warning
    warnings.filterwarnings("error", category=InputsContractWarning)
```

`InputsContractWarning` is a subclass of `ModelContractWarning`, so a single filter on the base class
catches all present and future contract-violation categories.

### What `Expose` fields are exempt from

Fields declared in `Expose` are intentionally exempt from this check.  `Expose` holds purely
internal intermediates — indexes created inside `compute()`, not received from the caller.  The
warning mechanism therefore never fires for them: an `Expose` index in the constructor's local
frame is known to be an output, not an undeclared input.  If `Expose` is misused so that an
index is passed in from outside, that is a model design error that code review should catch — the
warning mechanism does not cover it.

---

## `ModelVariant` — Switching Between Implementations

`ModelVariant` selects one `Model` instance from a named mapping at construction time and then acts as
a fully transparent proxy for the chosen (active) variant.

### Construction

```python
mv = ModelVariant(
    "TransportModel",
    variants={
        "bike":  BikeModel(inputs=BikeModel.default_inputs(100)),
        "train": TrainModel(inputs=TrainModel.default_inputs(500)),
    },
    selector="bike",
)
```

- `variants` is a mapping from `str` key to an **already-constructed** `Model` instance.  Each variant
  is fully built before `ModelVariant` is created; there is no lazy or deferred construction.
- `selector` is a plain string literal resolved once at construction time.  The active variant does not
  change after construction.
- `ModelVariant` raises `ValueError` immediately if `selector` is not a key in `variants`, if
  `variants` is empty, or if the `outputs` field names differ across variants.

### Transparent proxy

After construction, `mv` behaves as though it *is* the active variant:

```python
mv.outputs.emissions        # delegates to BikeModel.outputs.emissions
mv.inputs.capacity          # delegates to BikeModel.inputs.capacity
mv.indexes                  # index list of the active (BikeModel) variant only
mv.abstract_indexes()       # delegates to BikeModel.abstract_indexes()
mv.is_instantiated()        # delegates to BikeModel.is_instantiated()
```

Any attribute not defined directly on `ModelVariant` itself is forwarded to the active variant via
`__getattr__`, so a `ModelVariant` can be passed anywhere a plain `Model` is expected.

### Accessing inactive variants

Inactive variants' indexes are **not** reachable through `mv.indexes` or normal attribute access.
They are accessible only via explicit navigation:

```python
mv.variants["train"].outputs.emissions   # explicit — reaches inactive variant
mv.variants["train"].indexes             # index list of TrainModel only
```

### Interface contract

The `outputs` field *names* must be identical across all variants — this is what makes `ModelVariant`
a true drop-in replacement: downstream code that reads `mv.outputs.emissions` works regardless of
which variant is active.  `inputs` field names may differ across variants — `mv.inputs` delegates
to the active variant.

```python
# Both BikeModel and TrainModel must declare identically-named Outputs fields, e.g.:
#
#   class Outputs:
#       emissions:      Index   ← same name in both; different object
#       total_distance: Index   ← same name in both; different object
#
# A mismatch in Outputs field names raises ValueError at ModelVariant construction time.
```

### Runtime variant selection

`selector` can be a **`CategoricalIndex`** or a **`graph.Node`** to make the active variant
per-scenario rather than fixed for the entire run.  Because different scenarios may take different
branches, `mv.inputs` in runtime mode surfaces a union of all variants' inputs rather than
delegating to a single active variant.

#### `CategoricalIndex` selector — probabilistic, independent choice

A `CategoricalIndex` encodes a finite probability distribution over variant keys.
`DistributionEnsemble` samples it automatically: each scenario receives one variant key drawn
proportional to the declared weights.

```python
mode = CategoricalIndex("mode", {"bike": 0.3, "train": 0.7})

mv = ModelVariant(
    "TransportModel",
    variants={
        "bike":  BikeModel(inputs=BikeModel.default_inputs()),
        "train": TrainModel(inputs=TrainModel.default_inputs()),
    },
    selector=mode,
)
```

`CategoricalIndex` is added to the model's abstract indexes and must be assigned a value in every
scenario.  `DistributionEnsemble` handles this automatically.  At construction time `ModelVariant`
validates that every key in `mode.support` has a matching entry in `variants`; a `ValueError` is
raised immediately if any outcome key is unknown.

#### `graph.Node` selector — derived from model parameters

When the variant choice is a deterministic function of other model parameters, pass a `graph.Node`
directly.  Use `ModelVariant.guards_to_selector` to build one from a list of `(key, predicate)` pairs:

```python
from scipy import stats

from civic_digital_twins.dt_model import DistributionIndex, ModelVariant

cost_threshold = DistributionIndex("cost_threshold", stats.uniform, {"loc": 3.0, "scale": 8.0})

mv = ModelVariant(
    "TransportModel",
    variants={
        "bike":  BikeModel(inputs=BikeModel.default_inputs()),
        "train": TrainModel(inputs=TrainModel.default_inputs()),
        "metro": MetroModel(inputs=MetroModel.default_inputs()),
    },
    selector=ModelVariant.guards_to_selector([
        ("metro", (cost_threshold > 5.0) & (hour >= 8.0)),  # most-specific first
        ("train", cost_threshold > 5.0),
        ("bike",  True),                                    # fallback
    ]),
)
```

Guards are evaluated top-to-bottom (like `if / elif / else`): **place the most-specific condition
first**.  The last entry should use `True` as its predicate.  No new abstract index is introduced —
the variant selection emerges from existing sampled parameters.

| | `CategoricalIndex` | `graph.Node` |
|---|---|---|
| Variant choice | **independent** of model params | **derived** from model params |
| New abstract index | Yes | No |
| Sampled by | `DistributionEnsemble` (extended) | existing sampling pipeline |
| Typical use | "30 % bike / 70 % train" | "if cost > threshold → train" |

#### Deterministic variant sweep via `parameters=`

A `CategoricalIndex` selector can also be used as a **PARAMETER axis** to compare variants
side-by-side, without any probabilistic ensemble.  Pass it directly to `parameters=` with an
array of outcome strings, and wrap the model in `Scenario` before `Evaluation`:

```python
mode_param = CategoricalIndex("mode_param", {"bike": 0.5, "train": 0.5})
mv_param = ModelVariant(
    "TransportParam",
    variants={
        "bike":  BikeModel(inputs=BikeModel.default_inputs()),
        "train": TrainModel(inputs=TrainModel.default_inputs()),
    },
    selector=mode_param,
)

result = Evaluation(Scenario(mv_param)).evaluate(
    ensemble=None,
    parameters={mode_param: np.array(["bike", "train"])},
)
# result.expected_value(mv_param.outputs.emissions) → shape (2,)
# index 0 = bike emissions, index 1 = train emissions
```

`ensemble=None` signals a fully deterministic run: no ENSEMBLE axis is created and
`mode_param` is resolved by the PARAMETER grid alone.  A `CategoricalIndex` PARAMETER axis
can be combined with numeric PARAMETER axes for a 2-D grid.  Variant sub-models must
accept the numeric index as an abstract input:

```python
presence = Index("presence", None)  # abstract — swept by the grid
mv_grid = ModelVariant(
    "TransportGrid",
    variants={
        "bike":  BikeModelPres(inputs=BikeModelPres.Inputs(presence=presence)),
        "train": TrainModelPres(inputs=TrainModelPres.Inputs(presence=presence)),
    },
    selector=mode_param,
)

result = Evaluation(Scenario(mv_grid)).evaluate(
    ensemble=None,
    parameters={
        mode_param: np.array(["bike", "train"]),
        presence:   np.array([100.0, 200.0, 300.0]),
    },
)
# result.expected_value(mv_grid.outputs.emissions) → shape (2, 3)
# row 0 = bike emissions for each presence level
# row 1 = train emissions for each presence level
```

The variant ordering in the result follows the order of entries in the `parameters=` array,
not the declaration order of `variants`.

#### Runtime mode — what changes

In runtime mode `ModelVariant` builds a **merged computation graph** at construction time.
`mv.outputs.x` is always a real `Index` backed by a real graph node, usable in parent model
formulas regardless of which variant will be active per scenario.

| Property | Static mode | Runtime mode |
|---|---|---|
| `mv.inputs` | proxied from active variant | **union** of all variants' input fields |
| `mv.outputs.x` | proxied from active variant | `Index` backed by a merged graph node |
| `mv.expose` | proxied from active variant | **intersection** of field names across all variants |
| `mv.abstract_indexes()` | active variant only | **union** across all variants + selector (if `CategoricalIndex`) |
| `mv.indexes` | active variant only | union of all variants' indexes + selector + merged output indexes |
| `mv.is_instantiated()` | delegates to active | always `False` |

`mv._selector_index` is a thin `Index` wrapping the selector node.  After evaluation,
`result[mv._selector_index]` returns a `(S, 1)` string array of the active variant key per
scenario — useful for post-evaluation analysis.

#### `CategoricalIndex` as a formula guard (standalone)

`CategoricalIndex` is a first-class `Index` and can be used in any model formula, not only as a
`ModelVariant` selector:

```python
season = CategoricalIndex("season", {"summer": 0.25, "spring": 0.25, "autumn": 0.25, "winter": 0.25})

peak_factor = Index(
    "peak_factor",
    graph.piecewise(
        (1.8, season == "summer"),
        (1.2, season == "spring"),
        (1.0, season == "autumn"),
        (0.7, True),  # winter — default
    ),
)
```

`season == "summer"` produces a `graph.equal` node that the engine evaluates as a boolean mask
per scenario; this broadcasts correctly against scalar or timeseries formula branches.

---

## Decomposition Axes

Complex models can be decomposed along two orthogonal axes.  Most real models use a combination of
both.

### Pipeline stages

A pipeline decomposition reflects a **strict dependency order**: each stage takes the outputs of the
previous stage as inputs.  This is the natural structure when the computation graph has a clear
topological ordering at the domain level.

```python
#  StageA  →  StageB  →  StageC
#
# Each stage's constructor receives exactly what it needs from the previous stage.
# No stage knows about later stages.
```

The Bologna mobility model is a pure pipeline:

```
#  InflowModel  →  TrafficModel  →  EmissionsModel
#       ↘                ↘
#        BolognaModel (root)
```

`InflowModel` computes how the pricing policy modifies vehicle inflow and the per-euro-class split.
`TrafficModel` takes those modified flows and computes steady-state traffic for both the baseline and
modified scenarios.  `EmissionsModel` takes the traffic timeseries and euro-class split and computes
emission totals for both scenarios.

Each stage's `compute()` receives exactly what it needs from the previous stage and nothing more.  The
result is a chain of narrow, well-typed boundaries.

**When to use**: the computation graph has a clear left-to-right dependency; intermediate results from
one stage are the primary inputs to the next.

### Independent concerns

An independent-concerns decomposition reflects a domain that has multiple **parallel aspects** sharing
a common input base but not depending on each other.  The root model constructs each sub-model with
indexes from a shared pool and then merges their outputs into its own KPI set.

```python
#                   ┌─ ParkingModel       ─┐
#                   │                      │
#  RootModel ───────┼─ BeachModel         ─┼──→  KPI outputs
#                   │                      │
#                   └─ AccommodationModel ─┘
```

The Molveno overtourism model follows this pattern: `ParkingModel`, `BeachModel`,
`AccommodationModel`, and `FoodModel` all receive the same presence and context indexes from the root,
but none of them depends on the others' outputs.

**When to use**: each sub-model addresses a different aspect of the domain; there is no data flow
between them, only shared inputs flowing down from the root.

### Mixing axes

Most non-trivial models mix both axes.  A root model might first run a pipeline of transformation
stages and then fan out the results to independent concern sub-models.  The decomposition axes are
conceptual tools for reasoning about structure, not mutually exclusive choices.

---

## Worked Example: Bologna Mobility Model

The Bologna model is the canonical example of pipeline-stage decomposition.  This section walks
through every part of the implementation with annotations explaining each design choice.

The full source is in
[`examples/mobility_bologna/bologna_model.py`](../../examples/mobility_bologna/bologna_model.py).

### Overview

```
#  InflowModel  →  TrafficModel  →  EmissionsModel
#       ↘                ↘
#        BolognaModel (root)
#
# BolognaModel declares all policy (i_p_*) and behavioural (i_b_*) parameters
# in its own Inputs dataclass and passes them down to sub-models via compute().
# default_inputs() provides the reference-scenario values;
# default_fns() provides the reference timeseries solver.
```

The three sub-models and their roles:

| Sub-model | Inputs | Outputs | Notes |
|-----------|--------|---------|-------|
| `InflowModel` | 13 fields: raw timeseries, policy params, behavioural distributions | 11 fields: modified inflow/starting, payment stats, euro-class split | Abstract index: `i_b_p50_cost` (sampled by ensemble) |
| `TrafficModel` | 4 fields: raw timeseries + modified inflow/starting from `InflowModel` | 6 fields: baseline and modified traffic timeseries + ratios | No `Expose` — all intermediates are direct outputs |
| `EmissionsModel` | 6 fields: timeseries, policy window, traffic from `TrafficModel`, euro-class split from `InflowModel` | 5 fields: average emissions factor + baseline/modified totals | One internal local (`i_modified_average_emissions`) |

### `InflowModel` — policy-modified inflow

`InflowModel` takes 13 input indexes and produces 11 output indexes, plus diagnostic intermediates
in `Expose`.  The `@inputs`, `@outputs`, and `@expose` classes document the full interface at a glance:

```python
@define("Inflow")
class InflowModel(Model):

    @inputs
    class Inputs:
        ts_inflow:                    TimeseriesIndex
        ts_starting:                  TimeseriesIndex
        ts:                           TimeseriesIndex
        i_p_start_time:               Index
        i_p_end_time:                 Index
        i_p_cost:                     list[Index]          # one per euro class
        i_p_fraction_exempted:        Index
        i_b_p50_cost:                 DistributionIndex    # abstract — sampled by ensemble
        i_b_p50_anticipating:         Index
        i_b_p50_anticipation:         Index
        i_b_p50_postponing:           Index
        i_b_p50_postponement:         Index
        i_b_starting_modified_factor: Index

    @outputs
    class Outputs:
        modified_inflow:           Index
        modified_starting:         Index
        total_base_inflow:         Index
        total_modified_inflow:     Index
        fraction_rigid:            Index
        modified_euro_class_split: list[Index]   # consumed by EmissionsModel
        number_paying:             Index
        total_paying:              Index
        avg_cost:                  Index
        total_paid:                Index
        total_shifted:             Index

    @expose
    class Expose:                                # diagnostic intermediates only
        i_fraction_rigid_euro:   list[Index]
        i_delta_from_start:      TimeseriesIndex
        i_fraction_anticipating: TimeseriesIndex
        i_number_anticipating:   TimeseriesIndex
        ...
```

**Annotations:**

- `i_p_cost` and `modified_euro_class_split` are `list[Index]` fields — one entry per euro class
  (0–6).  The dataclass API supports `list` and `dict` field values natively; `IOProxy.__iter__`
  flattens them so the evaluation engine sees every scalar index.

- `i_b_p50_cost` is a `DistributionIndex`.  It is abstract — the ensemble samples it at evaluation
  time.  Declaring it in `Inputs` is correct because the root creates the distribution object and
  passes it down.  The sub-model does not own its own distributions; the root is responsible for all
  leaf index construction.

- `Expose` holds purely intermediate timeseries (anticipating/postponing behaviour, delta windows)
  that are useful for plotting but must not be wired into sibling models.

### `TrafficModel` — baseline and modified traffic

`TrafficModel` is the simplest of the three sub-models.  It receives four inputs — the raw timeseries
and the policy-modified versions from `InflowModel` — and computes steady-state traffic for both
scenarios together using a `ts_solve` functor injected through `@functions`:

```python
@define("Traffic")
class TrafficModel(Model):

    @inputs
    class Inputs:
        ts_inflow:         TimeseriesIndex
        ts_starting:       TimeseriesIndex
        modified_inflow:   Index             # ← from InflowModel.outputs
        modified_starting: Index             # ← from InflowModel.outputs

    @functions
    class Functions:
        ts_solve: Functor

    @outputs
    class Outputs:
        traffic:                TimeseriesIndex   # baseline steady-state
        modified_traffic:       TimeseriesIndex   # policy-modified steady-state
        total_modified_traffic: Index
        inflow_ratio:           Index
        starting_ratio:         Index
        traffic_ratio:          Index

    def compute(self, inputs: Inputs, *, fns: Functions) -> Outputs:
        traffic = TimeseriesIndex(
            "reference traffic",
            graph.function_call("ts_solve", inputs.ts_inflow + inputs.ts_starting),
        )
        modified_traffic = TimeseriesIndex(
            "modified traffic",
            graph.function_call("ts_solve", inputs.modified_inflow + inputs.modified_starting),
        )
        total_modified_traffic = Index("total modified traffic", modified_traffic.sum())
        ...
        return TrafficModel.Outputs(
            traffic=traffic,
            modified_traffic=modified_traffic,
            total_modified_traffic=total_modified_traffic,
            ...
        )
```

**Annotations:**

- `TrafficModel` has no `Expose` because there are no diagnostically interesting intermediates.  Every
  computed index feeds directly into an output.

- Both `traffic` (baseline) and `modified_traffic` (policy scenario) are co-located here because they
  share the same `ts_solve` computation structure.  Separating them across model boundaries would split
  a symmetric pair for no benefit — see "Why drop `BaseStateModel`?" in the
  [Design Rationale](#design-rationale) section.

- All intermediate computations access upstream indexes through `inputs.*`.  This is the required
  convention: once `inputs` is constructed, the parameters are no longer directly referenced.

### `EmissionsModel` — baseline and modified emissions

`EmissionsModel` receives the baseline and modified traffic timeseries, the per-euro-class distribution
from `InflowModel`, and the policy time window, and produces emission totals for both scenarios.

```python
@define("Emissions")
class EmissionsModel(Model):

    @inputs
    class Inputs:
        ts:                        TimeseriesIndex
        i_p_start_time:            Index
        i_p_end_time:              Index
        traffic:                   TimeseriesIndex   # ← from TrafficModel.outputs
        modified_traffic:          TimeseriesIndex   # ← from TrafficModel.outputs
        modified_euro_class_split: list[Index]       # ← from InflowModel.outputs

    @outputs
    class Outputs:
        average_emissions:        Index            # fleet-weighted baseline factor
        emissions:                TimeseriesIndex  # baseline timeseries
        modified_emissions:       Index            # policy-modified total
        total_emissions:          Index
        total_modified_emissions: Index

    def compute(self, inputs: Inputs) -> Outputs:
        average_emissions = Index(
            "average emissions (per vehicle, per km)",
            euro_class_emission["euro_0"] * euro_class_split["euro_0"] + ...,
        )

        # Pure local — not in Outputs or Expose.
        # The engine reaches it transitively via modified_emissions.
        i_modified_average_emissions = Index(
            "modified average emissions (per vehicle, per km)",
            euro_class_emission["euro_0"] * inputs.modified_euro_class_split[0] + ...,
        )

        emissions = TimeseriesIndex(
            "emissions",
            2.5 * average_emissions * inputs.traffic,
        )

        modified_emissions = Index(
            "modified emissions",
            graph.piecewise(
                (2.5 * i_modified_average_emissions * inputs.modified_traffic,
                 (inputs.ts >= inputs.i_p_start_time) & (inputs.ts <= inputs.i_p_end_time)),
                (2.5 * average_emissions * inputs.modified_traffic, True),
            ),
        )

        return EmissionsModel.Outputs(
            average_emissions=average_emissions,
            emissions=emissions,
            modified_emissions=modified_emissions,
            total_emissions=Index("total emissions", emissions.sum()),
            total_modified_emissions=Index("total modified emissions", modified_emissions.sum()),
        )
```

**Annotations:**

- `modified_euro_class_split` is a `list[Index]` that originates in `InflowModel` and is threaded
  directly into `EmissionsModel` by the root.  `TrafficModel` never sees it — sub-models depend only
  on what they need.

- `i_modified_average_emissions` is a Level-3 local variable.  It appears in the computation graph
  because `modified_emissions` holds a formula node that references it.  The engine traverses from
  `modified_emissions` to it automatically; there is no need to name it in any proxy field.

### `BolognaModel` — root wiring

`BolognaModel` follows the same `Inputs` pattern as its sub-models.  All policy (`i_p_*`) and
behavioural (`i_b_*`) parameters are declared in `Inputs` and received through `@define`'s generated
constructor.  Default values for the reference scenario are provided by the `default_inputs()` and
`default_fns()` class methods.

```python
@define("Bologna mobility")
class BolognaModel(Model):

    @inputs
    class Inputs:
        # Policy parameters
        i_p_start_time:              Index
        i_p_end_time:                Index
        i_p_cost:                    list[Index]
        i_p_fraction_exempted:       Index
        # Behavioural parameters
        i_b_p50_cost:                DistributionIndex
        i_b_p50_anticipating:        Index
        i_b_p50_anticipation:        Index
        i_b_p50_postponing:          Index
        i_b_p50_postponement:        Index
        i_b_starting_modified_factor: Index

    @functions
    class Functions:
        ts_solve: Functor

    @outputs
    class Outputs:
        total_base_inflow:        Index
        total_modified_inflow:    Index
        total_shifted:            Index
        total_paying:             Index
        avg_cost:                 Index
        total_paid:               Index
        total_emissions:          Index
        total_modified_emissions: Index

    @expose
    class Expose:
        # Timeseries surfaced for plotting helpers
        ts_inflow:          TimeseriesIndex
        modified_inflow:    Index
        traffic:            TimeseriesIndex
        modified_traffic:   TimeseriesIndex
        emissions:          TimeseriesIndex
        modified_emissions: Index

    @classmethod
    def default_inputs(cls) -> Inputs:
        """Reference-scenario parameters."""
        return cls.Inputs(
            i_p_start_time=Index("start time", ...),
            i_p_end_time=Index("end time", ...),
            i_p_cost=[Index(f"cost euro {e}", 5.00 - e * 0.25) for e in range(7)],
            i_p_fraction_exempted=Index("exempted vehicles %", 0.15),
            i_b_p50_cost=DistributionIndex("cost 50% threshold", stats.uniform, {...}),
            ...
        )

    @classmethod
    def default_fns(cls) -> Functions:
        """Reference solver."""
        return cls.Functions(ts_solve=NumpyBackend.adapt(...))

    def compute(self, inputs: Inputs, *, fns: Functions) -> tuple[Outputs, Expose]:
        # ── Internal timeseries (Level 3) ──────────────────────────────────────
        ts          = TimeseriesIndex("time range", np.array([...]))
        ts_inflow   = TimeseriesIndex("inflow", vehicle_inflow)
        ts_starting = TimeseriesIndex("starting", vehicle_starting)

        # ── Sub-models in pipeline order ──────────────────────────────────────
        _inflow = InflowModel(inputs=InflowModel.Inputs(
            ts_inflow=ts_inflow,
            ts_starting=ts_starting,
            ts=ts,
            i_p_start_time=inputs.i_p_start_time,
            ...
        ))

        _traffic = TrafficModel(
            inputs=TrafficModel.Inputs(
                ts_inflow=ts_inflow,
                ts_starting=ts_starting,
                modified_inflow=_inflow.outputs.modified_inflow,      # ← Level-1 wiring
                modified_starting=_inflow.outputs.modified_starting,
            ),
            fns=TrafficModel.Functions(ts_solve=fns.ts_solve),
        )

        _emissions = EmissionsModel(inputs=EmissionsModel.Inputs(
            ts=ts,
            i_p_start_time=inputs.i_p_start_time,
            i_p_end_time=inputs.i_p_end_time,
            traffic=_traffic.outputs.traffic,                     # ← Level-1 wiring
            modified_traffic=_traffic.outputs.modified_traffic,
            modified_euro_class_split=_inflow.outputs.modified_euro_class_split,
        ))

        return (
            BolognaModel.Outputs(
                total_base_inflow=_inflow.outputs.total_base_inflow,
                ...
                total_modified_emissions=_emissions.outputs.total_modified_emissions,
            ),
            BolognaModel.Expose(
                ts_inflow=ts_inflow,
                modified_inflow=_inflow.outputs.modified_inflow,
                traffic=_traffic.outputs.traffic,
                modified_traffic=_traffic.outputs.modified_traffic,
                emissions=_emissions.outputs.emissions,
                modified_emissions=_emissions.outputs.modified_emissions,
            ),
        )
```

**Annotation — `Inputs` ensures engine reachability:**

`BolognaModel.indexes` is derived by deduplicating all scalars from `inputs`, `outputs`, and `expose`.
Declaring all policy and behavioural parameters in `Inputs` — including the abstract `i_b_p50_cost`
`DistributionIndex` — guarantees they appear in `model.indexes` and are therefore reachable by the
engine.  `Outputs` covers the 8 KPI scalars; `Expose` covers the plotting timeseries.

**Annotation — `outputs` stores references to sub-model index objects:**

`total_base_inflow=_inflow.outputs.total_base_inflow` stores a reference to the same `Index` object
that lives inside `InflowModel`.  The evaluation engine operates on object identity, so no duplication
or aliasing occurs.  The `BolognaModel` does not own these indexes; it is a wiring hub.

### Using `BolognaModel`

```python
# Reference scenario — use built-in defaults
m = BolognaModel(inputs=BolognaModel.default_inputs(), fns=BolognaModel.default_fns())

# Alternative scenario — override one parameter via dataclasses.replace
m_strict = BolognaModel(
    inputs=dataclasses.replace(
        BolognaModel.default_inputs(),
        i_p_cost=[Index(f"cost euro {e}", 8.00 - e * 0.50) for e in range(7)],
    ),
    fns=BolognaModel.default_fns(),
)

scenario = Scenario(m)
ensemble = DistributionEnsemble(scenario, size=500)
result   = Evaluation(scenario).evaluate(ensemble=ensemble)

# Read KPI outputs by name
total_inflow_modified = result.expected_value(m.outputs.total_modified_inflow)
total_emissions       = result.expected_value(m.outputs.total_emissions)

# Access raw timeseries through expose
# modified_inflow depends on stochastic inputs → one timeseries per Monte Carlo sample
modified_inflow_ts = result[m.expose.modified_inflow]    # shape (S, T): S samples × T time-steps
# ts_inflow is a ConstTimeseriesIndex (no stochastic dependency) → single timeseries
reference_inflow   = result[m.expose.ts_inflow]          # shape (T,):  no sample axis
```

The evaluation layer is unaware of sub-models.  It sees a flat `m.indexes` list, resolves the graph,
and evaluates it.  The sub-model structure is a pure construction-time concern; it has zero runtime
overhead.

---

## End-to-End Evaluation

Wrap any model in `Scenario` before passing to `Evaluation`.  Use `DistributionEnsemble` for
probabilistic evaluation:

```python
m = BolognaModel(inputs=BolognaModel.default_inputs(), fns=BolognaModel.default_fns())
scenario = Scenario(m)
ensemble = DistributionEnsemble(scenario, size=500)
result   = Evaluation(scenario).evaluate(ensemble=ensemble)

total_inflow_modified = result.expected_value(m.outputs.total_modified_inflow)
total_emissions       = result.expected_value(m.outputs.total_emissions)
```

---

## API Reference

### `@define`

```python
# === illustrative ===
def define(name: str) -> Callable[[type[Model]], type[Model]]:
    """Class decorator.  Generates __init__(self, inputs: Inputs, *, fns: Functions | None).
    Calls compute() and wires result into super().__init__().
    """
    ...
```

The generated `__init__` signature matches the declared `Inputs` (and `Functions` if present).
`legacy=True` on the base class suppresses the `DeprecationWarning` for hand-written `__init__`
methods.

### `@inputs`, `@outputs`, `@expose`, `@functions`

```python
# === illustrative ===
def inputs(cls: type) -> type:
    """Wraps @dataclass and validates that every field holds a GenericIndex."""
    ...

def outputs(cls: type) -> type: ...
def expose(cls: type) -> type: ...
def functions(cls: type) -> type:
    """Wraps @dataclass and validates that every field holds a Functor."""
    ...
```

Each decorator wraps `@dataclass` internally and adds field-type validation.  An
`InputsContractWarning` is emitted when a `GenericIndex` is present in the constructor scope but
absent from the `Inputs` declaration (applies to `legacy=True` models).

### `Model`

```python
class Model:
    name:    str
    indexes: list[GenericIndex]
    inputs:  IOProxy[Inputs]
    outputs: IOProxy[Outputs]
    expose:  IOProxy[Expose]

    def __init__(
        self,
        name: str,
        indexes: list[GenericIndex] | None = None,  # deprecated
        *,
        inputs:  Any | None = None,
        outputs: Any | None = None,
        expose:  Any | None = None,
    ) -> None: ...

    def abstract_indexes(self) -> list[GenericIndex]: ...
    def is_instantiated(self) -> bool: ...
```

**Constructor parameters**

| Parameter | Type | Description |
|-----------|------|-------------|
| `name` | `str` | Human-readable name.  Used in `repr` and error messages. |
| `indexes` | `list[GenericIndex]` | *Deprecated.*  Explicit flat index list.  Emits `DeprecationWarning`.  Omit when using the dataclass API. |
| `inputs` | dataclass instance | Instance of the `Inputs` inner dataclass. |
| `outputs` | dataclass instance | Instance of the `Outputs` inner dataclass. |
| `expose` | dataclass instance | Instance of the `Expose` inner dataclass.  Optional. |

**`abstract_indexes() -> list[GenericIndex]`**

Returns all indexes whose `value` is `None` (explicit placeholder) or a `Distribution` (needs
sampling).  Constant and formula-based indexes are concrete and are not returned.  Used by
`DistributionEnsemble` and `Evaluation` to determine which indexes must be supplied by the ensemble.

**`is_instantiated() -> bool`**

Returns `True` when `abstract_indexes()` is empty — all indexes have concrete, evaluable values.

**`indexes`**

Derived automatically when using the dataclass API.  `_collect_indexes` iterates over all scalar
`GenericIndex` values in `inputs`, `outputs`, and `expose` (in that order) and deduplicates by object
identity (first-seen wins).  The result is a flat `list[GenericIndex]` in declaration order.

---

### `ModelVariant`

```python
class ModelVariant:
    name:     str
    variants: dict[str, Model]

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

    # Read-only properties — behaviour differs by mode (see tables below)
    @property
    def inputs(self)  -> IOProxy[Any]: ...
    @property
    def outputs(self) -> IOProxy[Any]: ...
    @property
    def expose(self)  -> IOProxy[Any]: ...
    @property
    def indexes(self) -> list[GenericIndex]: ...

    def abstract_indexes(self) -> list[GenericIndex]: ...
    def is_instantiated(self)  -> bool: ...

    # Fall-through: any other attribute is forwarded to the active variant (static mode only)
    def __getattr__(self, name: str) -> Any: ...
```

**Constructor parameters**

| Parameter | Type | Description |
|-----------|------|-------------|
| `name` | `str` | Human-readable name for the variant group. |
| `variants` | `Mapping[str, Model]` | Non-empty mapping from string key to constructed `Model` instance. |
| `selector` | `str` | *(Static mode)* Key of the variant to activate.  Resolved once at construction time. |
| `selector` | `CategoricalIndex` | *(Runtime mode)* Probabilistic selector; sampled per scenario by `DistributionEnsemble`. |
| `selector` | `graph.Node` | *(Runtime mode)* Derived selector; must produce a string matching a variant key per scenario. |

**Raises at construction**

| Exception | When |
|-----------|------|
| `ValueError` | `variants` is empty. |
| `ValueError` | *(static)* `selector` string is not a key in `variants`. |
| `ValueError` | *(runtime, `CategoricalIndex`)* any outcome key in `selector.support` is not in `variants`. |
| `ValueError` | `outputs` field names differ across variants. |

**Instance attributes**

| Attribute | Type | Description |
|-----------|------|-------------|
| `name` | `str` | Name passed at construction. |
| `variants` | `dict[str, Model]` | Full mapping of all variants (active and inactive). |

**Proxy attributes — static mode** (delegate to the active variant)

| Attribute / Method | Delegates to |
|--------------------|--------------|
| `inputs` | `active.inputs` |
| `outputs` | `active.outputs` |
| `expose` | `active.expose` |
| `indexes` | `active.indexes` |
| `abstract_indexes()` | `active.abstract_indexes()` |
| `is_instantiated()` | always `False` |
| Any other attribute | `getattr(active, name)` |

**Proxy attributes — runtime mode**

| Attribute / Method | Returns |
|--------------------|---------|
| `inputs` | `IOProxy` over the **union** of all variants' input fields (first-seen wins on name collision) |
| `outputs` | `IOProxy` where each field is an `Index` backed by a merged `exclusive_multi_clause_where` graph node |
| `expose` | `IOProxy` over the **intersection** of field names present in all variants |
| `indexes` | deduplicated union of all variants' `indexes` + selector (if `CategoricalIndex`) + merged output indexes |
| `abstract_indexes()` | union of all variants' `abstract_indexes()` + selector (if `CategoricalIndex`) |
| `is_instantiated()` | always `False` |
| `_selector_index` | thin `Index` wrapping the selector node; `result[mv._selector_index]` → `(S, 1)` variant-key string array |

**`guards_to_selector(guards)`**

Convenience static method that wraps `graph.piecewise` to build a string-valued selector node from
a list of `(key, predicate)` pairs.  Guards are evaluated top-to-bottom; the last entry should use
`True` as its predicate (unconditional fallback).  Place the most-specific condition first.

---

### `CategoricalIndex`

```python
class CategoricalIndex(Index):

    def __init__(self, name: str, outcomes: dict[str, float]) -> None: ...

    @property
    def support(self) -> list[str]: ...

    def sample(self, rng: np.random.Generator | None = None) -> str: ...
```

A placeholder `Index` whose per-scenario values are strings drawn from a finite set of named
outcomes.  Extends `Index` with `value=None`, so it is automatically identified as abstract by
`Model.abstract_indexes()` and must be assigned a concrete string value in every scenario.
`DistributionEnsemble` handles this automatically when a model contains a `CategoricalIndex`.

**Constructor parameters**

| Parameter | Type | Description |
|-----------|------|-------------|
| `name` | `str` | Human-readable name. |
| `outcomes` | `dict[str, float]` | Maps outcome key to probability.  All values must be positive and sum to 1.0 (validated at construction). |

**Raises at construction**

| Exception | When |
|-----------|------|
| `ValueError` | `outcomes` is empty. |
| `ValueError` | Any probability is ≤ 0, or the values do not sum to 1.0 within tolerance. |

**Methods**

| Method | Returns | Description |
|--------|---------|-------------|
| `support` | `list[str]` | Ordered list of outcome keys. |
| `sample(rng)` | `str` | Draw one key proportional to outcome probabilities. |

Because `CategoricalIndex` inherits the full `GenericIndex` algebra protocol, comparison operators
produce valid boolean-per-scenario values usable in `graph.piecewise` or any formula.

---

### `IOProxy`

```python
class IOProxy(Generic[_DC]):

    # Attribute access — returns scalar, list[GenericIndex], or dict[str, GenericIndex]
    def __getattr__(self, name: str) -> Any: ...
    def __setattr__(self, name: str, value: Any) -> None: ...  # always raises AttributeError

    # Iteration / sizing / containment — operate on flattened scalars
    def __iter__(self)            -> Iterator[GenericIndex]: ...
    def __len__(self)             -> int: ...
    def __contains__(self, item)  -> bool: ...  # identity-based

    def __repr__(self)            -> str: ...   # lists declared field names
```

`IOProxy` is a **read-only**, ordered, attribute-access proxy over a dataclass-declared set of index
fields.  It is generic over the dataclass type `_DC`; when built from a dataclass instance, field
access returns `Any`, which allows the declared field type on the dataclass to flow through at the
call site without requiring `cast()`.

**Field values**

Each registered slot holds one of:

- A single `GenericIndex` (scalar)
- A `list[GenericIndex]`
- A `dict[str, GenericIndex]`

`proxy.field` returns the raw value as declared.  Iteration, `len()`, and `in` **flatten** list and
dict values and operate on scalar indexes only.

**`__contains__`**

Uses identity comparison (`is`), not equality.  This is consistent with `GenericIndex.__hash__`,
which is also identity-based (because `GenericIndex.__eq__` returns a `graph.Node` rather than
`bool` to support lazy formula composition).

**`__iter__`**

Yields scalar indexes in declaration order.  List fields are yielded element-by-element; dict fields
are yielded in `.values()` order (insertion order in Python 3.7+).

---

### Warning classes

```python
class ModelContractWarning(UserWarning):
    """Base class for all Model I/O contract warnings."""

class InputsContractWarning(ModelContractWarning):
    """Emitted when a constructor parameter holds a GenericIndex not declared in Inputs."""

class AbstractIndexNotInInputsWarning(ModelContractWarning):
    """Emitted when an abstract index is not reachable via the model's Inputs."""
```

All three are subclasses of `UserWarning`.  Both concrete warnings are additionally subclasses of
`ModelContractWarning`, so a single filter on the base class covers all contract-violation
categories:

```python
import warnings

from civic_digital_twins.dt_model import InputsContractWarning, ModelContractWarning

with warnings.catch_warnings():
    # Recommended for CI — escalate all contract warnings to errors
    warnings.filterwarnings("error", category=ModelContractWarning)

    # Fine-grained — only escalate the inputs-specific warning
    warnings.filterwarnings("error", category=InputsContractWarning)
```

---

## Design Rationale

### Why `@define` generates `__init__` from `compute()`

The manual pattern — writing `__init__`, constructing `Inputs`, calling all sub-models, and calling
`super().__init__()` — is repetitive and error-prone.  `@define` automates this boilerplate while
enforcing the contract: because the generated `__init__` accepts only an `Inputs` dataclass instance,
every index in scope is guaranteed to be declared.  There is no mechanism by which an undeclared
index can slip through.

`compute()` is the factory method: it creates all indexes and returns the outputs.  This is a clean
separation of concerns — the framework handles construction protocol, the model handles domain logic.

### Why constructor arguments rather than a separate `wire()` step?

Wiring via `Inputs` and `compute()` gives a typed, IDE-navigable, one-shot configuration.  There is
no mutable state to reason about — once `compute()` returns, all indexes are fully wired and the
model is immutable.  A separate `wire()` step would require the model to hold partially-constructed
state, complicating `is_instantiated()` and making order-of-calls errors possible.

### Why `Expose` must not be wired

`Expose` is the boundary between the contractual (Level 1) and internal (Level 3) surfaces.  If a
parent model wires `child.expose.foo` into a sibling, it is depending on an unstable interface.
Keeping the rule simple — never wire `expose` fields — makes it easy to audit: a `grep expose\.`
across wiring code should return zero results.

### Why declare all parameters in root `Inputs`?

The evaluation engine traverses `model.indexes`, which is derived from `inputs`, `outputs`, and
`expose` only.  Abstract parameters — such as a `DistributionIndex` deep inside a sub-model — are
invisible to the engine unless they appear in one of those three surfaces of the root model.

The idiomatic solution is to declare all policy and behavioural parameters in the root's `Inputs`
dataclass and receive them as constructor arguments, exactly as sub-models do.  This makes abstract
parameters reachable through `model.inputs` without any special scaffolding, and it is semantically
correct: parameters whose values come from outside the model are inputs by definition.

An earlier approach placed `list(sub_model.indexes)` into named fields of the root's `Expose` to
achieve the same reachability.  That approach worked mechanically but mixed concerns: `Expose` is
meant for diagnostic timeseries, not for parameter surfacing.  It also prevented
`InputsContractWarning` from firing on the parameters that were absent from `Inputs`, silently
weakening the contract check.  Declaring parameters in `Inputs` is clearer and consistent with the
three-level access model.

### Why drop `BaseStateModel`?

Earlier versions of the Bologna decomposition had four sub-models:
`BaseStateModel → ModifiedInflowModel → ModifiedTrafficModel → ModifiedEmissionsModel`.
`BaseStateModel` existed solely to compute `traffic` (baseline) and `average_emissions`.

Once the dataclass API removed the need for a "wiring proxy" object, `BaseStateModel`'s only function
was to group two indexes together.  Both computations belong naturally in the models that use them:

- Baseline `traffic` is co-located with `modified_traffic` in `TrafficModel` — both call `ts_solve`,
  sharing the same iterative computation structure.  Separating them would split a symmetric pair
  across model boundaries for no benefit.

- `average_emissions` is co-located with `i_modified_average_emissions` in `EmissionsModel` — both
  are fleet-average emission-factor aggregations over the euro-class distribution, and the modified
  version is derived from the baseline.

The result is three sub-models with clean, symmetric interfaces:

```python
# Before: BaseStateModel → ModifiedInflowModel → ModifiedTrafficModel → ModifiedEmissionsModel
# After:  InflowModel    → TrafficModel         → EmissionsModel
#
# Same computation; three cohesive units instead of four.
```

### Why does `ModelVariant` have both a static and a runtime selector mode?

Static mode (`selector: str`) is a zero-overhead proxy: the inactive variants do not appear in the
computation graph at all.  This covers the common case of choosing between implementations before a
run (e.g. different transport assumptions for different cities).

Runtime mode (`selector: CategoricalIndex | graph.Node`) builds a full merged graph at construction
time so that `mv.outputs.x` is always a real `Index` node that can be wired into parent model
formulas.  At evaluation time the engine uses a stratified split-dispatch-merge path that evaluates
each variant in isolation with only its own scenario slice — zero wasted computation.  The two modes
are deliberately separate code paths in v0.8.x.  Post-0.8.x, when the engine gains constant-folding,
the static case could become an optimised degenerate case of the runtime representation.

### Why are only `outputs` field names required to match across variants?

Only `outputs` names must be identical: the merge graph (runtime mode) and the transparent proxy
(static mode) both expose `mv.outputs.x` to downstream code, so a name mismatch would break that
access for one variant.  `inputs` names are free to differ — in static mode `mv.inputs` delegates
to the active variant only, and in runtime mode all variants' inputs are surfaced as a union, so
there is no ambiguity in either case.

### Why the merged graph rather than split-evaluate-merge?

The alternative (split scenarios by variant key, evaluate each group separately, merge results) was
considered and rejected.  It breaks scenario ordering, makes `mv.outputs.x` a dead placeholder
rather than a real graph node (preventing use in parent model formulas), and requires a new
`VariantEvaluationResult` type.  The merged-graph approach keeps `EvaluationResult` unchanged and
lets a `ModelVariant` be composed freely inside a larger model.

---
