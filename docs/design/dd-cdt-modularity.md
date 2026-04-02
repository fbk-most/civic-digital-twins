# Model Modularity

|              | Document data                                  |
|--------------| ---------------------------------------------- |
| Author       | [@pistore](https://github.com/pistore)         |
| Last-Updated | 2026-04-01                                     |
| Status       | Draft                                          |
| Approved-By  | N/A                                            |

This guide explains how to decompose a `Model` into cooperating sub-models in the civic-digital-twins
framework.  It covers the three-level access contract, constructor wiring, `ModelVariant`, the two
decomposition axes (pipeline stages vs. independent concerns), and a full annotated walkthrough of the
Bologna mobility example.

See [dd-cdt-model.md](dd-cdt-model.md) for the index and evaluation layer reference, including the
full `Model` API, `ModelVariant`, and the dataclass I/O contract.

---

## TL;DR

A `Model` subclass declares its public interface through three inner dataclasses — `Inputs`, `Outputs`,
and optionally `Expose` — and receives its upstream dependencies as typed constructor arguments.  A
**root model** wires sub-models together by constructing them inside its own `__init__`, threading
output indexes from one sub-model into the constructor of the next.  `ModelVariant` lets the root
choose among alternative implementations at construction time without changing the downstream wiring.

```python
class TrafficModel(Model):

    @dataclass
    class Inputs:
        ts_inflow:         TimeseriesIndex
        ts_starting:       TimeseriesIndex
        modified_inflow:   Index
        modified_starting: Index

    @dataclass
    class Outputs:
        traffic:                TimeseriesIndex
        modified_traffic:       TimeseriesIndex
        total_modified_traffic: Index
        inflow_ratio:           Index
        starting_ratio:         Index
        traffic_ratio:          Index

    def __init__(
        self,
        ts_inflow: TimeseriesIndex,
        ts_starting: TimeseriesIndex,
        modified_inflow: Index,
        modified_starting: Index,
    ) -> None:
        Inputs  = TrafficModel.Inputs
        Outputs = TrafficModel.Outputs

        inputs = Inputs(
            ts_inflow=ts_inflow,
            ts_starting=ts_starting,
            modified_inflow=modified_inflow,
            modified_starting=modified_starting,
        )

        traffic          = TimeseriesIndex("reference traffic", ...)
        modified_traffic = TimeseriesIndex("modified traffic", ...)
        ...

        super().__init__(
            "Traffic",
            inputs=inputs,
            outputs=Outputs(traffic=traffic, modified_traffic=modified_traffic, ...),
        )
```

The root model wires sub-models by passing outputs of one as constructor arguments to the next:

```python
_inflow  = InflowModel(ts_inflow=ts_inflow, ...)
_traffic = TrafficModel(
    ts_inflow=ts_inflow,
    ts_starting=ts_starting,
    modified_inflow=_inflow.outputs.modified_inflow,     # Level-1 wiring
    modified_starting=_inflow.outputs.modified_starting,
)
_emissions = EmissionsModel(
    traffic=_traffic.outputs.traffic,                    # Level-1 wiring
    modified_traffic=_traffic.outputs.modified_traffic,
    modified_euro_class_split=_inflow.outputs.modified_euro_class_split,
    ...
)
```

---

## Background

### Why decompose a model?

A monolithic `Model.__init__` that constructs every index in a single flat function becomes hard to
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
- The dataclass I/O API (`Inputs`, `Outputs`, `Expose`) — see
  [dd-cdt-model.md](dd-cdt-model.md)

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

### Level 3 — Internal (local variables)

Indexes bound only to local variables inside `__init__` are engine-internal.  They participate in the
computation graph (because other indexes reference their nodes) but are not accessible from outside
the constructor.  No naming convention is required — use whatever makes the implementation readable.

```python
def __init__(self, ...) -> None:
    ...
    # i_modified_average_emissions is a local — never promoted to Outputs or Expose.
    # The engine reaches it transitively via modified_emissions, which references it.
    i_modified_average_emissions = Index(
        "modified average emissions (per vehicle, per km)",
        euro_class_emission["euro_0"] * inputs.modified_euro_class_split[0] + ...,
    )
    modified_emissions = Index(
        "modified emissions",
        graph.piecewise((2.5 * i_modified_average_emissions * inputs.modified_traffic, ...), ...),
    )
    ...
```

### Summary table

| Level | How to access | Stability | May wire into another model? |
|-------|---------------|-----------|-----------------------------|
| 1 | `model.outputs.<field>` / `model.inputs.<field>` | Contractual — stable across versions | Yes |
| 2 | `model.expose.<field>` | Inspectable — may change between versions | No |
| 3 | Local variables inside `__init__` | Internal — not accessible | — |

---

## Wiring Sub-Models via Constructor

Sub-models receive their upstream dependencies as **typed constructor arguments** that are declared in
the `Inputs` dataclass.  The root model constructs sub-models inside its own `__init__`, threading
indexes from one to the next.

### Pattern

```python
class PipelineModel(Model):

    @dataclass
    class Outputs:
        result: Index

    @dataclass
    class Expose:
        stage_a_indexes: list[GenericIndex]
        stage_b_indexes: list[GenericIndex]

    def __init__(self) -> None:
        Outputs = PipelineModel.Outputs
        Expose  = PipelineModel.Expose

        # Step 1 — construct sub-models in dependency order.
        # All leaf indexes are created here; sub-models receive them as arguments.
        stage_a = StageAModel(raw_data=some_timeseries)

        # Step 2 — thread Level-1 outputs of stage A into stage B
        stage_b = StageBModel(
            processed=stage_a.outputs.processed,
            ratio=stage_a.outputs.ratio,
        )

        # Step 3 — promote KPI outputs and collect sub-model indexes for engine visibility
        super().__init__(
            "Pipeline",
            outputs=Outputs(result=stage_b.outputs.result),
            expose=Expose(
                stage_a_indexes=list(stage_a.indexes),
                stage_b_indexes=list(stage_b.indexes),
            ),
        )
```

### Key rules

1. **Construct sub-models as local variables.**  Sub-model instances (`_inflow`, `_traffic`, …) live
   only inside the root's `__init__`.  They are not assigned to `self.*` and are not exposed directly
   — only their index *objects* are promoted to `outputs` or `expose`.

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

Every `GenericIndex` (or `list[GenericIndex]` / `dict[str, GenericIndex]`) passed into a `Model`
subclass `__init__` as a constructor parameter **must** be declared as a field of the `Inputs`
dataclass and forwarded to `super().__init__(inputs=Inputs(...))`.

This rule exists because `Inputs` is the only place where the inter-model wiring contract is expressed
as inspectable metadata.  `ModelVariant`'s cross-variant consistency check reads `model.inputs` field
names — if an index is received but not declared in `Inputs`, the check is blind to it.

```python
# CORRECT — every GenericIndex parameter is declared in Inputs and forwarded
class GoodModel(Model):

    @dataclass
    class Inputs:
        inflow: TimeseriesIndex      # declared here ...

    def __init__(self, inflow: TimeseriesIndex) -> None:
        Inputs = GoodModel.Inputs
        inputs = Inputs(inflow=inflow)  # ... and forwarded here

        super().__init__("Good", inputs=inputs, outputs=...)
```

```python
# INCORRECT — 'inflow' is received but absent from Inputs; InputsContractWarning fires
class BadModel(Model):

    @dataclass
    class Inputs:
        pass   # inflow is missing

    def __init__(self, inflow: TimeseriesIndex) -> None:
        # InputsContractWarning: parameter 'inflow' holds a GenericIndex
        # that is not declared in Inputs.
        super().__init__("Bad", inputs=BadModel.Inputs(), outputs=...)
```

### `InputsContractWarning`

At construction time, `Model.__init__` inspects the calling frame and compares the constructor's
`GenericIndex` parameters against the declared `Inputs` fields.  Any undeclared index parameter
triggers an `InputsContractWarning`.

The warning is **soft** — it does not abort execution — so that existing models can be migrated
incrementally.  During development and in CI, escalate it to an error:

```python
import warnings
from civic_digital_twins.dt_model import ModelContractWarning, InputsContractWarning

# Escalate all contract warnings to errors (recommended for CI)
warnings.filterwarnings("error", category=ModelContractWarning)

# Or target only the inputs-specific warning
warnings.filterwarnings("error", category=InputsContractWarning)
```

`InputsContractWarning` is a subclass of `ModelContractWarning`, so a single filter on the base class
catches all present and future contract-violation categories.

### What `Expose` fields are exempt from

Fields declared in `Expose` are intentionally exempt from this check.  `Expose` is meant to surface
purely internal intermediates; if an `Expose` field were receiving an index from outside the model it
would be a design error — caught by code review, not by the warning mechanism.

---

## `ModelVariant` — Switching Between Implementations

`ModelVariant` selects one `Model` instance from a named mapping at construction time and then acts as
a fully transparent proxy for the chosen (active) variant.

### Construction

```python
from civic_digital_twins.dt_model import ModelVariant

mv = ModelVariant(
    "TransportModel",
    variants={
        "bike":  BikeModel(capacity=100),
        "train": TrainModel(capacity=500),
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
which variant is active.  `inputs` field names may differ across variants — in static mode
`mv.inputs` delegates to the active variant, in runtime mode all variants' inputs are surfaced
as a union.

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
per-scenario rather than fixed for the entire run.

#### `CategoricalIndex` selector — probabilistic, independent choice

A `CategoricalIndex` encodes a finite probability distribution over variant keys.
`DistributionEnsemble` samples it automatically: each scenario receives one variant key drawn
proportional to the declared weights.

```python
from civic_digital_twins.dt_model import CategoricalIndex, ModelVariant

mode = CategoricalIndex("mode", {"bike": 0.3, "train": 0.7})

mv = ModelVariant(
    "TransportModel",
    variants={
        "bike":  BikeModel(),
        "train": TrainModel(),
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
    variants={"bike": BikeModel(), "train": TrainModel(), "metro": MetroModel()},
    selector=ModelVariant.guards_to_selector([
        ("metro", (cost_threshold > 5.0) & (hour >= 8.0)),  # most-specific first
        ("train", cost_threshold > 5.0),
        ("bike",  True),                                     # fallback
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
season = CategoricalIndex("season", {"summer": 0.25, "spring": 0.25,
                                      "autumn": 0.25, "winter": 0.25})

peak_factor = Index("peak_factor", graph.piecewise(
    (1.8, season == "summer"),
    (1.2, season == "spring"),
    (1.0, season == "autumn"),
    (0.7, True),              # winter — default
))
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

Each stage's constructor receives exactly what it needs from the previous stage and nothing more.  The
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
[`examples/mobility_bologna/mobility_bologna.py`](../../examples/mobility_bologna/mobility_bologna.py).

### Overview

```
#  InflowModel  →  TrafficModel  →  EmissionsModel
#       ↘                ↘
#        BolognaModel (root)
#
# BolognaModel declares all policy (i_p_*) and behavioural (i_b_*) parameters
# in its own Inputs dataclass and passes them down to sub-models as constructor
# arguments.  default_inputs() provides the reference-scenario values.
```

The three sub-models and their roles:

| Sub-model | Inputs | Outputs | Notes |
|-----------|--------|---------|-------|
| `InflowModel` | 13 fields: raw timeseries, policy params, behavioural distributions | 11 fields: modified inflow/starting, payment stats, euro-class split | Abstract index: `i_b_p50_cost` (sampled by ensemble) |
| `TrafficModel` | 4 fields: raw timeseries + modified inflow/starting from `InflowModel` | 6 fields: baseline and modified traffic timeseries + ratios | No `Expose` — all intermediates are direct outputs |
| `EmissionsModel` | 6 fields: timeseries, policy window, traffic from `TrafficModel`, euro-class split from `InflowModel` | 5 fields: average emissions factor + baseline/modified totals | One internal local (`i_modified_average_emissions`) |

### `InflowModel` — policy-modified inflow

`InflowModel` takes 13 input indexes and produces 11 output indexes.  The `Inputs` dataclass
documents the full interface at a glance:

```python
class InflowModel(Model):

    @dataclass
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

    @dataclass
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

    @dataclass
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

The constructor signature mirrors `Inputs` exactly:

```python
def __init__(
    self,
    ts_inflow: TimeseriesIndex,
    ts_starting: TimeseriesIndex,
    ts: TimeseriesIndex,
    i_p_start_time: Index,
    i_p_end_time: Index,
    i_p_cost: list[Index],
    i_p_fraction_exempted: Index,
    i_b_p50_cost: DistributionIndex,
    i_b_p50_anticipating: Index,
    i_b_p50_anticipation: Index,
    i_b_p50_postponing: Index,
    i_b_p50_postponement: Index,
    i_b_starting_modified_factor: Index,
) -> None:
    Inputs  = InflowModel.Inputs
    Outputs = InflowModel.Outputs
    Expose  = InflowModel.Expose

    inputs = Inputs(
        ts_inflow=ts_inflow,
        ts_starting=ts_starting,
        ts=ts,
        ...
    )
    # All subsequent formulas reference inputs.* — never the raw parameter names.
    # This ensures the Inputs dataclass is the single authoritative source.
```

### `TrafficModel` — baseline and modified traffic

`TrafficModel` is the simplest of the three sub-models.  It receives four inputs — the raw timeseries
and the policy-modified versions from `InflowModel` — and computes steady-state traffic for both
scenarios together.

```python
class TrafficModel(Model):

    @dataclass
    class Inputs:
        ts_inflow:         TimeseriesIndex
        ts_starting:       TimeseriesIndex
        modified_inflow:   Index             # ← from InflowModel.outputs
        modified_starting: Index             # ← from InflowModel.outputs

    @dataclass
    class Outputs:
        traffic:                TimeseriesIndex   # baseline steady-state
        modified_traffic:       TimeseriesIndex   # policy-modified steady-state
        total_modified_traffic: Index
        inflow_ratio:           Index
        starting_ratio:         Index
        traffic_ratio:          Index

    def __init__(
        self,
        ts_inflow: TimeseriesIndex,
        ts_starting: TimeseriesIndex,
        modified_inflow: Index,
        modified_starting: Index,
    ) -> None:
        Inputs  = TrafficModel.Inputs
        Outputs = TrafficModel.Outputs

        inputs = Inputs(
            ts_inflow=ts_inflow,
            ts_starting=ts_starting,
            modified_inflow=modified_inflow,
            modified_starting=modified_starting,
        )

        traffic          = TimeseriesIndex(
            "reference traffic",
            graph.function_call("ts_solve", inputs.ts_inflow + inputs.ts_starting),
        )
        modified_traffic = TimeseriesIndex(
            "modified traffic",
            graph.function_call("ts_solve", inputs.modified_inflow + inputs.modified_starting),
        )
        total_modified_traffic = Index("total modified traffic", modified_traffic.sum())
        inflow_ratio     = Index("ratio between modified flow and base flow",
                                inputs.ts_inflow / inputs.modified_inflow)
        starting_ratio   = Index("ratio between modified starting and base starting",
                                inputs.ts_starting / inputs.modified_starting)
        traffic_ratio    = Index("ratio between modified traffic and base traffic",
                                traffic / modified_traffic)

        super().__init__(
            "Traffic",
            inputs=inputs,
            outputs=Outputs(
                traffic=traffic,
                modified_traffic=modified_traffic,
                total_modified_traffic=total_modified_traffic,
                inflow_ratio=inflow_ratio,
                starting_ratio=starting_ratio,
                traffic_ratio=traffic_ratio,
            ),
        )
```

**Annotations:**

- `TrafficModel` has no `Expose` because there are no diagnostically interesting intermediates.  Every
  computed index feeds directly into an output.

- Both `traffic` (baseline) and `modified_traffic` (policy scenario) are co-located here because they
  share the same `ts_solve` computation structure.  Separating them across model boundaries would split
  a symmetric pair for no benefit — see "Why drop `BaseStateModel`?" in the
  [Design Rationale](#design-rationale) section.

- All intermediate computations access upstream indexes through `inputs.*` rather than through the raw
  constructor parameter names.  This is the required convention: once `inputs` is constructed, the
  parameters are no longer directly referenced.

### `EmissionsModel` — baseline and modified emissions

`EmissionsModel` receives the baseline and modified traffic timeseries, the per-euro-class distribution
from `InflowModel`, and the policy time window, and produces emission totals for both scenarios.

```python
class EmissionsModel(Model):

    @dataclass
    class Inputs:
        ts:                        TimeseriesIndex
        i_p_start_time:            Index
        i_p_end_time:              Index
        traffic:                   TimeseriesIndex   # ← from TrafficModel.outputs
        modified_traffic:          TimeseriesIndex   # ← from TrafficModel.outputs
        modified_euro_class_split: list[Index]       # ← from InflowModel.outputs

    @dataclass
    class Outputs:
        average_emissions:        Index            # fleet-weighted baseline factor
        emissions:                TimeseriesIndex  # baseline timeseries
        modified_emissions:       Index            # policy-modified total
        total_emissions:          Index
        total_modified_emissions: Index

    def __init__(
        self,
        ts: TimeseriesIndex,
        i_p_start_time: Index,
        i_p_end_time: Index,
        traffic: TimeseriesIndex,
        modified_traffic: TimeseriesIndex,
        modified_euro_class_split: list[Index],
    ) -> None:
        Inputs  = EmissionsModel.Inputs
        Outputs = EmissionsModel.Outputs

        inputs = Inputs(
            ts=ts,
            i_p_start_time=i_p_start_time,
            i_p_end_time=i_p_end_time,
            traffic=traffic,
            modified_traffic=modified_traffic,
            modified_euro_class_split=modified_euro_class_split,
        )

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

        super().__init__(
            "Emissions",
            inputs=inputs,
            outputs=Outputs(
                average_emissions=average_emissions,
                emissions=emissions,
                modified_emissions=modified_emissions,
                total_emissions=Index("total emissions", emissions.sum()),
                total_modified_emissions=Index("total modified emissions", modified_emissions.sum()),
            ),
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
behavioural (`i_b_*`) parameters are declared in `Inputs` and received as constructor arguments.
Default values for the reference scenario are provided by the `default_inputs()` class method,
which callers can override selectively using `**`-unpacking.

```python
class BolognaModel(Model):

    @dataclass
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

    @dataclass
    class Outputs:
        total_base_inflow:        Index
        total_modified_inflow:    Index
        total_shifted:            Index
        total_paying:             Index
        avg_cost:                 Index
        total_payed:              Index
        total_emissions:          Index
        total_modified_emissions: Index

    @dataclass
    class Expose:
        # Timeseries surfaced for plotting helpers
        ts_inflow:          TimeseriesIndex
        modified_inflow:    Index
        traffic:            TimeseriesIndex
        modified_traffic:   TimeseriesIndex
        emissions:          TimeseriesIndex
        modified_emissions: Index

    @classmethod
    def default_inputs(cls) -> dict:
        """Reference-scenario parameters as a keyword-argument dict."""
        return {
            "i_p_start_time":              Index("start time", ...),
            "i_p_end_time":                Index("end time", ...),
            "i_p_cost":                    [Index(f"cost euro {e}", 5.00 - e * 0.25) for e in range(7)],
            "i_p_fraction_exempted":       Index("exempted vehicles %", 0.15),
            "i_b_p50_cost":                DistributionIndex("cost 50% threshold", stats.uniform, {...}),
            "i_b_p50_anticipating":        Index("anticipation 50% likelihood", 0.5),
            "i_b_p50_anticipation":        Index("anticipation distribution 50% threshold", 0.25),
            "i_b_p50_postponing":          Index("postponement 50% likelihood", 0.8),
            "i_b_p50_postponement":        Index("postponement distribution 50% threshold", 0.50),
            "i_b_starting_modified_factor": Index("starting modified factor", 1.00),
        }

    def __init__(
        self,
        *,
        i_p_start_time: Index,
        i_p_end_time: Index,
        i_p_cost: list[Index],
        i_p_fraction_exempted: Index,
        i_b_p50_cost: DistributionIndex,
        i_b_p50_anticipating: Index,
        i_b_p50_anticipation: Index,
        i_b_p50_postponing: Index,
        i_b_p50_postponement: Index,
        i_b_starting_modified_factor: Index,
    ) -> None:
        Inputs  = BolognaModel.Inputs
        Outputs = BolognaModel.Outputs
        Expose  = BolognaModel.Expose

        inputs = Inputs(
            i_p_start_time=i_p_start_time,
            ...
        )

        # ── Internal timeseries (Level 3) ──────────────────────────────────────
        ts          = TimeseriesIndex("time range", np.array([...]))
        ts_inflow   = TimeseriesIndex("inflow", vehicle_inflow)
        ts_starting = TimeseriesIndex("starting", vehicle_starting)

        # ── Sub-models in pipeline order ──────────────────────────────────────
        _inflow = InflowModel(
            ts_inflow=ts_inflow,
            ts_starting=ts_starting,
            ts=ts,
            i_p_start_time=i_p_start_time,
            ...
        )

        _traffic = TrafficModel(
            ts_inflow=ts_inflow,
            ts_starting=ts_starting,
            modified_inflow=_inflow.outputs.modified_inflow,      # ← Level-1 wiring
            modified_starting=_inflow.outputs.modified_starting,
        )

        _emissions = EmissionsModel(
            ts=ts,
            i_p_start_time=i_p_start_time,
            i_p_end_time=i_p_end_time,
            traffic=_traffic.outputs.traffic,                     # ← Level-1 wiring
            modified_traffic=_traffic.outputs.modified_traffic,
            modified_euro_class_split=_inflow.outputs.modified_euro_class_split,
        )

        # ── Root super().__init__ ─────────────────────────────────────────────
        super().__init__(
            "Bologna mobility",
            inputs=inputs,
            outputs=Outputs(
                total_base_inflow=_inflow.outputs.total_base_inflow,
                ...
                total_modified_emissions=_emissions.outputs.total_modified_emissions,
            ),
            expose=Expose(
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
engine.  `Outputs` covers the 8 KPI scalars; `Expose` covers the plotting timeseries.  No bulk
`list[GenericIndex]` fields are needed.

**Annotation — `outputs` stores references to sub-model index objects:**

`total_base_inflow=_inflow.outputs.total_base_inflow` stores a reference to the same `Index` object
that lives inside `InflowModel`.  The evaluation engine operates on object identity, so no duplication
or aliasing occurs.  The `BolognaModel` does not own these indexes; it is a wiring hub.

### Using `BolognaModel`

```python
# Reference scenario — use built-in defaults
m = BolognaModel(**BolognaModel.default_inputs())

# Alternative scenario — override one parameter
m_strict = BolognaModel(**{
    **BolognaModel.default_inputs(),
    "i_p_cost": [Index(f"cost euro {e}", 8.00 - e * 0.50) for e in range(7)],
})

ensemble = DistributionEnsemble(m, size=500)
result   = Evaluation(m).evaluate(ensemble)

# Read KPI outputs by name
total_inflow_modified = result.marginalize(m.outputs.total_modified_inflow)
total_emissions       = result.marginalize(m.outputs.total_emissions)

# Access raw timeseries through expose (shape S×T)
modified_inflow_ts = result[m.expose.modified_inflow]

# Access constant timeseries (no scenario dimension)
reference_inflow = result[m.expose.ts_inflow]  # shape (T,)
```

The evaluation layer is unaware of sub-models.  It sees a flat `m.indexes` list, resolves the graph,
and evaluates it.  The sub-model structure is a pure construction-time concern; it has zero runtime
overhead.

---

## API Reference

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
produce graph nodes: `mode == "bike"` yields `graph.equal(mode.node, constant("bike"))`, a valid
boolean-per-scenario node usable in `graph.piecewise` or any formula.

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
from civic_digital_twins.dt_model import ModelContractWarning, InputsContractWarning

# Recommended for CI — escalate all contract warnings to errors
warnings.filterwarnings("error", category=ModelContractWarning)

# Fine-grained — only escalate the inputs-specific warning
warnings.filterwarnings("error", category=InputsContractWarning)
```

---

## Design Rationale

### Why constructor arguments rather than a separate `wire()` step?

Wiring via the constructor gives a typed, IDE-navigable, one-shot configuration.  There is no mutable
state to reason about — once `__init__` returns, all indexes are fully wired and the model is
immutable.  A separate `wire()` step would require the model to hold partially-constructed state,
complicating `is_instantiated()` and making order-of-calls errors possible.

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
