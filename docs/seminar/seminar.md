---
marp: true
theme: default
paginate: true
style: |
  section {
    font-size: 1.5rem;
    font-family: "Segoe UI", "Helvetica Neue", sans-serif;
  }
  section.lead {
    text-align: center;
    justify-content: center;
  }
  section.lead h1 {
    font-size: 2.4rem;
  }
  section.lead h2 {
    font-size: 1.6rem;
    color: #555;
    margin-top: 0.2em;
  }
  section.lead p {
    color: #777;
    font-size: 1.1rem;
  }
  section.section-header {
    background: #1a1a2e;
    color: white;
    text-align: center;
    justify-content: center;
  }
  section.section-header h1 {
    font-size: 2.2rem;
    color: white;
  }
  section.section-header p {
    color: #aaa;
    font-size: 1.2rem;
  }
  h1 { color: #1a1a2e; }
  h2 { color: #1a1a2e; border-bottom: 2px solid #e0e0e0; padding-bottom: 0.2em; }
  code { background: #f4f4f8; border-radius: 4px; padding: 0.1em 0.3em; }
  pre  { font-size: 0.82rem; line-height: 1.45; }
  table { font-size: 1.1rem; }
  ul { line-height: 1.8; }
  li { margin-bottom: 0.15em; }
  .columns { display: grid; grid-template-columns: 1fr 1fr; gap: 2em; }
  footer { font-size: 0.8rem; color: #aaa; }
---

<!-- _class: lead -->
<!-- _paginate: false -->

# Civic Digital Twins
## Modelling Urban Mobility with Typed Sub-Models

*IO Contracts · Modularity · Model Variants*

<br>

Fondazione Bruno Kessler — FBK MOST

---

<!-- _footer: "Section 1 — Context" -->

## The Bologna mobility challenge

**Setting:** Bologna city centre — a *Zona a Traffico Limitato* (ZTL)

- ≈ 400 000 vehicle entries per day
- Fleet composed of Euro 0 – Euro 6 vehicles
- Each Euro class carries a different NOx emission factor

**Policy under study:** road pricing

- Vehicles pay a per-entry fee, graduated by Euro class
- Higher-polluting vehicles pay more → expected to reduce entries
- Some vehicles *anticipate* (enter earlier), some *postpone* (enter later)

**Question:** given the pricing parameters, what are the effects on traffic
and NOx emissions?

---

<!-- _footer: "Section 1 — Context" -->

## What we want to compute

Three families of KPIs, all uncertain:

| KPI | Unit | Uncertainty source |
|-----|------|--------------------|
| Modified vehicle inflow | veh / day | price-elasticity threshold |
| Shifted inflow | veh / day | anticipation / postponement behaviour |
| NOx emissions (base vs. modified) | g / day | price-elasticity threshold |

<br>

The price-elasticity threshold `i_b_p50_cost` is modelled as a
`DistributionIndex` — a random variable uniform over [4 €, 11 €].
The model is evaluated over an **ensemble** of scenarios sampled from
this distribution.

---

<!-- _footer: "Section 1 — Context" -->

## How we will model it

The computation has a natural **pipeline** structure:

```
vehicle data                  policy parameters
     │                               │
     ▼                               ▼
 InflowModel  ──────────────────────────►  modified inflow
     │                                           │
     ▼                                           ▼
 TrafficModel  ◄──── modified inflow ────  circulating traffic
     │
     ▼
 EmissionsModel  ◄──── Euro-class mix ────  NOx emissions
     │
     ▼
 BolognaModel  (root — wires all three, exposes KPIs)
```

Each box is a **`Model` subclass** with a declared typed interface.
We will build the framework step by step.

---

<!-- _class: section-header -->

# Part 1 — CDT Framework Basics

*Indexes · Models · Evaluation*

---

<!-- _footer: "Part 1 — Basics" -->

## The framework in three layers

```
┌─────────────────────────────────────────────────────┐
│  Model layer                                        │
│  Index, TimeseriesIndex, DistributionIndex, Model   │
│  → domain language; what you write                  │
├─────────────────────────────────────────────────────┤
│  Simulation layer                                   │
│  DistributionEnsemble, Evaluation, EvaluationResult │
│  → sample → evaluate → aggregate                    │
├─────────────────────────────────────────────────────┤
│  Engine layer                                       │
│  graph (DAG nodes), linearize, executor (NumPy)     │
│  → how computations are actually run                │
└─────────────────────────────────────────────────────┘
```

As a modeller you work almost exclusively in the **model** and
**simulation** layers.  The engine is largely invisible.

---

<!-- _footer: "Part 1 — Basics" -->

## Indexes — the atoms of a model

Every quantity in the model is an **`Index`**:

```python
from civic_digital_twins.dt_model import Index, DistributionIndex, TimeseriesIndex
from scipy import stats
import numpy as np

# Constant
cost_euro0 = Index("cost euro_0", 5.00)

# Formula  (lazy graph expression)
avg_cost   = Index("avg cost", cost_euro0 * 0.30 + ...)

# Uncertain — sampled at evaluation time
threshold  = DistributionIndex("cost threshold",
                stats.uniform, {"loc": 4.0, "scale": 7.0})

# Time-indexed — one value per time step
inflow     = TimeseriesIndex("inflow", np.array([...]))  # shape (T,)
```

Arithmetic operators (`+`, `*`, `/`, …) build a **lazy computation
graph** — nothing is evaluated until `Evaluation.evaluate()` is called.

---

<!-- _footer: "Part 1 — Basics" -->

## Uncertainty: `DistributionIndex`

A `DistributionIndex` is an **abstract** index: its value is not fixed
— it is drawn from a distribution in each scenario.

```python
# Price-elasticity threshold: uniform over [4 €, 11 €]
i_b_p50_cost = DistributionIndex(
    "cost 50% threshold",
    stats.uniform,
    {"loc": 4.00, "scale": 7.00},
)
```

- `model.abstract_indexes()` returns all distribution-backed indexes
- `model.is_instantiated()` is `False` while any abstract index exists
- `DistributionEnsemble` draws `size` independent samples and yields
  equally-weighted scenarios

---

<!-- _footer: "Part 1 — Basics" -->

## Running a simulation

```python
from civic_digital_twins.dt_model import DistributionEnsemble, Evaluation
from civic_digital_twins.dt_model.engine.numpybackend import executor
from mobility_bologna import BolognaModel, compute_kpis, _ts_solve

# 1. Build the model
model = BolognaModel()

# 2. Sample uncertain parameters (200 scenarios)
ensemble = DistributionEnsemble(model, size=200)

# 3. Evaluate — plug in the traffic solver function
result = Evaluation(model).evaluate(
    ensemble,
    functions={"ts_solve": executor.LambdaAdapter(_ts_solve)},
)

# 4. Read KPIs
kpis = compute_kpis(model, result)
```

`result[idx]` → raw array, shape `(200, 1)` (one value per scenario)
`result.marginalize(idx)` → weighted mean across scenarios

---

<!-- _footer: "Part 1 — Basics" -->

## KPI output (sample run, 200 scenarios)

| KPI | Value |
|-----|-------|
| Base inflow | 396 482 veh/day |
| Modified inflow | 368 714 veh/day |
| Shifted inflow | 27 768 veh/day |
| Paying inflow | 270 593 veh/day |
| Collected fees | 954 127 €/day |
| Base emissions | 21 456 318 NOx g/day |
| Emission reduction | 2 013 445 NOx g/day |

<br>

*Numbers are expectations over the 200-scenario ensemble.*
*The uncertainty on the price-elasticity threshold propagates into all KPIs.*

---

<!-- _footer: "Part 1 — Basics" -->

## The problem with a flat model

Before the current design, the model looked like this:

```python
# Everything in one flat list — hard to read, test, or extend
model = Model("Bologna", [
    ts_inflow, ts_starting, ts,
    i_p_start_time, i_p_end_time,
    i_p_cost_euro0, i_p_cost_euro1, ...,   # 7 cost indexes
    i_b_p50_cost, i_b_p50_anticipating, ..., # behavioural params
    fraction_rigid, i_fraction_rigid_euro0, ...,  # intermediate
    modified_inflow, modified_starting, ...,
    traffic, modified_traffic, traffic_ratio,
    average_emissions, emissions, modified_emissions,
    # ... 60+ indexes total
])
```

**Problems:**
- Which indexes are *inputs*? Which are *outputs*?
- How do you test the traffic sub-computation in isolation?
- How do you swap one computation for another?

---

<!-- _class: section-header -->

# Part 2 — IO Contracts

*Inputs · Outputs · Expose · Warnings*

---

<!-- _footer: "Part 2 — IO Contracts" -->

## From flat lists to structured interfaces

The key idea: declare **what a model needs** and **what it produces**
as inner `@dataclass` classes.

```python
from dataclasses import dataclass
from civic_digital_twins.dt_model import Model, Index, TimeseriesIndex, DistributionIndex

class InflowModel(Model):

    @dataclass
    class Inputs:
        ts_inflow:    TimeseriesIndex   # raw vehicle inflow
        i_p_cost:     list[Index]       # pricing schedule (per Euro class)
        i_b_p50_cost: DistributionIndex # price-elasticity threshold ← uncertain
        ...

    @dataclass
    class Outputs:
        modified_inflow: Index   # inflow after policy effect
        total_paying:    Index   # number of paying vehicles
        avg_cost:        Index   # average fee paid
        ...
```

`model.indexes` is derived **automatically** — no flat list to maintain.

---

<!-- _footer: "Part 2 — IO Contracts" -->

## Inside `InflowModel.__init__`

```python
def __init__(self, ts_inflow, ts_starting, ..., i_b_p50_cost, ...) -> None:
    Inputs  = InflowModel.Inputs
    Outputs = InflowModel.Outputs

    # 1. Pack all inputs into the dataclass instance
    inputs = Inputs(
        ts_inflow=ts_inflow,
        i_b_p50_cost=i_b_p50_cost,
        ...
    )

    # 2. Build the computation graph (lazy — not evaluated yet)
    fraction_rigid  = Index("rigid vehicles %", ...)
    modified_inflow = Index("modified inflow", ...)
    total_paying    = Index("total paying", ...)

    # 3. Declare outputs — stable public contract
    super().__init__(
        "Inflow",
        inputs=inputs,
        outputs=Outputs(modified_inflow=modified_inflow,
                        total_paying=total_paying, ...),
    )
```

---

<!-- _footer: "Part 2 — IO Contracts" -->

## Three levels of visibility

| Level | How to access | Stability | Can be wired? |
|-------|---------------|-----------|---------------|
| **1 — Contractual** | `model.outputs.*` / `model.inputs.*` | Stable across versions | ✅ Yes |
| **2 — Inspectable** | `model.expose.*` | May change | ❌ No |
| **3 — Internal** | local variables in `__init__` | Not accessible | — |

<br>

A caller that depends on `inflow.outputs.modified_inflow` is protected
by the contract.  A caller that reads `inflow.expose.i_fraction_anticipating`
for a plot is **not** wiring it into another model — only reading it.

---

<!-- _footer: "Part 2 — IO Contracts" -->

## Level 2 — `Expose`: diagnostics without contract

`Expose` surfaces intermediate indexes useful for plotting or debugging,
**without** making them part of the stable API.

```python
@dataclass
class Expose:
    i_fraction_anticipating: TimeseriesIndex  # fraction of vehicles anticipating
    i_number_anticipating:   TimeseriesIndex  # count of anticipating vehicles
    i_delta_from_start:      TimeseriesIndex  # time elapsed since pricing window
    ...  # 16 fields in total
```

```python
inflow = InflowModel(...)

# ✅ Reading for a plot — perfectly fine
frac = inflow.expose.i_fraction_anticipating

# ❌ Wiring into another model — FORBIDDEN
bad_model = SomeModel(anticipating=inflow.expose.i_fraction_anticipating)
```

`Expose` is for **reading**, never for **wiring**.

---

<!-- _footer: "Part 2 — IO Contracts" -->

## `InputsContractWarning` — soft enforcement

Every `GenericIndex` received as a constructor parameter **must** appear
in `Inputs`.  If it does not, a warning is emitted at construction time:

```python
class BadModel(Model):
    @dataclass
    class Inputs:
        pass  # 'inflow' is missing

    def __init__(self, inflow: TimeseriesIndex) -> None:
        # ⚠️  InputsContractWarning:
        #     parameter 'inflow' holds a GenericIndex not declared in Inputs
        super().__init__("Bad", inputs=BadModel.Inputs(), outputs=...)
```

The warning is **soft** (execution continues) so existing models can be
migrated incrementally.

```python
# Harden in CI — turn all contract warnings into errors:
import warnings
from civic_digital_twins.dt_model import ModelContractWarning
warnings.filterwarnings("error", category=ModelContractWarning)
```

---

<!-- _footer: "Part 2 — IO Contracts" -->

## What IO contracts give you

<div class="columns">
<div>

**Clarity**
- The `Inputs` dataclass *is* the documentation
- Type annotations checked statically by pyright
- No need to grep through `__init__` to find what is needed

**Safety**
- `ModelVariant` validates field names across variants (later)
- `InputsContractWarning` catches wiring mistakes early
- `model.indexes` is derived — never out of sync

</div>
<div>

**Testability**

```python
# Test InflowModel in isolation
inflow = InflowModel(
    ts_inflow=stub_inflow,
    ts_starting=stub_starting,
    ts=stub_ts,
    i_p_cost=fixed_costs,
    i_b_p50_cost=DistributionIndex(...),
    ...
)
assert inflow.outputs.total_paying  is not None
assert inflow.outputs.modified_inflow is not None
assert inflow.is_instantiated() is False
```

</div>
</div>

---

<!-- _class: section-header -->

# Part 3 — Modularity

*Constructor wiring · Pipeline · Root model*

---

<!-- _footer: "Part 3 — Modularity" -->

## Why decompose?

A monolithic `__init__` with 60 + indexes is:

- **Unreadable** — no screen fits it; the reader cannot see boundaries
- **Untestable** — the traffic computation cannot be isolated
- **Rigid** — replacing the traffic solver requires touching everything

The solution: each **conceptual stage** becomes its own `Model` subclass,
receiving its upstream dependencies as typed constructor arguments.

```
InflowModel   → what happens to vehicle entries under pricing?
TrafficModel  → given inflow, what is the circulating traffic?
EmissionsModel→ given traffic and fleet mix, what are NOx emissions?
BolognaModel  → root: wires the three stages, exposes KPIs
```

---

<!-- _footer: "Part 3 — Modularity" -->

## The wiring pattern

Sub-models are constructed **inside the root's `__init__`** and are
never stored on `self` — only their **output index objects** are kept.

```python
class BolognaModel(Model):
    def __init__(self) -> None:

        # Leaf-level indexes created here
        ts_inflow    = TimeseriesIndex("inflow", vehicle_inflow)
        i_b_p50_cost = DistributionIndex("cost threshold", stats.uniform, ...)
        ...

        # Stage 1
        _inflow = InflowModel(ts_inflow=ts_inflow, ..., i_b_p50_cost=i_b_p50_cost)

        # Stage 2 — receives Level-1 outputs of stage 1
        _traffic = TrafficModel(
            ts_inflow=ts_inflow,
            modified_inflow=_inflow.outputs.modified_inflow,   # ← wiring
            modified_starting=_inflow.outputs.modified_starting,
        )

        # Stage 3 — receives Level-1 outputs of stages 1 and 2
        _emissions = EmissionsModel(
            traffic=_traffic.outputs.traffic,                  # ← wiring
            modified_euro_class_split=_inflow.outputs.modified_euro_class_split,
            ...
        )
```

---

<!-- _footer: "Part 3 — Modularity" -->

## `TrafficModel` in full

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

    def __init__(self, ts_inflow, ts_starting, modified_inflow, modified_starting):
        inputs  = TrafficModel.Inputs(...)
        traffic = TimeseriesIndex("traffic",
                      graph.function_call("ts_solve",
                                          inputs.ts_inflow + inputs.ts_starting))
        ...
        super().__init__("Traffic", inputs=inputs, outputs=TrafficModel.Outputs(...))
```

---

<!-- _footer: "Part 3 — Modularity" -->

## `EmissionsModel` wiring

`EmissionsModel` receives outputs from **both** previous stages:

```python
_emissions = EmissionsModel(
    ts=ts,
    i_p_start_time=i_p_start_time,
    i_p_end_time=i_p_end_time,

    traffic=_traffic.outputs.traffic,                        # ← TrafficModel
    modified_traffic=_traffic.outputs.modified_traffic,      # ← TrafficModel

    modified_euro_class_split=_inflow.outputs.modified_euro_class_split,
    #                                           ▲ InflowModel
)
```

`EmissionsModel.Inputs` declares **all six** of these fields — the
contract is explicit and machine-checkable.

---

<!-- _footer: "Part 3 — Modularity" -->

## `BolognaModel` — collecting the KPIs

```python
super().__init__(
    "Bologna mobility",
    outputs=Outputs(
        total_base_inflow=_inflow.outputs.total_base_inflow,
        total_modified_inflow=_inflow.outputs.total_modified_inflow,
        total_shifted=_inflow.outputs.total_shifted,
        total_paying=_inflow.outputs.total_paying,
        avg_cost=_inflow.outputs.avg_cost,
        total_payed=_inflow.outputs.total_paid,
        total_emissions=_emissions.outputs.total_emissions,
        total_modified_emissions=_emissions.outputs.total_modified_emissions,
    ),
    expose=Expose(
        # Named timeseries for plotting
        ts_inflow=ts_inflow,
        traffic=_traffic.outputs.traffic,
        emissions=_emissions.outputs.emissions,
        ...
        # All sub-model indexes, so the engine can reach every graph node
        inflow_indexes=list(_inflow.indexes),
        traffic_indexes=list(_traffic.indexes),
        emissions_indexes=list(_emissions.indexes),
    ),
)
```

---

<!-- _footer: "Part 3 — Modularity" -->

## Reading KPIs through the contract

```python
def compute_kpis(m: BolognaModel, evals: dict) -> dict:
    return {
        "Base inflow [veh/day]":
            int(evals[m.outputs.total_base_inflow].mean()),

        "Modified inflow [veh/day]":
            int(evals[m.outputs.total_modified_inflow].mean()),

        "Emissions [NOx g/day]":
            int(evals[m.outputs.total_emissions].mean()),

        "Emission reduction [NOx g/day]":
            int(evals[m.outputs.total_emissions].mean())
            - int(evals[m.outputs.total_modified_emissions].mean()),
        ...
    }
```

All access goes through `m.outputs.*` — the contract.
No index is addressed by name string or list position.

---

<!-- _footer: "Part 3 — Modularity" -->

## The full picture

```
                       BolognaModel
                      ┌────────────────────────────────────┐
leaf indexes ────────►│ __init__                           │
(ts_inflow,           │                                    │
 i_b_p50_cost, …)     │   _inflow   = InflowModel(…)       │
                      │                     │ outputs.*    │
                      │   _traffic  = TrafficModel(…      ◄┘
                      │              modified_inflow=      │
                      │              _inflow.outputs.…)    │
                      │                     │ outputs.*    │
                      │   _emissions= EmissionsModel(…    ◄┘
                      │              traffic=              │
                      │              _traffic.outputs.…)   │
                      │                                    │
                      │   super().__init__(outputs=…  ◄────┘
                      └────────────────────────────────────┘
                            │ outputs.*
                            ▼
                       compute_kpis(m, evals)
```

---

<!-- _class: section-header -->

# Part 4 — Model Variants

*Static selection · Two kinds of variation · Future work*

---

<!-- _footer: "Part 4 — Model Variants" -->

## Motivation: swapping implementations

`TrafficModel` computes circulating traffic via `_ts_solve` —
an iterative steady-state solver written in Python.

Two natural questions arise:

**a) Can we use a different model formulation?**
Perhaps a simpler linear approximation for fast exploration,
or a more refined non-linear formula for higher accuracy —
same phenomenon, different mathematical description.

**b) Can we plug in an external simulator?**
Real deployments may want to call an external traffic simulator
(e.g. SUMO) instead of the built-in Python solver —
same data flow, different computation engine.

Both cases share the **same `Inputs` / `Outputs` interface**.
`ModelVariant` makes the swap explicit, validated, and reversible.

---

<!-- _footer: "Part 4 — Model Variants" -->

## Case a — same phenomenon, two model formulations

```python
# Variant A: linear approximation (fast; useful for exploration)
class SimpleTrafficModel(Model):
    Inputs  = TrafficModel.Inputs   # identical interface
    Outputs = TrafficModel.Outputs

    def __init__(self, ts_inflow, ts_starting, modified_inflow, modified_starting):
        inputs = SimpleTrafficModel.Inputs(...)

        # No iterative solver — direct sum is the approximation
        traffic          = TimeseriesIndex("traffic",
                               inputs.ts_inflow + inputs.ts_starting)
        modified_traffic = TimeseriesIndex("modified traffic",
                               inputs.modified_inflow + inputs.modified_starting)
        ...
        super().__init__("SimpleTraffic", inputs=inputs,
                         outputs=SimpleTrafficModel.Outputs(...))


# Variant B: iterative steady-state (the existing TrafficModel)
# traffic = ts_solve(ts_inflow + ts_starting)   ← 50-iteration feedback loop
```

*Same interface. Different formula. Downstream code is unaware of the choice.*

---

<!-- _footer: "Part 4 — Model Variants" -->

## Case b — same structure, different engine

```python
# Variant C: external traffic simulator (e.g. SUMO)
class ExternalSimulatorTrafficModel(Model):
    Inputs  = TrafficModel.Inputs   # identical interface
    Outputs = TrafficModel.Outputs

    def __init__(self, ts_inflow, ts_starting, modified_inflow, modified_starting):
        inputs = ExternalSimulatorTrafficModel.Inputs(...)

        # Raw timeseries are passed directly to the external solver
        # — the CDT graph does not pre-combine them.
        # .node extracts the underlying graph.Node from each GenericIndex.
        traffic = TimeseriesIndex(
            "traffic",
            graph.function_call("sumo_simulate",
                                inputs.ts_inflow.node, inputs.ts_starting.node),
        )
        modified_traffic = TimeseriesIndex(
            "modified traffic",
            graph.function_call("sumo_simulate",
                                inputs.modified_inflow.node, inputs.modified_starting.node),
        )
        ...
```

At evaluation time, `"sumo_simulate"` is registered with a
`LambdaAdapter` that calls the external process.

---

<!-- _footer: "Part 4 — Model Variants" -->

## `ModelVariant` — the selector

```python
from civic_digital_twins.dt_model import ModelVariant

traffic = ModelVariant(
    "TrafficModel",
    variants={
        "simple":   SimpleTrafficModel(
                        ts_inflow=ts_inflow, ts_starting=ts_starting,
                        modified_inflow=..., modified_starting=...),
        "iterative": TrafficModel(
                        ts_inflow=ts_inflow, ts_starting=ts_starting,
                        modified_inflow=..., modified_starting=...),
        "sumo":      ExternalSimulatorTrafficModel(
                        ts_inflow=ts_inflow, ts_starting=ts_starting,
                        modified_inflow=..., modified_starting=...),
    },
    selector="iterative",   # resolved once at construction time
)
```

`traffic` now behaves **exactly** like the active `Model` instance:

```python
_emissions = EmissionsModel(
    traffic=traffic.outputs.traffic,           # same as before
    modified_traffic=traffic.outputs.modified_traffic,
    ...
)
```

---

<!-- _footer: "Part 4 — Model Variants" -->

## Contract enforcement across variants

`ModelVariant` validates at construction time that all variants share
**identical `Inputs` and `Outputs` field names**:

```python
# ❌ This raises ValueError immediately
traffic = ModelVariant(
    "TrafficModel",
    variants={
        "iterative": TrafficModel(...),
        # WrongModel.Outputs has field 'result' instead of 'traffic'
        "wrong":     WrongModel(...),
    },
    selector="iterative",
)
# ValueError: variants 'iterative' and 'wrong' have different
#             outputs field names: {'traffic', ...} vs {'result', ...}
```

Inactive variants remain fully accessible for inspection:

```python
traffic.variants["simple"].outputs.traffic   # the SimpleTrafficModel output
traffic.variants["sumo"].is_instantiated()   # the SUMO variant's state
```

---

<!-- _footer: "Part 4 — Model Variants" -->

## Future: dynamic (runtime) variant selection

**Current behaviour:** `selector` is a static string — the active
variant is fixed at construction time and does not change.

**Future:** select a variant *per scenario*, driven by a random variable.

```python
# Conceptual — not yet implemented
transport_mode = CategoricalIndex(
    "transport_mode",
    {"private_car": 0.7, "public_transit": 0.3},  # probabilities
)

traffic = ModelVariant(
    "TrafficModel",
    variants={"car": CarTrafficModel(...), "transit": TransitModel(...)},
    selector=transport_mode,   # ← would select per scenario
)
```

This requires a `CategoricalIndex` type and evaluation-layer support.
It is tracked as a future roadmap item (issue #130).

---

<!-- _class: section-header -->

# Part 5 — Putting it all together

*Results · Summary*

---

<!-- _footer: "Part 5 — Summary" -->

## Bologna: simulation results

*200-scenario ensemble; `i_b_p50_cost` ~ Uniform(4 €, 11 €)*

```
┌─────────────────────────────────────┬────────────────┐
│ KPI                                 │ Expected value │
├─────────────────────────────────────┼────────────────┤
│ Base inflow                         │  396 482 veh/d │
│ Modified inflow                     │  368 714 veh/d │
│ Shifted inflow                      │   27 768 veh/d │
│ Paying inflow                       │  270 593 veh/d │
│ Collected fees                      │  954 127 €/d   │
│ Base NOx emissions                  │   21 456 kg/d  │
│ NOx emission reduction              │    2 013 kg/d  │
└─────────────────────────────────────┴────────────────┘
```

The ~9.4 % reduction in NOx comes entirely from model uncertainty in
the price-elasticity threshold — the single `DistributionIndex` in the
whole model propagates through the pipeline to every KPI.

---

<!-- _footer: "Part 5 — Summary" -->

## What we built — summary

| Concept | Mechanism | What it gives you |
|---------|-----------|-------------------|
| **IO Contract** | `Inputs` / `Outputs` `@dataclass` | Explicit, type-checked interface |
| **Three-level access** | `outputs`, `expose`, locals | Stable API vs diagnostics vs internals |
| **Contract warning** | `InputsContractWarning` | Early detection of wiring mistakes |
| **Modularity** | Constructor wiring | Testable, replaceable sub-models |
| **Root model** | `BolognaModel` | Single entry point; all indexes visible to engine |
| **ModelVariant** | Static selector | A/B swap between implementations |

<br>

The same pattern scales from a 5-index toy model to the full
60-index Bologna model — and to any other civic domain.

---

<!-- _footer: "Part 5 — Summary" -->

## Roadmap

**v0.8.0 (current)**
- ✅ IO contracts (`Inputs`, `Outputs`, `Expose`)
- ✅ `InputsContractWarning`
- ✅ `ModelVariant` with static selector
- ✅ Bologna and Molveno modular examples

**Next milestones**
- `CategoricalIndex` — discrete uncertain parameter (e.g. transport mode)
- Dynamic `ModelVariant` — per-scenario variant selection driven by `CategoricalIndex`
- Model introspection tooling — auto-generate wiring diagrams from `Inputs` / `Outputs`

---

<!-- _class: lead -->
<!-- _paginate: false -->

# Thank you

<br>

**Companion code**
`docs/seminar/seminar_bologna.py`

**Documentation**
`docs/getting-started.md`
`docs/design/dd-cdt-model.md`
`docs/design/dd-cdt-modularity.md`

**Source**
`examples/mobility_bologna/mobility_bologna.py`

<br>

*Questions & discussion*